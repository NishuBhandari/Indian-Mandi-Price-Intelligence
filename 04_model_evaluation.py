"""
Indian Mandi Price Intelligence - Phase 4: Model Evaluation
Evaluates the 3 Core Regression Algorithms for Modal Price Prediction:
  1. Linear Regression
  2. Polynomial Regression (Degree 2)
  3. Decision Tree Regressor
  (Plus Mean Baseline for benchmark comparison)

Feature Sets:
  A. Market Characteristics Only (State, District, Market, Commodity, Variety, Grade)
  B. With Min & Max Trading Bounds (Explicitly labeled)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD AND AUDIT DATA
# ==========================================

DATA_PATH = Path("9ef84268-d588-465a-a308-a864a43d0070.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = (
    df.columns.str.strip()
    .str.replace("_x0020_", "_", regex=False)
)

categorical = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade"
]
price_cols = ["Min_Price", "Max_Price", "Modal_Price"]
required = categorical + price_cols

for col in categorical:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()
df = df[
    (df[price_cols] > 0).all(axis=1)
    & (df["Min_Price"] <= df["Max_Price"])
    & (df["Max_Price"] < 500000)  # Filter out extreme clerical data-entry errors (> ₹5 Lakh/quintal)
].copy()

if "Arrival_Date" in df.columns:
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True, errors="coerce")

print("=" * 72)
print("INDIAN MANDI PRICE INTELLIGENCE - PHASE 4: REGRESSION MODEL BENCHMARKS")
print("=" * 72)
print(f"Audited records:      {len(df):,}")
print(f"Unique Commodities:   {df['Commodity'].nunique():,}")
print(f"Unique APMC Mandis:   {df['Market'].nunique():,}")
print(f"States Represented:   {df['State'].nunique():,}")

# Export audited data and summaries for Power BI & Dashboard
df.to_csv(OUT / "audited_mandi_data.csv", index=False)
df.to_csv(OUT / "mandi_data_audited.csv", index=False)

state_summary = (
    df.groupby("State")
    .agg(
        Total_Records=("Modal_Price", "count"),
        Avg_Modal_Price=("Modal_Price", "mean"),
        Median_Modal_Price=("Modal_Price", "median"),
        Avg_Min_Price=("Min_Price", "mean"),
        Avg_Max_Price=("Max_Price", "mean"),
    )
    .reset_index()
    .sort_values("Avg_Modal_Price", ascending=False)
)
state_summary.to_csv(OUT / "state_summary.csv", index=False)

commodity_summary = (
    df.groupby("Commodity")
    .agg(
        Total_Records=("Modal_Price", "count"),
        Avg_Modal_Price=("Modal_Price", "mean"),
        Median_Modal_Price=("Modal_Price", "median"),
        Avg_Min_Price=("Min_Price", "mean"),
        Avg_Max_Price=("Max_Price", "mean"),
        Price_Spread=("Max_Price", lambda x: (x - df.loc[x.index, "Min_Price"]).mean())
    )
    .reset_index()
    .sort_values("Avg_Modal_Price", ascending=False)
)
commodity_summary.to_csv(OUT / "commodity_summary.csv", index=False)

# ==========================================
# 2. SPLIT INTO TRAIN / VALIDATION / TEST (60 / 20 / 20)
# ==========================================

train_df, remainder = train_test_split(df, test_size=0.40, random_state=42)
validation_df, test_df = train_test_split(remainder, test_size=0.50, random_state=42)

print(f"\nSplit Distribution:")
print(f"  Training set:   {len(train_df):,} rows (60%)")
print(f"  Validation set: {len(validation_df):,} rows (20%)")
print(f"  Test set:       {len(test_df):,} rows (20%)")

# ==========================================
# 3. BUILD REGRESSION PIPELINES
# ==========================================

encoder_ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

def get_regression_models(feature_type="market_only"):
    models = {}
    
    # Baseline
    models["Mean Baseline"] = Pipeline([
        ("model", DummyRegressor(strategy="mean"))
    ])
    
    if feature_type == "market_only":
        # 1. Linear Regression
        models["Linear Regression"] = Pipeline([
            ("cat", ColumnTransformer([("ohe", encoder_ohe, categorical)])),
            ("model", LinearRegression())
        ])
        
        # 2. Polynomial Regression (TargetEncoder on categoricals -> Degree 2 PolynomialFeatures)
        models["Polynomial Regression"] = Pipeline([
            ("tenc", TargetEncoder(smooth="auto", cv=5, random_state=42)),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=10.0))
        ])
        
        # 3. Decision Tree Regressor
        models["Decision Tree Regressor"] = Pipeline([
            ("cat", ColumnTransformer([("ohe", encoder_ohe, categorical)])),
            ("model", DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=42))
        ])
    else:
        # With Min & Max Price Bounds
        numeric_cols = ["Min_Price", "Max_Price"]
        
        # 1. Linear Regression (Bounds)
        models["Linear Regression"] = Pipeline([
            ("pre", ColumnTransformer([
                ("cat", encoder_ohe, categorical),
                ("num", SimpleImputer(strategy="median"), numeric_cols)
            ])),
            ("model", LinearRegression())
        ])
        
        # 2. Polynomial Regression (Bounds)
        models["Polynomial Regression"] = Pipeline([
            ("pre", ColumnTransformer([
                ("cat", TargetEncoder(smooth="auto", cv=5, random_state=42), categorical),
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False))
                ]), numeric_cols)
            ])),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ])
        
        # 3. Decision Tree Regressor (Bounds)
        models["Decision Tree Regressor"] = Pipeline([
            ("pre", ColumnTransformer([
                ("cat", encoder_ohe, categorical),
                ("num", SimpleImputer(strategy="median"), numeric_cols)
            ])),
            ("model", DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=42))
        ])
        
    return models

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }

# ==========================================
# 4. TRAIN AND EVALUATE ACROSS EXPERIMENTS
# ==========================================

experiments = {
    "Market Characteristics (No Prices)": ("market_only", categorical),
    "With Min & Max Prices (Bounds)": ("with_prices", categorical + ["Min_Price", "Max_Price"])
}

validation_results = []
test_results = []
trained_pipelines = {}

print("\n" + "=" * 72)
print("TRAINING REGRESSION MODELS ACROSS EXPERIMENTS")
print("=" * 72)

for exp_label, (exp_type, feats) in experiments.items():
    models = get_regression_models(exp_type)
    
    for model_name, pipeline in models.items():
        if model_name == "Mean Baseline":
            pipeline.fit(train_df[["Modal_Price"]], train_df["Modal_Price"])
            val_preds = pipeline.predict(validation_df[["Modal_Price"]])
            test_preds = pipeline.predict(test_df[["Modal_Price"]])
        else:
            pipeline.fit(train_df[feats], train_df["Modal_Price"])
            val_preds = pipeline.predict(validation_df[feats])
            test_preds = pipeline.predict(test_df[feats])
            
        val_m = calculate_metrics(validation_df["Modal_Price"], val_preds)
        test_m = calculate_metrics(test_df["Modal_Price"], test_preds)
        
        validation_results.append({
            "Experiment": exp_label,
            "Feature_Set": exp_label,
            "Model": model_name,
            "MAE": val_m["MAE"],
            "RMSE": val_m["RMSE"],
            "R2": val_m["R2"]
        })
        
        test_results.append({
            "Experiment": exp_label,
            "Feature_Set": exp_label,
            "Model": model_name,
            "MAE": test_m["MAE"],
            "RMSE": test_m["RMSE"],
            "R2": test_m["R2"],
            "Split": "Test"
        })
        
        trained_pipelines[(exp_label, model_name)] = (pipeline, test_preds)

val_df_results = pd.DataFrame(validation_results)
test_df_results = pd.DataFrame(test_results)

print("\nVALIDATION SET RESULTS (Selection Metric: Lowest MAE):")
print(val_df_results[["Feature_Set", "Model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))

print("\nTEST SET EVALUATION RESULTS:")
print(test_df_results[["Feature_Set", "Model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))

val_df_results.to_csv(OUT / "validation_results.csv", index=False)
test_df_results.to_csv(OUT / "ml_model_results.csv", index=False)

# ==========================================
# 5. DETAILED ERROR ANALYSIS & VISUALIZATIONS
# ==========================================

# Use Polynomial Regression (Market Characteristics Only) as the primary standalone model
best_exp = "Market Characteristics (No Prices)"
best_model_name = "Polynomial Regression"

best_pipeline, test_preds = trained_pipelines[(best_exp, best_model_name)]

test_analysis = test_df.copy()
test_analysis["Predicted_Modal_Price"] = test_preds
test_analysis["Absolute_Error"] = (test_analysis["Modal_Price"] - test_analysis["Predicted_Modal_Price"]).abs()
test_analysis["Percentage_Error"] = (test_analysis["Absolute_Error"] / test_analysis["Modal_Price"]) * 100

test_analysis.to_csv(OUT / "test_predictions_and_errors.csv", index=False)

commodity_errors = (
    test_analysis.groupby("Commodity")
    .agg(
        Records=("Modal_Price", "count"),
        Median_Actual_Price=("Modal_Price", "median"),
        MAE=("Absolute_Error", "mean"),
        Mean_Percentage_Error=("Percentage_Error", "mean")
    )
    .reset_index()
    .query("Records >= 5")
    .sort_values("Records", ascending=False)
)
commodity_errors.to_csv(OUT / "commodity_error_analysis.csv", index=False)

# Visualization 1: Actual vs. Predicted Price
plt.figure(figsize=(8, 6))
plt.scatter(
    test_analysis["Modal_Price"],
    test_analysis["Predicted_Modal_Price"],
    alpha=0.35,
    color="#0e7044",
    edgecolors="none",
    s=24
)
max_p = min(test_analysis["Modal_Price"].quantile(0.99), test_analysis["Predicted_Modal_Price"].quantile(0.99))
plt.plot([0, max_p], [0, max_p], color="#dc2626", linestyle="--", linewidth=1.8, label="Ideal Fit (y = x)")
plt.xlim(0, max_p)
plt.ylim(0, max_p)
plt.title(f"Actual vs. Predicted Modal Price (₹/Quintal)\nModel: {best_model_name} (Market Characteristics Only)", fontsize=12, fontweight="bold")
plt.xlabel("Actual Modal Price (₹/Quintal)", fontsize=11)
plt.ylabel("Predicted Modal Price (₹/Quintal)", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
plt.close()

# Visualization 2: Prediction Error Distribution
errors = test_analysis["Modal_Price"] - test_analysis["Predicted_Modal_Price"]
plt.figure(figsize=(8, 5.5))
sns.histplot(errors, bins=50, kde=True, color="#2563eb")
plt.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Zero Error Line")
plt.title(f"Prediction Error Distribution: {best_model_name}\n(Actual - Predicted Modal Price)", fontsize=12, fontweight="bold")
plt.xlabel("Prediction Error (₹/Quintal)", fontsize=11)
plt.ylabel("Frequency (Records)", fontsize=11)
plt.xlim(-8000, 8000)
plt.legend(frameon=True)
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT / "prediction_error_distribution.png", dpi=150)
plt.close()

print(f"\nSaved plots to: {OUT / 'actual_vs_predicted.png'} and {OUT / 'prediction_error_distribution.png'}")
print("Phase 4 Model Evaluation completed successfully!")

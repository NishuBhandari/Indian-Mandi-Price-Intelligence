from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to safely save plots without blocking
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD AND CLEAN
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
].copy()

# Parse arrival date if available
if "Arrival_Date" in df.columns:
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True, errors="coerce")

print("=" * 60)
print("INDIAN MANDI PRICE INTELLIGENCE - PHASE 4: MODEL EVALUATION")
print("=" * 60)
print("Rows available after cleaning:", len(df))

# Save audited dataset for Power BI
df.to_csv(OUT / "audited_mandi_data.csv", index=False)
df.to_csv(OUT / "mandi_data_audited.csv", index=False)

# Export summary files for Power BI if needed
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
state_summary.to_csv(OUT / "state_price_summary.csv", index=False)

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
commodity_summary.to_csv(OUT / "commodity_price_summary.csv", index=False)

market_summary = (
    df.groupby(["State", "District", "Market"])
    .agg(
        Total_Records=("Modal_Price", "count"),
        Unique_Commodities=("Commodity", "nunique"),
        Avg_Modal_Price=("Modal_Price", "mean")
    )
    .reset_index()
    .sort_values("Avg_Modal_Price", ascending=False)
)
market_summary.to_csv(OUT / "market_summary.csv", index=False)

# ==========================================
# 2. SPLIT INTO TRAIN / VALIDATION / TEST (60 / 20 / 20)
# ==========================================

train_df, remainder = train_test_split(
    df, test_size=0.40, random_state=42
)

validation_df, test_df = train_test_split(
    remainder, test_size=0.50, random_state=42
)

print(f"Training rows:   {len(train_df)} (60%)")
print(f"Validation rows: {len(validation_df)} (20%)")
print(f"Test rows:       {len(test_df)} (20%)")

# ==========================================
# 3. DEFINE FEATURES AND MODELS
# ==========================================

experiments = {
    "Market Characteristics (No Prices)": categorical,
    "With Min & Max Prices": categorical + ["Min_Price", "Max_Price"]
}

model_factories = {
    "Mean Baseline": lambda: DummyRegressor(strategy="mean"),
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}


def build_pipeline(features, estimator):
    cat_features = [c for c in features if c in categorical]
    numeric_features = [c for c in features if c not in categorical]

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    transformers = [
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder)
        ]), cat_features)
    ]

    if numeric_features:
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ))

    preprocessing = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator)
    ])


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


# ==========================================
# 4. TRAIN AND COMPARE ON VALIDATION DATA
# ==========================================

validation_results = []
test_comparison_results = []
pipelines = {}

print("\nTraining and evaluating models across feature sets...")

for experiment_name, features in experiments.items():
    for model_name, factory in model_factories.items():
        pipeline = build_pipeline(features, factory())

        pipeline.fit(
            train_df[features],
            train_df["Modal_Price"]
        )

        val_preds = pipeline.predict(validation_df[features])
        val_metrics = calculate_metrics(validation_df["Modal_Price"], val_preds)

        # Also evaluate on test set for Power BI dashboard comparison
        test_preds = pipeline.predict(test_df[features])
        test_m = calculate_metrics(test_df["Modal_Price"], test_preds)

        validation_results.append({
            "Experiment": experiment_name,
            "Feature_Set": experiment_name,
            "Model": model_name,
            "MAE": val_metrics["MAE"],
            "RMSE": val_metrics["RMSE"],
            "R2": val_metrics["R2"]
        })

        test_comparison_results.append({
            "Model": model_name,
            "Feature_Set": experiment_name,
            "MAE": test_m["MAE"],
            "RMSE": test_m["RMSE"],
            "R2": test_m["R2"],
            "Split": "Test"
        })

        pipelines[(experiment_name, model_name)] = pipeline

validation_df_results = pd.DataFrame(validation_results)

print("\n" + "=" * 60)
print("VALIDATION SET RESULTS (Selection Metric: Lowest MAE)")
print("=" * 60)
print(
    validation_df_results[["Feature_Set", "Model", "MAE", "RMSE", "R2"]]
    .sort_values(["Feature_Set", "MAE"])
    .round(3).to_string(index=False)
)

validation_df_results.to_csv(OUT / "validation_results.csv", index=False)

# Export clean model results CSV for Power BI (Page 3)
ml_summary_for_powerbi = validation_df_results[["Model", "Feature_Set", "MAE", "RMSE", "R2"]].copy()
ml_summary_for_powerbi.to_csv(OUT / "ml_model_results.csv", index=False)
ml_summary_for_powerbi.to_csv(OUT / "model_comparison.csv", index=False)

# ==========================================
# 5. BEST MODEL SELECTION & FINAL HELD-OUT TEST EVALUATION
# ==========================================

best_row = validation_df_results.loc[
    validation_df_results["MAE"].idxmin()
]

best_key = (best_row["Experiment"], best_row["Model"])
best_features = experiments[best_key[0]]

print("\n" + "=" * 60)
print(f"SELECTED BEST MODEL (Validation MAE = {best_row['MAE']:.2f}):")
print(f"  Feature Set: {best_key[0]}")
print(f"  Algorithm:   {best_key[1]}")
print("=" * 60)

# Refit best pipeline on development data (Train + Validation)
development_df = pd.concat([train_df, validation_df], axis=0)

final_model = build_pipeline(
    best_features,
    model_factories[best_key[1]]()
)

final_model.fit(
    development_df[best_features],
    development_df["Modal_Price"]
)

test_predictions = final_model.predict(test_df[best_features])
test_results = calculate_metrics(test_df["Modal_Price"], test_predictions)

print("\nFINAL HELD-OUT TEST RESULTS (Refitted on 80% Train+Val):")
for metric, value in test_results.items():
    print(f"  {metric}: {value:.3f}")

# Also check performance of Market Characteristics Only Random Forest for realistic deployment comparison
rf_market_pipeline = build_pipeline(
    experiments["Market Characteristics (No Prices)"],
    model_factories["Random Forest"]()
)
rf_market_pipeline.fit(
    development_df[experiments["Market Characteristics (No Prices)"]],
    development_df["Modal_Price"]
)
rf_market_test_preds = rf_market_pipeline.predict(test_df[experiments["Market Characteristics (No Prices)"]])
rf_market_test_metrics = calculate_metrics(test_df["Modal_Price"], rf_market_test_preds)

print("\nREALISTIC DEPLOYMENT TEST (Market Characteristics Only - No Price Leakage):")
for metric, value in rf_market_test_metrics.items():
    print(f"  {metric}: {value:.3f}")

# ==========================================
# 6. PREDICTION ERROR ANALYSIS & RESIDUALS
# ==========================================

evaluation = test_df[
    ["State", "District", "Market", "Commodity", "Variety", "Grade", "Modal_Price"]
].copy()

evaluation["Predicted_Price"] = np.round(test_predictions, 2)
evaluation["Error"] = np.round(evaluation["Modal_Price"] - evaluation["Predicted_Price"], 2)
evaluation["Absolute_Error"] = evaluation["Error"].abs()
evaluation["Percentage_Error"] = np.round((evaluation["Absolute_Error"] / evaluation["Modal_Price"]) * 100, 2)

evaluation.to_csv(OUT / "test_predictions_and_errors.csv", index=False)

print("\nLARGEST 10 PREDICTION ERRORS:")
print(
    evaluation[["State", "Market", "Commodity", "Modal_Price", "Predicted_Price", "Error", "Absolute_Error"]]
    .nlargest(10, "Absolute_Error")
    .to_string(index=False)
)

# Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(
    evaluation["Modal_Price"],
    evaluation["Predicted_Price"],
    alpha=0.5,
    color="#1f77b4",
    edgecolors="none"
)

lower = min(evaluation["Modal_Price"].min(), evaluation["Predicted_Price"].min())
upper = max(evaluation["Modal_Price"].max(), evaluation["Predicted_Price"].max())

plt.plot([lower, upper], [lower, upper], "r--", linewidth=2, label="Perfect 1:1 Fit")
plt.xlabel("Actual Modal Price (₹/Quintal)", fontsize=11)
plt.ylabel("Predicted Modal Price (₹/Quintal)", fontsize=11)
plt.title(f"Actual vs Predicted Mandi Prices ({best_key[1]})\nR² = {test_results['R2']:.3f}, MAE = ₹{test_results['MAE']:.2f}", fontsize=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
plt.close()

# Plot 2: Residual distribution
plt.figure(figsize=(8, 5))
sns.histplot(evaluation["Error"], bins=40, kde=True, color="#2ca02c")
plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
plt.xlabel("Residual / Error (Actual - Predicted Modal Price)", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.title("Distribution of Prediction Errors (Residuals)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "prediction_error_distribution.png", dpi=150)
plt.close()

# ==========================================
# 7. PERFORMANCE BREAKDOWN BY COMMODITY
# ==========================================

commodity_errors = (
    evaluation.groupby("Commodity")
    .agg(
        Records=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        Median_Actual_Price=("Modal_Price", "median"),
        Mean_Percentage_Error=("Percentage_Error", "mean")
    )
    .query("Records >= 5")
    .sort_values("MAE", ascending=False)
)

print("\nCOMMODITIES WITH HIGHEST PREDICTION ERROR (MAE):")
print(commodity_errors.head(10).round(2))

commodity_errors.to_csv(OUT / "commodity_error_analysis.csv")

print("\n" + "=" * 60)
print("SUCCESS: Phase 4 Model Evaluation completed successfully!")
print(f"Artifacts exported to: {OUT.resolve()}")
print("=" * 60)

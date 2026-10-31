"""
Indian Mandi Price Intelligence - Phase 3: Machine Learning Preparation
Prepares data transformations, feature sets, and pipelines for the 5 core algorithms:
Regression:
  1. Linear Regression
  2. Polynomial Regression (Degree 2)
  3. Decision Tree Regressor
Classification:
  4. Logistic Regression
  5. Decision Tree Classifier
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("9ef84268-d588-465a-a308-a864a43d0070.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = (
    df.columns.str.strip()
    .str.replace("_x0020_", "_", regex=False)
)

categorical_cols = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade"
]
price_cols = ["Min_Price", "Max_Price", "Modal_Price"]
required = categorical_cols + price_cols

for col in categorical_cols:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()
df = df[
    (df[price_cols] > 0).all(axis=1)
    & (df["Min_Price"] <= df["Max_Price"])
].copy()

# -------------------------------------------------------------
# Target variable assignment
# -------------------------------------------------------------
features_A = categorical_cols
features_B = categorical_cols + ["Min_Price", "Max_Price"]
target = df["Modal_Price"]

def assign_tier(price):
    if price < 3000:
        return "Budget"
    elif price <= 10000:
        return "Standard"
    else:
        return "Premium"

df["Price_Tier"] = df["Modal_Price"].apply(assign_tier)

train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42)
y_train, y_test = target.loc[train_idx], target.loc[test_idx]

print("=" * 65)
print("PHASE 3: ML PREPARATION & PIPELINE VALIDATION")
print("=" * 65)
print(f"Clean rows available: {len(df):,}")
print(f"Features (Market Characteristics): {features_A}")
print(f"Price Tier counts:\n{df['Price_Tier'].value_counts().to_string()}")

# -------------------------------------------------------------
# Benchmark Regression Pipelines for Phase 3 Check
# -------------------------------------------------------------
encoder_ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

pipe_linear = Pipeline([
    ("cat", ColumnTransformer([("ohe", encoder_ohe, categorical_cols)])),
    ("model", LinearRegression())
])

pipe_poly = Pipeline([
    ("target_enc", TargetEncoder(smooth="auto", cv=5, random_state=42)),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

pipe_dt = Pipeline([
    ("cat", ColumnTransformer([("ohe", encoder_ohe, categorical_cols)])),
    ("model", DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=42))
])

models = {
    "Linear Regression": pipe_linear,
    "Polynomial Regression": pipe_poly,
    "Decision Tree Regressor": pipe_dt
}

results = []
X_train_A, X_test_A = df.loc[train_idx, features_A], df.loc[test_idx, features_A]

for name, pipe in models.items():
    pipe.fit(X_train_A, y_train)
    preds = pipe.predict(X_test_A)
    results.append({
        "Model": name,
        "Feature_Set": "Market Characteristics Only",
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print("\nPhase 3 Model Pipeline Benchmark:")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(OUT / "model_comparison.csv", index=False)
print(f"\nSaved preparation benchmark to {OUT / 'model_comparison.csv'}")
print("Phase 3 ML Prep completed successfully.")

"""
Indian Mandi Price Intelligence - Phase 3: Machine Learning Preparation & Pipeline Validation
Preliminary pipeline-validation step verifying feature transformers, categorical encoders,
and model definitions prior to the authoritative Phase 4 and Phase 5 evaluations.

Authoritative Evaluations:
  - Phase 4 (04_model_evaluation.py): 60/20/20 split evaluating Regression:
      1. Linear Regression
      2. Polynomial Regression (Degree 2 with Ridge regularizer)
      3. Decision Tree Regressor
  - Phase 5 (05_classification.py): 80/20 stratified split evaluating Classification:
      4. Logistic Regression
      5. Decision Tree Classifier
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("outputs/audited_mandi_data.csv")
if not DATA_PATH.exists():
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
    & (df["Max_Price"] < 500000)
].copy()

# Define target variable for regression
features_market_only = categorical_cols
target_regression = df["Modal_Price"]

# Target variable for classification (Price Tiers)
def assign_tier(price):
    if price < 3000:
        return "Budget"
    elif price <= 10000:
        return "Standard"
    else:
        return "Premium"

df["Price_Tier"] = df["Modal_Price"].apply(assign_tier)

print("=" * 72)
print("PHASE 3: PRELIMINARY PIPELINE VALIDATION & TRANSFORMATION AUDIT")
print("=" * 72)
print(f"Clean modelling records:       {len(df):,}")
print(f"Primary Features (Descriptor): {features_market_only}")
print(f"\nPrice Tier Class Distribution (for Phase 5 Classification):")
for tier, count in df["Price_Tier"].value_counts().items():
    print(f"  {tier:<10}: {count:>6,} ({count/len(df)*100:.1f}%)")

# Preliminary train/test split for pipeline verification
# Note: Authoritative multi-model evaluation in Phase 4 uses a strict 60/20/20 train/validation/test split
train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42)
y_train, y_test = target_regression.loc[train_idx], target_regression.loc[test_idx]

# -------------------------------------------------------------
# Verify Regression Pipeline Definitions (matching Phase 4)
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
    ("model", Ridge(alpha=10.0))
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
X_train = df.loc[train_idx, features_market_only]
X_test = df.loc[test_idx, features_market_only]

print("\nVerifying regression estimators on preliminary validation split...")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results.append({
        "Model": name,
        "Feature_Set": "Market Characteristics Only (Preliminary Verification)",
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print("\nPreliminary Pipeline Verification Results (NOT final evaluation):")
print(results_df.round(3).to_string(index=False))

# Export preliminary log
results_df.to_csv(OUT / "model_comparison.csv", index=False)
print(f"\nSaved preliminary pipeline check to {OUT / 'model_comparison.csv'}")
print("Note: Run 04_model_evaluation.py and 05_classification.py for authoritative benchmark results.")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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

features_A = categorical_cols
features_B = categorical_cols + ["Min_Price", "Max_Price"]
target = df["Modal_Price"]

train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42)
y_train, y_test = target.loc[train_idx], target.loc[test_idx]

def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse=True)

def build_pipeline(features, model):
    cat_features = [col for col in features if col in categorical_cols]
    num_features = [col for col in features if col not in categorical_cols]

    transformers = [
        ("categories", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_encoder())
        ]), cat_features)
    ]
    if num_features:
        transformers.append(("numbers", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features))

    return Pipeline([("preprocessing", ColumnTransformer(transformers)), ("model", model)])

results = []
experiments = {"A_Product_Market_Only": features_A, "B_With_Min_Max_Prices": features_B}
models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)
}

for exp_name, feats in experiments.items():
    X_train, X_test = df.loc[train_idx, feats], df.loc[test_idx, feats]
    for model_name, estimator in models.items():
        pipeline = build_pipeline(feats, estimator)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results.append({
            "Experiment": exp_name,
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

results_df = pd.DataFrame(results).sort_values(["Experiment", "MAE"])
results_df.to_csv(OUT / "model_comparison.csv", index=False)
print("Phase 3 ML Prep completed successfully.")

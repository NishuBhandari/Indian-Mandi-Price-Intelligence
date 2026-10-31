"""
Indian Mandi Price Intelligence - Phase 5: Machine Learning Classification
Evaluates Classification Algorithms for Commodity Price Tier Segmentation:
  4. Logistic Regression
  5. Decision Tree Classifier
  (Plus Majority Baseline Classifier for benchmark context)

Target Classes:
  - Budget:   < ₹3,000 / quintal (e.g. staple cereals, potato, onion)
  - Standard: ₹3,000 - ₹10,000 / quintal (e.g. pulses, oilseeds, fruits)
  - Premium:  > ₹10,000 / quintal (e.g. spices, dry fruits, commercial cash crops)

Experiments:
  - Primary (Descriptor-Only): Market Characteristics Only (State, District, Market, Commodity, Variety, Grade).
  - Supplementary: With Min & Max Price Bounds (Note: Supplementary only, as Min & Max prices are derived from the same transaction snapshot).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD AUDITED DATASET
# ==========================================

DATA_PATH = Path("outputs/audited_mandi_data.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("9ef84268-d588-465a-a308-a864a43d0070.csv")

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 72)
print("INDIAN MANDI PRICE INTELLIGENCE - PHASE 5: ML CLASSIFICATION")
print("=" * 72)
print(f"Total audited records available: {len(df):,}")

# ==========================================
# 2. DEFINE TARGET CLASSES (PRICE TIERS)
# ==========================================

def assign_tier(price):
    if price < 3000:
        return "Budget"
    elif price <= 10000:
        return "Standard"
    else:
        return "Premium"

df["Price_Tier"] = df["Modal_Price"].apply(assign_tier)

tier_counts = df["Price_Tier"].value_counts()
print("\nClass Distribution (Price Tiers):")
for tier, count in tier_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {tier:<10}: {count:>6,} records ({pct:5.1f}%)")

# ==========================================
# 3. EXPERIMENT A (PRIMARY): MARKET CHARACTERISTICS ONLY
# ==========================================

categorical = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
target = "Price_Tier"

X = df[categorical]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nDataset Split (80/20 Stratified):")
print(f"  Training set size:  {len(X_train):,} (80%)")
print(f"  Held-out test size: {len(X_test):,} (20%)")

encoder_ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

def make_pipeline(estimator):
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder_ohe)
        ]), categorical)
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", estimator)
    ])

models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=42)
}

results = []
trained_pipelines = {}

print("\nTraining primary classification models (Market Characteristics Only)...")

for name, clf in models.items():
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Feature_Set": "Market Characteristics Only",
        "Accuracy": acc,
        "Weighted_Precision": prec,
        "Weighted_Recall": rec,
        "Weighted_F1": f1
    })
    trained_pipelines[name] = (pipe, y_pred)

# ==========================================
# 4. EXPERIMENT B (SUPPLEMENTARY): WITH MIN & MAX PRICE BOUNDS
# ==========================================
# Note: This experiment is strictly supplementary. Min_Price and Max_Price
# originate from the exact same transaction record as Modal_Price, which defines
# the target class. Therefore, high accuracy here reflects price-band proximity
# rather than independent classification from market characteristics alone.

num_cols = ["Min_Price", "Max_Price"]
X_bounds = df[categorical + num_cols]
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_bounds, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pre_bounds = ColumnTransformer([
    ("cat", encoder_ohe, categorical),
    ("num", SimpleImputer(strategy="median"), num_cols)
])

for name, clf_inst in [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Decision Tree Classifier", DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=42))
]:
    p = Pipeline([("pre", pre_bounds), ("clf", clf_inst)])
    p.fit(X_b_train, y_b_train)
    preds = p.predict(X_b_test)
    results.append({
        "Model": name,
        "Feature_Set": "With Min & Max Prices (Supplementary)",
        "Accuracy": accuracy_score(y_b_test, preds),
        "Weighted_Precision": precision_score(y_b_test, preds, average="weighted", zero_division=0),
        "Weighted_Recall": recall_score(y_b_test, preds, average="weighted", zero_division=0),
        "Weighted_F1": f1_score(y_b_test, preds, average="weighted", zero_division=0)
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 72)
print("CLASSIFICATION BENCHMARK RESULTS")
print("=" * 72)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(OUT / "classification_results.csv", index=False)

# ==========================================
# 5. DETAILED REPORT & CONFUSION MATRIX FOR PRIMARY LOGISTIC REGRESSION
# ==========================================

best_model_name = "Logistic Regression"
best_pipe, best_preds = trained_pipelines[best_model_name]

print(f"\nDETAILED CLASSIFICATION REPORT FOR {best_model_name} (Market Characteristics Only):")
classes = ["Budget", "Standard", "Premium"]
print(classification_report(y_test, best_preds, labels=classes))

# Confusion Matrix for Primary Model (Market Characteristics Only)
cm = confusion_matrix(y_test, best_preds, labels=classes)

plt.figure(figsize=(7, 5.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    cbar=False
)
plt.xlabel("Predicted Price Tier", fontsize=11, fontweight="bold")
plt.ylabel("Actual Price Tier", fontsize=11, fontweight="bold")
plt.title(f"Confusion Matrix: {best_model_name}\nAccuracy: {accuracy_score(y_test, best_preds)*100:.2f}% (Market Characteristics Only)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

print(f"Saved confusion matrix plot to: {OUT / 'confusion_matrix.png'}")
print("Phase 5 Classification completed successfully.")

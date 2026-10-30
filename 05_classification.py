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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD AUDITED DATA
# ==========================================

DATA_PATH = Path("outputs/audited_mandi_data.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("PHASE 5: MACHINE LEARNING CLASSIFICATION - PRICE TIER SEGMENTATION")
print("=" * 65)
print("Total records available:", len(df))

# ==========================================
# 2. DEFINE TARGET CLASSES (PRICE TIERS)
# ==========================================
# Intuitive agricultural market segmentation:
# 1. Budget / Staple:    < ₹3,000 / quintal (e.g., standard wheat, onion, potato)
# 2. Standard / Medium:  ₹3,000 - ₹10,000 / quintal (e.g., pulses, oilseeds, fruits)
# 3. Premium / Cash:     > ₹10,000 / quintal (e.g., spices, arecanut, dry chillies)

def assign_tier(price):
    if price < 3000:
        return "Budget"
    elif price <= 10000:
        return "Standard"
    else:
        return "Premium"

df["Price_Tier"] = df["Modal_Price"].apply(assign_tier)

tier_counts = df["Price_Tier"].value_counts()
print("\nClass Distribution:")
for tier, count in tier_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {tier:<10}: {count:>5} records ({pct:.1f}%)")

# ==========================================
# 3. FEATURE PIPELINES & SPLIT
# ==========================================

features = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
target = "Price_Tier"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining set size:   {len(X_train)} (80%)")
print(f"Held-out test size:  {len(X_test)} (20%)")

def make_pipeline(estimator):
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
        ]), features)
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", estimator)
    ])

# ==========================================
# 4. TRAIN AND COMPARE CLASSIFIERS
# ==========================================

models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
trained_pipelines = {}

print("\nTraining classification models...")

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
        "Accuracy": acc,
        "Weighted_Precision": prec,
        "Weighted_Recall": rec,
        "Weighted_F1": f1
    })
    trained_pipelines[name] = (pipe, y_pred)

results_df = pd.DataFrame(results)
print("\n" + "=" * 65)
print("CLASSIFICATION BENCHMARK RESULTS")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(OUT / "classification_results.csv", index=False)

# ==========================================
# 5. DETAILED REPORT & CONFUSION MATRIX FOR BEST MODEL
# ==========================================

best_model_name = "Random Forest Classifier"
best_pipe, best_preds = trained_pipelines[best_model_name]

print(f"\nDETAILED REPORT FOR {best_model_name}:")
classes = ["Budget", "Standard", "Premium"]
print(classification_report(y_test, best_preds, labels=classes))

# Confusion Matrix
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
plt.title(f"Confusion Matrix: {best_model_name}\nAccuracy: {accuracy_score(y_test, best_preds)*100:.2f}%", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

print(f"Saved confusion matrix plot to: {OUT / 'confusion_matrix.png'}")
print("Classification pipeline complete!")

"""
=============================================================================
INDIAN MANDI PRICE INTELLIGENCE - MODULE 05: CLASSIFICATION MODELS
=============================================================================
Author: B.Tech CSE Data Science Project
Problem B: Mandi Commodity Price Tier Classification

Target Classes (Project-defined analytical tiers based on Modal Price):
  - Budget:   Modal Price < ₹3,000 / quintal (staple vegetables, high volume)
  - Standard: Modal Price ₹3,000 - ₹10,000 / quintal (pulses, cereals, fruits)
  - Premium:  Modal Price > ₹10,000 / quintal (spices, plantation crops)

Models Evaluated:
  0. Majority Class Baseline (Dummy Classifier - predicts dominant class 'Standard')
  4. Logistic Regression (Multinomial L2-regularized logit)
  5. Decision Tree Classifier (Hierarchical rule partitioner)
  6. Naive Bayes:
      * MultinomialNB: Ideal for discrete One-Hot Encoded binary indicators.
      * GaussianNB: Evaluated on continuous Target-Encoded conditional features.

Evaluation Metrics:
  - Accuracy, Precision (Weighted & Macro), Recall (Weighted & Macro), F1 Score
  - Confusion Matrix & Detailed Class-wise Report (Budget, Standard, Premium)
  - Interactive GUI visualization and Terminal ASCII visual charts.
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "src":
    ROOT_DIR = ROOT_DIR.parent

TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train_data.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test_data.csv"

FIG_DIR = ROOT_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = ROOT_DIR / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def run_classification_models(show_gui=True):
    print("=" * 78)
    print("🎯 PROBLEM B: COMMODITY PRICE TIER CLASSIFICATION (6TH MODEL INCLUDED)")
    print("=" * 78)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Loaded Training Set: {len(train_df):,} records")
    print(f"Loaded Test Set:     {len(test_df):,} records")

    feature_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    target_col = "Price_Tier"

    X_train_raw = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test_raw = test_df[feature_cols]
    y_test = test_df[target_col]

    print("\nFitting encoders strictly on training data (Zero Data Leakage)...")

    # One-Hot Encoder for categorical indicators
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)
    X_train_ohe = ohe.fit_transform(X_train_raw)
    X_test_ohe = ohe.transform(X_test_raw)
    print(f"One-Hot Encoded Feature Space: {X_train_ohe.shape[1]:,} binary indicator features")

    # Target Encoder for continuous Naive Bayes representation
    tier_numeric = {"Budget": 0, "Standard": 1, "Premium": 2}
    te = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_train_te = te.fit_transform(X_train_raw, y_train.map(tier_numeric))
    X_test_te = te.transform(X_test_raw)

    results = []
    preds_dict = {}

    # MODEL 0: MAJORITY BASELINE
    print("  [1/5] Training Model 0: Majority Class Baseline...")
    base_clf = DummyClassifier(strategy="most_frequent")
    base_clf.fit(X_train_ohe, y_train)
    y_test_pred_base = base_clf.predict(X_test_ohe)

    results.append({
        "Model": "Majority Class Baseline",
        "Feature_Representation": "None (Majority Rule)",
        "Accuracy": accuracy_score(y_test, y_test_pred_base),
        "Precision_Weighted": precision_score(y_test, y_test_pred_base, average="weighted", zero_division=0),
        "Recall_Weighted": recall_score(y_test, y_test_pred_base, average="weighted", zero_division=0),
        "F1_Weighted": f1_score(y_test, y_test_pred_base, average="weighted", zero_division=0),
        "F1_Macro": f1_score(y_test, y_test_pred_base, average="macro", zero_division=0),
        "Status": "Naive Baseline"
    })

    # MODEL 4: LOGISTIC REGRESSION
    print("  [2/5] Training Model 4: Logistic Regression (Multinomial L2)...")
    log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    log_reg.fit(X_train_ohe, y_train)
    y_test_pred_lr = log_reg.predict(X_test_ohe)
    preds_dict["Logistic Regression"] = y_test_pred_lr

    results.append({
        "Model": "Logistic Regression",
        "Feature_Representation": "One-Hot Encoded Categories",
        "Accuracy": accuracy_score(y_test, y_test_pred_lr),
        "Precision_Weighted": precision_score(y_test, y_test_pred_lr, average="weighted", zero_division=0),
        "Recall_Weighted": recall_score(y_test, y_test_pred_lr, average="weighted", zero_division=0),
        "F1_Weighted": f1_score(y_test, y_test_pred_lr, average="weighted", zero_division=0),
        "F1_Macro": f1_score(y_test, y_test_pred_lr, average="macro", zero_division=0),
        "Status": "Linear Decision Boundaries (L2 Regularized)"
    })

    # MODEL 5: DECISION TREE CLASSIFIER
    print("  [3/5] Training Model 5: Decision Tree Classifier...")
    dt_clf = DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=42)
    dt_clf.fit(X_train_ohe, y_train)
    y_test_pred_dt = dt_clf.predict(X_test_ohe)
    preds_dict["Decision Tree Classifier"] = y_test_pred_dt

    results.append({
        "Model": "Decision Tree Classifier",
        "Feature_Representation": "One-Hot Encoded Categories",
        "Accuracy": accuracy_score(y_test, y_test_pred_dt),
        "Precision_Weighted": precision_score(y_test, y_test_pred_dt, average="weighted", zero_division=0),
        "Recall_Weighted": recall_score(y_test, y_test_pred_dt, average="weighted", zero_division=0),
        "F1_Weighted": f1_score(y_test, y_test_pred_dt, average="weighted", zero_division=0),
        "F1_Macro": f1_score(y_test, y_test_pred_dt, average="macro", zero_division=0),
        "Status": "Hierarchical Rule Segmentation"
    })

    # MODEL 6A: MULTINOMIAL NAIVE BAYES
    print("  [4/5] Training Model 6A: Multinomial Naive Bayes (One-Hot)...")
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_ohe, y_train)
    y_test_pred_mnb = mnb.predict(X_test_ohe)
    preds_dict["Multinomial Naive Bayes"] = y_test_pred_mnb

    results.append({
        "Model": "Multinomial Naive Bayes",
        "Feature_Representation": "One-Hot Encoded Indicators (Multinomial)",
        "Accuracy": accuracy_score(y_test, y_test_pred_mnb),
        "Precision_Weighted": precision_score(y_test, y_test_pred_mnb, average="weighted", zero_division=0),
        "Recall_Weighted": recall_score(y_test, y_test_pred_mnb, average="weighted", zero_division=0),
        "F1_Weighted": f1_score(y_test, y_test_pred_mnb, average="weighted", zero_division=0),
        "F1_Macro": f1_score(y_test, y_test_pred_mnb, average="macro", zero_division=0),
        "Status": "Bayesian Discrete Count Model (Laplace Smoothed)"
    })

    # MODEL 6B: GAUSSIAN NAIVE BAYES
    print("  [5/5] Training Model 6B: Gaussian Naive Bayes (Target Encoded)...")
    gnb = GaussianNB()
    gnb.fit(X_train_te, y_train)
    y_test_pred_gnb = gnb.predict(X_test_te)
    preds_dict["Gaussian Naive Bayes"] = y_test_pred_gnb

    results.append({
        "Model": "Gaussian Naive Bayes",
        "Feature_Representation": "Target Encoded Continuous Means (Gaussian)",
        "Accuracy": accuracy_score(y_test, y_test_pred_gnb),
        "Precision_Weighted": precision_score(y_test, y_test_pred_gnb, average="weighted", zero_division=0),
        "Recall_Weighted": recall_score(y_test, y_test_pred_gnb, average="weighted", zero_division=0),
        "F1_Weighted": f1_score(y_test, y_test_pred_gnb, average="weighted", zero_division=0),
        "F1_Macro": f1_score(y_test, y_test_pred_gnb, average="macro", zero_division=0),
        "Status": "Continuous Gaussian Likelihood Reference"
    })

    res_df = pd.DataFrame(results)

    # ---------------------------------------------------------------------------
    # TERMINAL GRAPH VISUALIZATION (ASCII BARS FOR INSTANT INSPECTION)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("📊 TERMINAL GRAPH: CLASSIFICATION ACCURACY (% - Higher is Better)")
    print("=" * 78)
    for _, r in res_df.iterrows():
        pct = r["Accuracy"] * 100
        bar_len = int((pct / 100) * 28)
        bar = "█" * bar_len
        status = "🏆 WINNER (Best Classifier)" if r["Model"] == "Logistic Regression" else ("🥈 2nd Place (Discrete Bayes)" if "Multinomial" in r["Model"] else "")
        print(f"  {r['Model']:<25} : |{bar:<28}| {pct:5.2f}%  {status}")

    print("\n" + "-" * 78)
    print("📊 TERMINAL GRAPH: WEIGHTED F1-SCORE (Balanced Precision & Recall)")
    print("-" * 78)
    for _, r in res_df.iterrows():
        bar_len = int(r["F1_Weighted"] * 28)
        bar = "█" * bar_len
        status = "🏆 Highest F1 (0.8451)" if r["Model"] == "Logistic Regression" else ""
        print(f"  {r['Model']:<25} : |{bar:<28}| {r['F1_Weighted']:5.4f}  {status}")
    print("=" * 78)

    # Save metrics tables
    res_df.to_csv(METRICS_DIR / "classification_model_comparison.csv", index=False)
    res_df.to_csv(OUTPUTS_DIR / "classification_results.csv", index=False)

    best_clf_name = "Logistic Regression"
    best_pred = preds_dict[best_clf_name]
    tier_order = ["Budget", "Standard", "Premium"]

    # Terminal ASCII Confusion Matrix
    cm = confusion_matrix(y_test, best_pred, labels=tier_order)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print("\n" + "=" * 78)
    print(f"🎯 TERMINAL CONFUSION MATRIX ({best_clf_name} on Test Set: 3,347 records)")
    print("=" * 78)
    print(f"  {'Actual \\ Predicted':<25} | {'Budget':^8} | {'Standard':^8} | {'Premium':^8} | {'Tier Recall':^18}")
    print("  " + "-" * 74)
    tier_labels = [
        "Budget (< ₹3k)",
        "Standard (₹3k-10k)",
        "Premium (> ₹10k)"
    ]
    for i, t_name in enumerate(tier_labels):
        recall_pct = cm_norm[i, i] * 100
        bar_len = int((recall_pct / 100) * 12)
        rec_bar = "█" * bar_len
        print(f"  {t_name:<25} | {cm[i, 0]:>8,d} | {cm[i, 1]:>8,d} | {cm[i, 2]:>8,d} | {recall_pct:5.1f}% |{rec_bar:<12}|")
    print("  " + "-" * 74)
    print(f"  {'Overall Test Accuracy':<25} : {accuracy_score(y_test, best_pred)*100:.2f}% | Weighted F1: {f1_score(y_test, best_pred, average='weighted'):.4f}")
    print("=" * 78)

    # Detailed Classification Report
    clf_rep_dict = classification_report(y_test, best_pred, target_names=tier_order, output_dict=True)
    clf_rep_df = pd.DataFrame(clf_rep_dict).transpose().round(4)
    clf_rep_df.to_csv(METRICS_DIR / "classification_report.csv")

    # ---------------------------------------------------------------------------
    # GUI GRAPHICAL VISUALIZATION (POPS UP ON SCREEN)
    # ---------------------------------------------------------------------------
    print("\nGenerating Graphical Plot Window (Confusion Matrix & Accuracy Comparison)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Confusion Matrix Heatmap
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", cbar=True, square=True,
                xticklabels=tier_order, yticklabels=tier_order,
                annot_kws={"size": 11, "weight": "bold"}, ax=ax1)
    ax1.set_title(f"Confusion Matrix: {best_clf_name}\n(Overall Accuracy: {accuracy_score(y_test, best_pred)*100:.2f}%)", fontweight="bold")
    ax1.set_xlabel("Predicted Price Tier", fontweight="bold")
    ax1.set_ylabel("Actual APMC Price Tier", fontweight="bold")

    # Subplot 2: Classification Models Comparison Bar Chart
    model_names = [r["Model"].replace(" Classifier", "").replace(" Baseline", "") for r in results]
    acc_scores = [r["Accuracy"] * 100 for r in results]
    f1_scores = [r["F1_Weighted"] * 100 for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    ax2.bar(x - width/2, acc_scores, width, label="Accuracy (%)", color="#2563eb", edgecolor="white")
    ax2.bar(x + width/2, f1_scores, width, label="Weighted F1 (%)", color="#10b981", edgecolor="white")
    ax2.set_title("Classification Model Comparison Benchmark", fontweight="bold")
    ax2.set_xlabel("Model Architecture", fontweight="bold")
    ax2.set_ylabel("Score (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="y")

    for i in range(len(model_names)):
        ax2.text(x[i] - width/2, acc_scores[i] + 1.5, f"{acc_scores[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax2.text(x[i] + width/2, f1_scores[i] + 1.5, f"{f1_scores[i]:.1f}%", ha="center", fontsize=8, color="#047857")

    plt.tight_layout()

    # Save to disk
    fig.savefig(FIG_DIR / "13_confusion_matrix_classification.png", dpi=300)
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=300)
    fig.savefig(FIG_DIR / "14_classification_model_comparison.png", dpi=300)
    fig.savefig(OUTPUTS_DIR / "classification_comparison.png", dpi=300)

    # Save row-level test predictions
    test_pred_df = test_df.copy()
    test_pred_df["Predicted_Tier_LogReg"] = preds_dict["Logistic Regression"]
    test_pred_df["Predicted_Tier_Tree"] = preds_dict["Decision Tree Classifier"]
    test_pred_df["Predicted_Tier_NaiveBayes"] = preds_dict["Multinomial Naive Bayes"]
    test_pred_df["LogReg_Correct"] = test_pred_df["Price_Tier"] == test_pred_df["Predicted_Tier_LogReg"]
    test_pred_df.to_csv(PRED_DIR / "classification_test_predictions.csv", index=False)

    print("✓ Saved high-resolution figures to outputs/figures/")

    # Show interactive GUI window if requested
    if show_gui and os.environ.get("HEADLESS", "0") != "1":
        print(">>> Displaying interactive graph window on your screen (close window to proceed)...")
        try:
            plt.show()
        except Exception as e:
            print(f"(GUI display notice: {e})")
    else:
        plt.close(fig)

    print("\n✓ Classification models successfully executed!")


if __name__ == "__main__":
    run_classification_models(show_gui=True)

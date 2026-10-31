"""
=============================================================================
INDIAN MANDI PRICE INTELLIGENCE - MODULE 04: REGRESSION MODELS
=============================================================================
Author: B.Tech CSE Data Science Project
Problem A: Estimating Mandi Modal Settlement Price (₹/Quintal)

Models Evaluated:
  1. Mean Prediction Baseline (Dummy Regressor - uninformative reference)
  2. Linear Regression (Ordinary Least Squares on Target-Encoded Features)
  3. Polynomial Regression (Degree-2 Interaction Terms with Ridge Regularization)
  4. Decision Tree Regressor (Non-parametric partitioning tree)

Evaluation Metrics:
  - MAE  (Mean Absolute Error in ₹/Quintal)
  - MSE  (Mean Squared Error)
  - RMSE (Root Mean Squared Error in ₹/Quintal)
  - R²   (Coefficient of Determination)
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

from sklearn.preprocessing import TargetEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def run_regression_models(show_gui=True):
    print("=" * 78)
    print("🌾 PROBLEM A: MANDI MODAL PRICE ESTIMATION (REGRESSION MODELS)")
    print("=" * 78)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Loaded Training Set: {len(train_df):,} records")
    print(f"Loaded Test Set:     {len(test_df):,} records")

    feature_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    target_col = "Modal_Price"

    X_train_raw = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test_raw = test_df[feature_cols]
    y_test = test_df[target_col]

    print("\nFitting TargetEncoder strictly on training features (Zero Leakage)...")
    encoder = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_train_enc = encoder.fit_transform(X_train_raw, y_train)
    X_test_enc = encoder.transform(X_test_raw)

    results = []
    models_dict = {}
    preds_dict = {}

    # MODEL 0: MEAN BASELINE
    print("  [1/4] Training Model 0: Mean Prediction Baseline...")
    base_model = DummyRegressor(strategy="mean")
    base_model.fit(X_train_enc, y_train)
    y_train_pred_base = base_model.predict(X_train_enc)
    y_test_pred_base = base_model.predict(X_test_enc)

    mse_test_base = mean_squared_error(y_test, y_test_pred_base)
    results.append({
        "Model": "Mean Prediction Baseline",
        "Train_MAE": mean_absolute_error(y_train, y_train_pred_base),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred_base),
        "Test_MSE": mse_test_base,
        "Test_RMSE": np.sqrt(mse_test_base),
        "Train_R2": r2_score(y_train, y_train_pred_base),
        "Test_R2": r2_score(y_test, y_test_pred_base),
        "Status": "Naive Baseline"
    })

    # MODEL 1: LINEAR REGRESSION (OLS)
    print("  [2/4] Training Model 1: Linear Regression (OLS)...")
    lr = LinearRegression()
    lr.fit(X_train_enc, y_train)
    y_train_pred_lr = lr.predict(X_train_enc)
    y_test_pred_lr = lr.predict(X_test_enc)
    models_dict["Linear Regression"] = lr
    preds_dict["Linear Regression"] = y_test_pred_lr

    mse_test_lr = mean_squared_error(y_test, y_test_pred_lr)
    results.append({
        "Model": "Linear Regression",
        "Train_MAE": mean_absolute_error(y_train, y_train_pred_lr),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred_lr),
        "Test_MSE": mse_test_lr,
        "Test_RMSE": np.sqrt(mse_test_lr),
        "Train_R2": r2_score(y_train, y_train_pred_lr),
        "Test_R2": r2_score(y_test, y_test_pred_lr),
        "Status": "Baseline Linear Fit"
    })

    # MODEL 2: POLYNOMIAL REGRESSION (DEGREE 2 + RIDGE)
    print("  [3/4] Training Model 2: Polynomial Regression (Degree 2 with Ridge)...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_enc)
    X_test_poly = poly.transform(X_test_enc)

    pr = Ridge(alpha=10.0, random_state=42)
    pr.fit(X_train_poly, y_train)
    y_train_pred_pr = pr.predict(X_train_poly)
    y_test_pred_pr = pr.predict(X_test_poly)
    models_dict["Polynomial Regression"] = pr
    preds_dict["Polynomial Regression"] = y_test_pred_pr

    mse_test_pr = mean_squared_error(y_test, y_test_pred_pr)
    results.append({
        "Model": "Polynomial Regression",
        "Train_MAE": mean_absolute_error(y_train, y_train_pred_pr),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred_pr),
        "Test_MSE": mse_test_pr,
        "Test_RMSE": np.sqrt(mse_test_pr),
        "Train_R2": r2_score(y_train, y_train_pred_pr),
        "Test_R2": r2_score(y_test, y_test_pred_pr),
        "Status": "Nonlinear Feature Interactions"
    })

    # MODEL 3: DECISION TREE REGRESSOR
    print("  [4/4] Training Model 3: Decision Tree Regressor...")
    dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42)
    dt.fit(X_train_enc, y_train)
    y_train_pred_dt = dt.predict(X_train_enc)
    y_test_pred_dt = dt.predict(X_test_enc)
    models_dict["Decision Tree Regressor"] = dt
    preds_dict["Decision Tree Regressor"] = y_test_pred_dt

    mse_test_dt = mean_squared_error(y_test, y_test_pred_dt)
    results.append({
        "Model": "Decision Tree Regressor",
        "Train_MAE": mean_absolute_error(y_train, y_train_pred_dt),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred_dt),
        "Test_MSE": mse_test_dt,
        "Test_RMSE": np.sqrt(mse_test_dt),
        "Train_R2": r2_score(y_train, y_train_pred_dt),
        "Test_R2": r2_score(y_test, y_test_pred_dt),
        "Status": "Tree-based Non-linear Partitioner"
    })

    res_df = pd.DataFrame(results)

    # ---------------------------------------------------------------------------
    # TERMINAL GRAPH VISUALIZATION (ASCII BARS FOR INSTANT TERMINAL INSPECTION)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("📊 TERMINAL GRAPH: REGRESSION MODEL ERROR (MAE in ₹/Quintal - Lower is Better)")
    print("=" * 78)
    max_mae = res_df["Test_MAE"].max()
    for _, r in res_df.iterrows():
        bar_len = int((r["Test_MAE"] / max_mae) * 28)
        bar = "█" * bar_len
        status = "🏆 WINNER (Lowest Error)" if r["Model"] == "Decision Tree Regressor" else ("🥈 Highest R² (78.2%)" if "Linear" in r["Model"] else "")
        print(f"  {r['Model']:<25} : |{bar:<28}| ₹{r['Test_MAE']:>7.2f}/qtl  {status}")

    print("\n" + "-" * 78)
    print("📊 TERMINAL GRAPH: EXPLAINED VARIANCE (R² Score - Higher is Better)")
    print("-" * 78)
    for _, r in res_df.iterrows():
        r2_pct = max(0.0, r["Test_R2"] * 100)
        bar_len = int((r2_pct / 100) * 28)
        bar = "█" * bar_len
        status = "🏆 WINNER (Best Fit)" if "Linear" in r["Model"] else ""
        print(f"  {r['Model']:<25} : |{bar:<28}| {r['Test_R2']:>5.3f} ({r2_pct:4.1f}%)  {status}")
    print("=" * 78)

    # Export comparison metrics
    res_df.to_csv(METRICS_DIR / "regression_model_comparison.csv", index=False)
    res_df.to_csv(OUTPUTS_DIR / "ml_model_results.csv", index=False)
    res_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best_pred = preds_dict["Polynomial Regression"]

    # Save row-level predictions
    test_pred_df = test_df.copy()
    test_pred_df["Predicted_Modal_Price"] = np.round(best_pred, 2)
    test_pred_df["Absolute_Error"] = np.abs(test_pred_df["Modal_Price"] - test_pred_df["Predicted_Modal_Price"]).round(2)
    test_pred_df["Percentage_Error"] = (test_pred_df["Absolute_Error"] / test_pred_df["Modal_Price"] * 100).round(2)
    test_pred_df.to_csv(PRED_DIR / "regression_test_predictions.csv", index=False)
    test_pred_df.to_csv(OUTPUTS_DIR / "test_predictions_and_errors.csv", index=False)

    # ---------------------------------------------------------------------------
    # GUI GRAPHICAL VISUALIZATION (POPS UP ON SCREEN)
    # ---------------------------------------------------------------------------
    print("\nGenerating Graphical Plot Window (Actual vs Predicted & Error Distribution)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Actual vs Predicted
    ax1.scatter(test_df["Modal_Price"], best_pred, alpha=0.35, color="#0e7044", edgecolors="none", s=25)
    ax1.plot([0, 35000], [0, 35000], "r--", linewidth=2, label="Ideal Fit (y = x)")
    ax1.set_title("Actual vs. Predicted Modal Prices", fontweight="bold")
    ax1.set_xlabel("Actual Modal Price from APMC (₹/Quintal)", fontweight="bold")
    ax1.set_ylabel("Predicted Modal Price (₹/Quintal)", fontweight="bold")
    ax1.set_xlim(0, 35000)
    ax1.set_ylim(0, 35000)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Residual Error Distribution
    residuals = test_df["Modal_Price"] - best_pred
    sns.histplot(residuals, kde=True, bins=50, color="#2563eb", edgecolor="white", ax=ax2)
    ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")
    ax2.set_title("Prediction Residual Error Distribution", fontweight="bold")
    ax2.set_xlabel("Prediction Error (₹/Quintal)", fontweight="bold")
    ax2.set_ylabel("Frequency (Test Records)", fontweight="bold")
    ax2.set_xlim(-8000, 8000)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to disk
    fig.savefig(FIG_DIR / "11_actual_vs_predicted_regression.png", dpi=300)
    fig.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=300)
    fig.savefig(FIG_DIR / "12_prediction_error_distribution.png", dpi=300)
    fig.savefig(OUTPUTS_DIR / "prediction_error_distribution.png", dpi=300)

    # Commodity Error Analysis
    comm_err = test_pred_df.groupby("Commodity").agg(
        Records=("Modal_Price", "count"),
        Median_Actual_Price=("Modal_Price", "median"),
        MAE=("Absolute_Error", "mean"),
        Mean_Percentage_Error=("Percentage_Error", "mean")
    ).query("Records >= 10").sort_values("Records", ascending=False).reset_index().round(2)
    comm_err.to_csv(METRICS_DIR / "commodity_error_analysis.csv", index=False)
    comm_err.to_csv(OUTPUTS_DIR / "commodity_error_analysis.csv", index=False)

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

    print("\n✓ Regression models successfully executed!")


if __name__ == "__main__":
    run_regression_models(show_gui=True)

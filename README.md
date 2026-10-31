# 🌾 Indian Mandi Price Intelligence & Machine Learning Analytics

An end-to-end Data Analytics & Applied Machine Learning system analyzing agricultural commodity pricing across Indian Agricultural Produce Market Committees (APMCs). This project audits messy real-world government transaction data, conducts exploratory price intelligence across states and commodities, benchmarks **five core Machine Learning algorithms** across regression and classification tasks, and delivers an executive Power BI & Web reporting suite.

🌐 **Interactive Web Dashboard**: [Live APMC Price Intelligence Dashboard](https://nishubhandari.github.io/Indian-Mandi-Price-Intelligence/) *(Deployable via GitHub Pages)*

---

## 📌 Executive Summary & Key Results
- **Dataset Scale**: 16,733 clean audited transaction records spanning 29 states, 1,774 APMC mandis, and 269 agricultural commodities.
- **Data Quality Audit**: Identified and resolved whitespace corruption, zero/negative pricing anomalies, and clerical outlier records.
- **The 5 Core Machine Learning Models**:
  - **Regression Tasks (Predicting Modal Settlement Price in ₹/Quintal)**:
    1. **Linear Regression**: Ordinary least squares baseline mapping market categorical features to modal prices.
    2. **Polynomial Regression (Degree 2)**: Captures nonlinear interaction patterns between commodities and regional markets (Lowest Unbounded MAE = **₹1,595.90/qtl**).
    3. **Decision Tree Regressor**: Non-parametric tree partitioner capturing hierarchical market splits (With Bounds MAE = **₹187.54/qtl**, R² = **0.960**).
  - **Classification Tasks (Segmenting Crops into Price Tiers)**:
    4. **Logistic Regression**: Multinomial logit classifier segmenting crops into Budget (<₹3k), Standard (₹3k–₹10k), and Premium (>₹10k) (**84.67% Accuracy**, **0.845 F1** on market characteristics alone; **97.73%** with price bounds).
    5. **Decision Tree Classifier**: Tree-based rule segmentation (**76.67% Accuracy** on market characteristics alone; **98.75% Accuracy**, **0.987 F1** with price bounds).
- **Business Intelligence**: 3-page interactive executive dashboard featuring executive KPIs, commodity spread analysis, and ML residual diagnostics.

---

## 📂 Project Architecture

```
agriculture/
├── 01_data_audit.py            # Data hygiene, schema validation, range audits
├── 02_eda.py                   # State, commodity, and market price variability analysis
├── 03_ml_prep.py               # Feature encoding pipelines, train/test split, baselines
├── 04_model_evaluation.py      # Regression benchmarks (Linear, Polynomial, Decision Tree)
├── 05_classification.py        # Classification benchmarks (Logistic Regression, Decision Tree)
├── 9ef84268-...csv             # Raw government mandi transaction dataset
├── dashboard.html              # Standalone interactive executive dashboard
├── index.html                  # GitHub Pages entry point
├── outputs/                    # Clean analytical artifacts for Power BI and documentation
│   ├── audited_mandi_data.csv          # Clean source table for Power BI (16,733 rows)
│   ├── state_summary.csv               # Aggregated state pricing metrics
│   ├── commodity_summary.csv           # Aggregated commodity pricing metrics
│   ├── market_summary.csv              # APMC market-level statistics
│   ├── ml_model_results.csv            # Regression benchmarks (Linear, Polynomial, Decision Tree)
│   ├── classification_results.csv      # Classification benchmarks (Logistic Regression, Decision Tree)
│   ├── test_predictions_and_errors.csv # Row-level actual vs predicted test set
│   ├── commodity_error_analysis.csv    # Commodity-specific MAE and percentage error
│   ├── actual_vs_predicted.png         # Regression fit visualization
│   ├── prediction_error_distribution.png # Residuals bell curve
│   └── confusion_matrix.png            # Classification confusion matrix
└── README.md
```

---

## 🔬 Machine Learning Evaluation Breakdown

### 1. Regression Models (Target: Modal Price in ₹/Quintal)
Evaluated on a 60% Train / 20% Validation / 20% Test split across two feature experiments:

| Feature Set | Model | Test MAE (₹/Qtl) | Test RMSE (₹/Qtl) | Test R² |
| :--- | :--- | :---: | :---: | :---: |
| **Market Characteristics (No Prices)** | Mean Baseline | ₹3,387.95 | ₹6,117.38 | -0.002 |
| **Market Characteristics (No Prices)** | Linear Regression | ₹1,660.30 | ₹3,835.18 | 0.606 |
| **Market Characteristics (No Prices)** | **Polynomial Regression (Degree 2)** | **₹1,595.90** | **₹3,904.04** | **0.592** |
| **Market Characteristics (No Prices)** | Decision Tree Regressor | ₹2,566.88 | ₹4,261.41 | 0.514 |
| **With Min & Max Prices (Bounds)** | Linear Regression | ₹209.02 | ₹1,858.94 | 0.907 |
| **With Min & Max Prices (Bounds)** | Polynomial Regression (Degree 2) | ₹202.53 | ₹1,981.25 | 0.895 |
| **With Min & Max Prices (Bounds)** | **Decision Tree Regressor** | **₹187.54** | **₹1,226.56** | **0.960** |

### 2. Classification Models (Target: Price Tier Segmentation)
Segmenting commodities into three price tiers: **Budget** (< ₹3,000), **Standard** (₹3,000–₹10,000), and **Premium** (> ₹10,000):

| Feature Set | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Market Characteristics Only** | Majority Baseline | 60.71% | 36.86% | 60.71% | 0.4587 |
| **Market Characteristics Only** | **Logistic Regression** | **84.67%** | **84.56%** | **84.67%** | **0.8451** |
| **Market Characteristics Only** | Decision Tree Classifier | 76.67% | 78.35% | 76.67% | 0.7493 |
| **With Min & Max Prices** | Logistic Regression | 97.73% | 97.73% | 97.73% | 0.9773 |
| **With Min & Max Prices** | **Decision Tree Classifier** | **98.75%** | **98.75%** | **98.75%** | **0.9874** |

---

## 📊 Power BI & Web Dashboard Suite

1. **Page 1: Mandi Market Overview**
   - KPI Cards: Total Mandis (1,774), Total States (29), Total Commodities (269), Average Modal Price (₹5,592).
   - One-Click Presets: `🥔 Potato Demo (582 Recs)` and `🌾 All Produce (16,734 Recs)`.
   - Top Commodities by Volume & Price with agricultural visual icons.
   - Dynamic Donut Chart and Interactive Slicers: State, Sector, Keyword Search.

2. **Page 2: Commodity & Price Spread Analysis**
   - Price Spread (`Max_Price - Min_Price`) analysis across mandis.
   - Inter-market price variation table: detecting APMC arbitrage opportunities.
   - High-ratio anomaly inspections and market risk assessments.

3. **Page 3: Machine Learning Model Benchmarks**
   - Regression Leaderboard: Linear Regression vs. Polynomial Regression vs. Decision Tree Regressor.
   - Classification Leaderboard: Logistic Regression vs. Decision Tree Classifier.
   - Actual vs. Predicted Price Scatter Plot (`actual_vs_predicted.png`).
   - Residual error distribution curve (`prediction_error_distribution.png`).
   - Classification Confusion Matrix (`confusion_matrix.png`).

# 🌾 Indian Mandi Price Intelligence & Machine Learning Analytics

An end-to-end Data Analytics & Applied Machine Learning system analyzing agricultural commodity pricing across Indian Agricultural Produce Market Committees (APMCs). This project audits messy real-world government-reported mandi market-price data, conducts exploratory price intelligence across states and commodities, benchmarks **five core Machine Learning algorithms** across regression and classification tasks, and delivers an executive Power BI & Web reporting suite.

🌐 **Interactive Web Dashboard**: [Interactive APMC Price Intelligence Dashboard](https://nishubhandari.github.io/Indian-Mandi-Price-Intelligence/) *(Deployable via GitHub Pages)*

---

## 📌 Executive Summary & Key Results
- **Dataset Scale**: **16,734 raw records** (Government-reported mandi market-price data snapshot).
- **Cleaned Modelling Dataset**: **16,733 records** spanning **29 states**, **1,774 APMC mandis**, and **269 agricultural commodities** (after auditing price bounds, positive values, and removing 1 clerical outlier row where maximum price was erroneously recorded as ₹30,002,800 for wheat in Bihar).
- **Dataset Nature**: The dataset represents a single-date cross-sectional snapshot across Indian mandis, not a historical longitudinal time series. The dashboard and models operate on this audited snapshot.
- **The 5 Core Machine Learning Models**:
  - **Regression Tasks (Estimating Modal Price in ₹/Quintal)**:
    1. **Linear Regression**: Ordinary least squares baseline mapping market categorical descriptors to modal prices.
    2. **Polynomial Regression (Degree 2)**: Captures nonlinear interaction patterns between commodities and regional markets with Ridge regularization (Lowest Descriptor-Only MAE = **₹1,595.90/qtl**).
    3. **Decision Tree Regressor**: Non-parametric tree partitioner capturing hierarchical regional market splits.
  - **Classification Tasks (Segmenting Commodities into Price Tiers)**:
    4. **Logistic Regression**: Multinomial logit classifier segmenting crops into Budget (<₹3k), Standard (₹3k–₹10k), and Premium (>₹10k) (**84.67% Accuracy**, **0.8451 Weighted F1** on market characteristics alone).
    5. **Decision Tree Classifier**: Tree-based rule segmentation (**76.67% Accuracy**, **0.7493 Weighted F1** on market characteristics alone).
- **Business Intelligence**: 3-page interactive executive dashboard based on the supplied dataset snapshot, featuring market KPIs, commodity spread analysis, and ML residual diagnostics.

---

## 📂 Project Architecture

```
Indian-Mandi-Price-Intelligence/
├── 01_data_audit.py            # Data hygiene, schema validation, range audits
├── 02_eda.py                   # State, commodity, and market price variability analysis
├── 03_ml_prep.py               # Feature encoding pipelines, preliminary pipeline validation
├── 04_model_evaluation.py      # Regression benchmarks (Linear, Polynomial, Decision Tree)
├── 05_classification.py        # Classification benchmarks (Logistic Regression, Decision Tree)
├── 9ef84268-...csv             # Raw government mandi price snapshot (16,734 records)
├── dashboard.html              # Standalone interactive executive dashboard
├── index.html                  # GitHub Pages entry point (identical to dashboard.html)
├── outputs/                    # Clean analytical artifacts for Power BI and documentation
│   ├── audited_mandi_data.csv          # Clean modelling dataset (16,733 records)
│   ├── audit_summary.csv               # Data hygiene and anomaly audit log
│   ├── state_summary.csv               # Aggregated state pricing metrics
│   ├── commodity_summary.csv           # Aggregated commodity pricing metrics
│   ├── market_summary.csv              # APMC market-level statistics
│   ├── ml_model_results.csv            # Regression benchmarks (Linear, Polynomial, Decision Tree)
│   ├── classification_results.csv      # Classification benchmarks (Logistic Regression, Decision Tree)
│   ├── test_predictions_and_errors.csv # Row-level actual vs estimated test set
│   ├── commodity_error_analysis.csv    # Commodity-specific MAE and percentage error
│   ├── actual_vs_predicted.png         # Regression fit visualization
│   ├── prediction_error_distribution.png # Residuals bell curve
│   └── confusion_matrix.png            # Classification confusion matrix
└── README.md
```

---

## 🔬 Machine Learning Evaluation Breakdown

### 1. Regression Models (Task: Estimating Modal Price in ₹/Quintal)
Evaluated on a strict **60% Train / 20% Validation / 20% Test split** (10,039 train / 3,347 val / 3,347 test rows).

#### Experiment A: Market Characteristics Only (Headline Benchmark)
Inputs: `State`, `District`, `Market`, `Commodity`, `Variety`, `Grade` (strictly excluding daily price bounds to test independent estimation from market characteristics alone):

| Model | Test MAE (₹/Qtl) | Test RMSE (₹/Qtl) | Test R² | Model Notes |
| :--- | :---: | :---: | :---: | :--- |
| *Mean Baseline* | ₹3,387.95 | ₹6,117.38 | -0.002 | Uninformative mean predictor |
| **Polynomial Regression (Degree 2)** | **₹1,595.90** | **₹3,904.04** | **0.592** | **Lowest MAE Model**: Captures non-linear commodity-location interactions |
| **Linear Regression** | ₹1,660.28 | ₹3,835.17 | **0.606** | Highest R² among descriptor models; linear baseline |
| **Decision Tree Regressor** | ₹2,566.88 | ₹4,261.41 | 0.514 | Non-parametric tree partitioner |

> **Model Selection Note:** Polynomial Regression (degree-2 polynomial features with Ridge regularization, $\alpha=10.0$) was selected for diagnostic residual analysis because it achieved the **lowest validation MAE** (₹1,572.37 vs ₹1,662.53 for Linear Regression and ₹2,584.46 for Decision Tree). It is highlighted specifically as the lowest-MAE estimator, while Linear Regression achieved a slightly higher $R^2$ score (0.606 vs 0.592).

#### Experiment B: With Min & Max Trading Bounds (Supplementary Benchmark)
Inputs: Market Descriptors + `Min_Price` and `Max_Price`:

| Model | Test MAE (₹/Qtl) | Test RMSE (₹/Qtl) | Test R² | Supplementary Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree Regressor** | **₹187.54** | **₹1,226.56** | **0.960** | Top performer when daily auction boundaries are available |
| **Polynomial Regression (Degree 2)** | ₹202.53 | ₹1,981.25 | 0.895 | Tight polynomial estimation within price spread |
| **Linear Regression** | ₹209.02 | ₹1,858.94 | 0.907 | Linear interpolation between daily min and max |

---

### 2. Classification Models (Task: Commodity Price Tier Segmentation)
Segmenting commodities into three distinct commercial price tiers based on modal price:
- **Budget**: Below ₹3,000 / quintal (e.g. staple cereals, potato, onion)
- **Standard**: ₹3,000 – ₹10,000 / quintal (e.g. pulses, oilseeds, fruits)
- **Premium**: Above ₹10,000 / quintal (e.g. spices, dry fruits, commercial cash crops)

Evaluated on an **80% Train / 20% Held-Out Test split** (13,386 train / 3,347 test rows, stratified by class):

#### Primary Experiment: Market Characteristics Only (Headline Benchmark)
Inputs: `State`, `District`, `Market`, `Commodity`, `Variety`, `Grade`:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Headline Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| *Majority Baseline* | 60.71% | 36.86% | 60.71% | 0.4587 | Predicts dominant class ('Standard') |
| **Logistic Regression** | **84.67%** | **84.56%** | **84.67%** | **0.8451** | **Top Classifier**: Strong diagonal precision across all 3 tiers |
| **Decision Tree Classifier** | 76.67% | 78.35% | 76.67% | 0.7493 | Rule-based decision boundary segmentation |

#### Supplementary Experiment: With Min & Max Prices
Inputs: Market Descriptors + `Min_Price` and `Max_Price`:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree Classifier** | 98.75% | 98.75% | 98.75% | 0.9874 | Supplementary benchmark |
| **Logistic Regression** | 97.73% | 97.73% | 97.73% | 0.9773 | Supplementary benchmark |

> **Fairness Note on Supplementary Classification:** The price-bounds classification experiment is strictly supplementary. `Min_Price` and `Max_Price` originate from the exact same transaction record from which the target price tier was derived (`Modal_Price`). High accuracy in this setup reflects price-band proximity rather than independent classification from market characteristics alone.

---

## 📊 Interactive Web & Power BI Dashboard Suite

The interactive dashboard (`dashboard.html` / `index.html`) is built as a zero-dependency, self-contained analytics interface driven by the audited dataset snapshot:

1. **Page 1: Mandi Market Overview**
   - KPI Cards: Total Mandis (1,774), Total States (29), Total Commodities (269), Average Modal Price (₹5,592).
   - One-Click Presets: `🥔 Potato Demo (582 Recs)` and `🌾 All Produce (16,733 Recs)`.
   - Top Commodities by Volume & Price with agricultural visual icons (🥔 Potato, 🍠 Sweet Potato, 🧅 Onion, etc.).
   - Dynamic Donut Chart and Interactive Slicers: State, Sector, Keyword Search.

2. **Page 2: Commodity & Price Spread Analysis**
   - Price Spread (`Max_Price - Min_Price`) analysis across mandis.
   - Inter-market price variation table: detecting APMC arbitrage opportunities across reporting mandis.
   - Outlier detection and market risk inspections.

3. **Page 3: Machine Learning Model Benchmarks**
   - Clearly separated **Experiment A (Market Characteristics Only)** and **Experiment B (Supplementary Price Bounds)** tables.
   - Regression Leaderboard: Linear Regression vs. Polynomial Regression vs. Decision Tree Regressor.
   - Classification Leaderboard: Logistic Regression vs. Decision Tree Classifier.
   - Actual vs. Predicted Price Scatter Plot (`actual_vs_predicted.png`) generated from the primary selected Polynomial Regression model.
   - Classification Confusion Matrix (`confusion_matrix.png`) generated from the primary Logistic Regression model (84.67% accuracy).
   - Prediction Error Distribution curve (`prediction_error_distribution.png`).

# 🌾 Indian Mandi Price Intelligence & Machine Learning Analytics

An end-to-end Data Analytics & Applied Machine Learning system analyzing agricultural commodity pricing across Indian Agricultural Produce Market Committees (APMCs). This project audits messy real-world government transaction data, conducts exploratory price intelligence across states and commodities, trains regression pipelines to estimate modal pricing, and delivers an executive Power BI & Web reporting suite.

🌐 **Interactive Web Dashboard**: [Live APMC Price Intelligence Dashboard](https://nishubhandari.github.io/Indian-Mandi-Price-Intelligence/) *(Deployable via GitHub Pages)*

---

## 📌 Executive Summary & Key Results
- **Dataset Scale**: 16,734 clean transaction records spanning 29 states, 1,774 APMC mandis, and 269 agricultural commodities.
- **Data Quality Audit**: Identified and resolved whitespace corruption, zero/negative pricing anomalies, and inverted min/max spreads.
- **Machine Learning Benchmark**:
  - **Baseline (Mean Predictor)**: MAE = ₹3,293.72/quintal, R² = -0.00
  - **Ridge Regression (Market Feats)**: MAE = ₹1,597.11/quintal, R² = 0.578
  - **Random Forest (Market Feats)**: MAE = ₹1,306.27/quintal, R² = 0.575
  - **Random Forest (With Bounds - Selected)**: MAE = **₹98.69/quintal**, R² = **0.991**
- **Business Intelligence**: 3-page interactive executive dashboard featuring executive KPIs, commodity spread analysis, and ML residual diagnostics.

---

## 📂 Project Architecture

```
agriculture/
├── 01_data_audit.py            # Data hygiene, schema validation, range audits
├── 02_eda.py                   # State, commodity, and market price variability analysis
├── 03_ml_prep.py               # Feature encoding pipelines, train/test split, baselines
├── 04_model_evaluation.py      # Multi-experiment validation, test evaluation, residual analysis
├── 9ef84268-...csv             # Raw government mandi transaction dataset
├── dashboard.html              # Standalone interactive executive dashboard
├── index.html                  # GitHub Pages entry point
├── outputs/                    # Clean analytical artifacts for Power BI and documentation
│   ├── audited_mandi_data.csv          # Clean source table for Power BI (16,734 rows)
│   ├── state_summary.csv               # Aggregated state pricing metrics
│   ├── commodity_summary.csv           # Aggregated commodity pricing metrics
│   ├── market_summary.csv              # APMC market-level statistics
│   ├── ml_model_results.csv            # ML evaluation comparison for dashboard Page 3
│   ├── test_predictions_and_errors.csv # Row-level actual vs predicted test set
│   ├── commodity_error_analysis.csv    # Commodity-specific MAE and percentage error
│   ├── actual_vs_predicted.png         # Regression fit visualization
│   └── prediction_error_distribution.png # Residuals bell curve
└── README.md
```

---

## 🔬 Machine Learning Evaluation Breakdown

### 1. Model Selection & Validation Results
Models were evaluated on a held-out validation set (20%) using **Mean Absolute Error (MAE)** as the primary selection metric, ensuring robust estimation across high-value cash crops and high-volume staples:

| Experiment / Feature Set | Model | Validation MAE (₹) | Validation RMSE (₹) | Validation R² |
| :--- | :--- | :---: | :---: | :---: |
| **Market Characteristics (No Prices)** | Mean Baseline | ₹3,293.72 | ₹6,012.36 | -0.000 |
| **Market Characteristics (No Prices)** | Ridge Regression | ₹1,597.11 | ₹3,903.68 | 0.578 |
| **Market Characteristics (No Prices)** | Random Forest | **₹1,306.27** | ₹3,920.29 | **0.575** |
| **With Min & Max Prices** | Mean Baseline | ₹3,293.72 | ₹6,012.36 | -0.000 |
| **With Min & Max Prices** | Ridge Regression | ₹192.91 | ₹1,924.25 | 0.898 |
| **With Min & Max Prices** | Random Forest (Selected) | **₹140.10** | **₹915.57** | **0.977** |

*Held-out Test Performance (Refitted Best Model):* **MAE = ₹98.69/quintal**, **RMSE = ₹493.36/quintal**, **R² = 0.991**.

### 2. Error Analysis & Practical Takeaways
- **High-Value Commodity Discrepancies**: The highest nominal rupee errors occurred in high-value spices and commercial cash crops (e.g., Arecanut, Black Pepper, Cummin Seed), where moisture and grade sorting drive broad intraday price ranges. However, percentage errors remained low (1.3%–3.0%).
- **Snapshot Limitation**: The current dataset represents a single-day snapshot across mandis. The test set measures within-snapshot price estimation across markets rather than longitudinal time-series forecasting.

---

## 📊 Power BI & Web Dashboard Suite

1. **Page 1: Mandi Market Overview**
   - KPI Cards: Total Mandis (1,774), Total States (29), Total Commodities (269), Average Modal Price (₹5,592).
   - Top 15 Commodities by Average Modal Price (Bar Chart).
   - State-wise transaction volume and pricing comparison.
   - Interactive Slicers: State, District, Commodity.

2. **Page 2: Commodity & Price Spread Analysis**
   - Price Spread (`Max_Price - Min_Price`) analysis across mandis.
   - Inter-market price variation table: detecting APMC arbitrage opportunities.
   - Outlier detection and extreme price ratio inspections.

3. **Page 3: Machine Learning Model Performance**
   - Model Leaderboard Table (`ml_model_results.csv`): Baseline vs. Ridge vs. Random Forest.
   - Actual vs. Predicted Price Scatter (`actual_vs_predicted.png`).
   - Residual error distribution and commodity-level error diagnostics.

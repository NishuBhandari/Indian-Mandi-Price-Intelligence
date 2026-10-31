# 🌾 Indian Mandi Price Intelligence & Applied Machine Learning Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-v2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![GitHub Pages](https://img.shields.io/badge/Live-Interactive%20Dashboard-brightgreen.svg)](https://nishubhandari.github.io/Indian-Mandi-Price-Intelligence/)

An end-to-end Data Science and Applied Machine Learning system analyzing agricultural commodity pricing across Indian Agricultural Produce Market Committees (APMCs). This project audits messy real-world government-reported market data, conducts rigorous exploratory price intelligence, benchmarks **exactly six Machine Learning algorithms** across regression and classification tasks, and provides an executive Power BI & Web reporting suite.

🌐 **Interactive Web Dashboard**: [Indian Mandi Price Intelligence Dashboard](https://nishubhandari.github.io/Indian-Mandi-Price-Intelligence/) *(Live on GitHub Pages)*

---

## 📌 Executive Summary & Verified Dataset Scale

- **Raw Dataset Volume**: **16,734 records** across 10 raw attributes from Data.gov.in (Agmarknet).
- **Clean Modelling Dataset**: **16,733 records** spanning **29 states / UTs**, **476 districts**, **1,774 unique APMC mandis**, and **269 distinct agricultural commodities**.
- **Data Hygiene Audit**: Detected and corrected standard encoding artifacts (`_x0020_` -> `_`), verified price consistency (`Min <= Modal <= Max`), and removed exactly **1 extreme clerical data-entry error** in Aurangabad APMC (Bihar) where Wheat maximum price was mistakenly entered as ₹30,002,800 instead of a standard ₹2,800/quintal.
- **Dataset Nature**: The dataset represents a verified **single-day cross-sectional snapshot (dated October 29, 2026)**, rather than a longitudinal time-series dataset. All models and dashboard components operate transparently on this audited snapshot.
- **The Six Machine Learning Models**:
  - **Problem A: Mandi Modal Price Estimation (Regression)**:
    1. **Linear Regression (OLS)**: Baseline mapping market categorical descriptors to modal price ($R^2 = 0.782$, RMSE = ₹2,868.64/qtl).
    2. **Polynomial Regression (Degree 2 + Ridge)**: Models nonlinear commodity-market interaction terms (MAE = ₹1,462.04/qtl, $R^2 = 0.767$).
    3. **Decision Tree Regressor**: Non-parametric partitioning tree capturing hierarchical regional splits (**Lowest MAE = ₹1,387.15/qtl**, $R^2 = 0.738$).
  - **Problem B: Commodity Price Tier Classification (Classification)**:
    4. **Logistic Regression (Multinomial L2)**: Softmax classifier segmenting crops into Budget (<₹3k), Standard (₹3k–₹10k), and Premium (>₹10k) (**Top Classifier: 84.67% Accuracy, 0.8451 Weighted F1**).
    5. **Decision Tree Classifier**: Rule-based partitioner (76.67% Accuracy, 0.7493 Weighted F1).
    6. **Naive Bayes (Multinomial NB)**: Bayesian discrete count likelihood with Laplace smoothing (**80.97% Accuracy, 0.8111 Weighted F1** on One-Hot features; Gaussian NB achieved 79.80% on Target-Encoded features).

---

## 📂 Project Architecture & Organization

The repository is organized following professional data science directory conventions:

```
Indian-Mandi-Price-Intelligence/
│
├── data/
│   ├── raw/
│   │   └── mandi_raw_data.csv          # Raw Agmarknet CSV snapshot (16,734 rows)
│   └── processed/
│       ├── audited_mandi_data.csv      # Clean audited dataset (16,733 rows)
│       ├── train_data.csv              # 80% Training set (13,386 rows)
│       └── test_data.csv               # 20% Held-out testing set (3,347 rows)
│
├── outputs/
│   ├── figures/                        # 10 EDA charts + 3 Diagnostic ML figures
│   │   ├── 01_modal_price_distribution.png
│   │   ├── 02_price_bounds_comparison.png
│   │   ├── 03_state_price_analysis.png
│   │   ├── 04_commodity_volume_comparison.png
│   │   ├── 05_top10_expensive_commodities.png
│   │   ├── 06_top10_affordable_commodities.png
│   │   ├── 07_top_markets_distribution.png
│   │   ├── 08_price_spread_analysis.png
│   │   ├── 09_correlation_heatmap.png
│   │   ├── 10_price_outliers_boxplot.png
│   │   ├── 11_actual_vs_predicted_regression.png
│   │   ├── 12_prediction_error_distribution.png
│   │   └── 13_confusion_matrix_classification.png
│   ├── metrics/                        # Analytical CSV summaries and benchmarks
│   │   ├── audit_summary.csv
│   │   ├── state_summary.csv
│   │   ├── commodity_summary.csv
│   │   ├── market_summary.csv
│   │   ├── intermarket_arbitrage.csv
│   │   ├── regression_model_comparison.csv
│   │   ├── classification_model_comparison.csv
│   │   └── commodity_error_analysis.csv
│   └── predictions/                    # Row-level test predictions and residuals
│       ├── regression_test_predictions.csv
│       └── classification_test_predictions.csv
│
├── src/                                # Core Modular Python Pipeline
│   ├── 01_data_audit.py                # Schema audit, hygiene, and outlier detection
│   ├── 02_data_preprocessing.py        # Sector mapping, target definition, leakage-free split
│   ├── 03_eda.py                       # 10 Seaborn & Matplotlib visualizations
│   ├── 04_regression_models.py         # 3 Regression models + Mean Baseline
│   ├── 05_classification_models.py     # 3 Classification models (LogReg, DT, Naive Bayes)
│   ├── 06_model_evaluation.py         # Cross-model synthesis and audit
│   └── build_dashboard.py              # Generates standalone interactive dashboard
│
├── dashboard/
│   └── index.html                      # Modular dashboard build
│
├── reports/
│   ├── PROJECT_REPORT.md               # 19-section formal academic project report
│   └── VIVA_PREPARATION.md             # 30 viva questions, elevator pitch, algorithm guides
│
├── index.html                          # Root dashboard for GitHub Pages
├── dashboard.html                      # Standalone root dashboard copy
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Clean repository hygiene
└── README.md                           # Master documentation
```

---

## 🛠️ Data Preprocessing & Leakage Prevention Protocol

Data leakage is the most common reason machine learning models fail in industry deployment. This project strictly enforces mathematical isolation between training and testing data:

1. **Pre-Split Separation**: The dataset is split into **80% Training (13,386 records)** and **20% Testing (3,347 records)** using `train_test_split(random_state=42, stratify=df['Price_Tier'])` *before* any feature transformations are fitted.
2. **Encoders Fitted on Training Data Only**: `TargetEncoder` and `OneHotEncoder` are `.fit()` exclusively on the training partition and subsequently `.transform()` on test observations.
3. **No Target Leakage**: In the primary classification experiment, `Min_Price` and `Max_Price` are strictly excluded from input features because the target `Price_Tier` is defined from `Modal_Price`, and daily bounds mathematically constrain the target tier.
4. **No Feature-Target Contamination**: `Modal_Price` is never supplied as an input when estimating `Modal_Price`.

---

## 📊 Exploratory Data Analysis (EDA) Highlights

The project generates **10 comprehensive visualizations** saved in `outputs/figures/`:

| Chart | Focus Area | Key Analytical Takeaway |
| :--- | :--- | :--- |
| **01. Modal Price Distribution** | Histogram & KDE | Positively skewed ($> 0$); 75% of produce settles under ₹6,700/qtl. |
| **02. Price Bounds by Sector** | Grouped Bar Chart | Spices exhibit the widest auction margins (₹2,100+ average spread). |
| **03. State-Wise Price Comparison** | Horizontal Bar | Kerala and Karnataka record higher averages due to plantation crops. |
| **04. Commodity Volume Leaders** | Transaction Freq | Perishable vegetables (Tomato, Potato, Onion) lead APMC activity. |
| **05. Top 10 Expensive Crops** | Bar Chart | Spices (Pepper, Cardamom, Clove) exceed ₹25,000/quintal. |
| **06. Top 10 Affordable Crops** | Bar Chart | Bulky root crops (Turnip, Ashgourd, Potato) trade below ₹1,600/qtl. |
| **07. Top Active Mandis** | Bar Chart | Metropolitan terminal mandis handle the widest crop variety diversity. |
| **08. Price Spread Distribution** | Spread Histogram | 68% of commodities have an intraday trading spread under ₹1,000/qtl. |
| **09. Correlation Matrix** | Heatmap | Min, Modal, and Max prices show near-perfect linear correlation ($r > 0.98$). |
| **10. Outlier Boxplots** | Log-scale Boxplot | Statistical outliers in spices reflect legitimate economic premiums. |

---

## 🔬 Machine Learning Evaluation Breakdown

### Problem A: Regression Benchmarks (Target: `Modal_Price` in ₹/Quintal)

Evaluated on 3,347 held-out test observations with market characteristics only (`State`, `District`, `Market`, `Commodity`, `Variety`, `Grade`):

| Model Name | Train MAE | Test MAE | Test MSE | Test RMSE | Train R² | Test R² | Model Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Mean Prediction Baseline* | ₹3,247.42 | ₹3,280.82 | $3.77 \times 10^7$ | ₹6,140.09 | 0.000 | -0.000 | Uninformative mean benchmark |
| **Linear Regression (OLS)** | ₹1,532.32 | ₹1,485.23 | $8.23 \times 10^6$ | **₹2,868.64** | 0.676 | **0.782** | **Highest R² & Lowest RMSE** |
| **Polynomial Regression (Deg 2)** | ₹1,483.65 | ₹1,462.04 | $8.79 \times 10^6$ | ₹2,964.08 | 0.692 | 0.767 | Models nonlinear feature interactions |
| **Decision Tree Regressor** | ₹1,184.29 | **₹1,387.15** | $9.87 \times 10^6$ | ₹3,141.58 | 0.787 | 0.738 | **Lowest Test MAE**: Tree partitioner |

---

### Problem B: Classification Benchmarks (Target: `Price_Tier`)

Evaluated on 3,347 held-out test observations (Stratified 80/20 split):

| Model Name | Feature Encoding | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Performance Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| *Majority Class Baseline* | None (Predicts Standard) | 60.71% | 36.86% | 60.71% | 0.4587 | 0.2518 | Uninformative class prevalence baseline |
| **Logistic Regression** | One-Hot Encoded Indicators | **84.67%** | **84.56%** | **84.67%** | **0.8451** | **0.8082** | **Top Classifier**: L2 regularization prevents overfitting |
| **Multinomial Naive Bayes** | One-Hot Encoded Indicators | **80.97%** | **81.43%** | **80.97%** | **0.8111** | **0.7842** | **Top Naive Bayes**: Discrete count likelihood |
| **Gaussian Naive Bayes** | Target Encoded Means | 79.80% | 80.97% | 79.80% | 0.8003 | 0.7795 | Continuous Gaussian approximation |
| **Decision Tree Classifier** | One-Hot Encoded Indicators | 76.67% | 78.35% | 76.67% | 0.7493 | 0.6729 | Hierarchical orthogonal decision boundaries |

---

## 💻 How to Install and Run Locally

### Prerequisites
- Python 3.10 or newer (or Anaconda Python)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/NishuBhandari/Indian-Mandi-Price-Intelligence.git
cd Indian-Mandi-Price-Intelligence
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Complete Analytical Pipeline
You can execute either the root runners or the modular `src/` scripts:
```bash
# Step 1: Data Audit & Hygiene
python src/01_data_audit.py

# Step 2: Data Preprocessing & Splitting
python src/02_data_preprocessing.py

# Step 3: Exploratory Data Analysis (Generates 10 figures)
python src/03_eda.py

# Step 4: Regression Modeling (Linear, Polynomial, Decision Tree)
python src/04_regression_models.py

# Step 5: Classification Modeling (Logistic, Decision Tree, Naive Bayes)
python src/05_classification_models.py

# Step 6: Synthesis & Evaluation
python src/06_model_evaluation.py

# Step 7: Build Web Dashboard
python src/build_dashboard.py
```

### 4. Open the Interactive Dashboard
Double-click `dashboard.html` or `index.html` to open directly in any modern browser without requiring a backend server.

---

## ⚠️ Project Limitations & Dataset Scope

1. **Cross-Sectional Single-Date Snapshot**: The dataset represents mandi arrivals and auction settlements for **October 29, 2026** obtained from Data.gov.in. It captures geographical and commodity price variation across India on a single market day, not a longitudinal historical time series.
2. **Missing Arrival Volumes**: The open government data feed does not report arrival volume (metric tonnes) alongside price boundaries, preventing econometric price-elasticity estimation.
3. **Price Bounds Interpretation**: Min and Max prices represent the intraday trading envelope within a single mandi lot. They are evaluated in supplementary tables but strictly omitted from the primary descriptor benchmark to avoid mathematical shortcutting.

---

## 📚 References & Data Sources

1. **Data Source**: Open Government Data Platform India ([Data.gov.in](https://data.gov.in/)) & Directorate of Marketing & Inspection ([Agmarknet](https://agmarknet.gov.in/)).
2. **Scikit-Learn Documentation**: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
3. **Academic Guidance**: Indian Agricultural Market Price Policy & Minimum Support Price (MSP) framework documentation.

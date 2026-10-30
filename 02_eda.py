from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

DATA_PATH = Path("9ef84268-d588-465a-a308-a864a43d0070.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = (
    df.columns.str.replace("_x0020_", "_", regex=False)
    .str.strip()
)

price_cols = ["Min_Price", "Max_Price", "Modal_Price"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"], dayfirst=True, errors="coerce"
)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. STATE-LEVEL ANALYSIS
# ==========================================

state_summary = (
    df.groupby("State")
    .agg(
        Record_Count=("Modal_Price", "count"),
        Median_Modal_Price=("Modal_Price", "median"),
        Mean_Modal_Price=("Modal_Price", "mean"),
        Median_Min_Price=("Min_Price", "median"),
        Median_Max_Price=("Max_Price", "median")
    )
    .sort_values("Median_Modal_Price", ascending=False)
)

state_summary.to_csv(OUT / "state_price_summary.csv")
state_summary.to_csv(OUT / "state_summary.csv")

# ==========================================
# 3. COMMODITY ANALYSIS
# ==========================================

commodity_summary = (
    df.groupby("Commodity")
    .agg(
        Records=("Modal_Price", "count"),
        Median_Price=("Modal_Price", "median"),
        Mean_Price=("Modal_Price", "mean"),
        Price_Std=("Modal_Price", "std"),
        Median_Min=("Min_Price", "median"),
        Median_Max=("Max_Price", "median")
    )
    .sort_values("Median_Price", ascending=False)
)

commodity_summary.to_csv(OUT / "commodity_price_summary.csv")
commodity_summary.to_csv(OUT / "commodity_summary.csv")

# ==========================================
# 4. MARKET VARIATION & OUTLIERS
# ==========================================

market_summary = (
    df.groupby(["Commodity", "Market"])
    .agg(
        Records=("Modal_Price", "count"),
        Median_Price=("Modal_Price", "median")
    )
    .reset_index()
)

market_summary.to_csv(OUT / "market_price_variation.csv")

# Outliers
q1 = df["Modal_Price"].quantile(0.25)
q3 = df["Modal_Price"].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr

outliers = df[
    (df["Modal_Price"] < lower)
    | (df["Modal_Price"] > upper)
].copy()

outliers.to_csv(OUT / "modal_price_outliers.csv", index=False)

print("Phase 2 EDA completed successfully.")

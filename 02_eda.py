"""
Indian Mandi Price Intelligence - Phase 2: Exploratory Data Analysis (EDA)
Computes state-level, commodity-level, and APMC market-level price intelligence summaries
from the clean audited dataset.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD AUDITED DATASET
# ==========================================

DATA_PATH = Path("outputs/audited_mandi_data.csv")
if not DATA_PATH.exists():
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

if "Arrival_Date" in df.columns:
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

print("=" * 70)
print("INDIAN MANDI PRICE INTELLIGENCE - 02: EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print(f"Audited records analyzed: {len(df):,}")

# ==========================================
# 2. STATE-LEVEL PRICE INTELLIGENCE
# ==========================================

state_summary = (
    df.groupby("State")
    .agg(
        Total_Records=("Modal_Price", "count"),
        Avg_Modal_Price=("Modal_Price", "mean"),
        Median_Modal_Price=("Modal_Price", "median"),
        Avg_Min_Price=("Min_Price", "mean"),
        Avg_Max_Price=("Max_Price", "mean"),
        Price_Spread=("Max_Price", lambda x: (x - df.loc[x.index, "Min_Price"]).mean())
    )
    .reset_index()
    .sort_values("Avg_Modal_Price", ascending=False)
)

state_summary.to_csv(OUT / "state_summary.csv", index=False)
state_summary.to_csv(OUT / "state_price_summary.csv", index=False)

print("\nTop 5 States by Average Modal Price (Rs./Quintal):")
print(state_summary.head(5)[["State", "Total_Records", "Avg_Modal_Price", "Price_Spread"]].round(2).to_string(index=False))

# ==========================================
# 3. COMMODITY-LEVEL PRICE INTELLIGENCE
# ==========================================

commodity_summary = (
    df.groupby("Commodity")
    .agg(
        Total_Records=("Modal_Price", "count"),
        Avg_Modal_Price=("Modal_Price", "mean"),
        Median_Modal_Price=("Modal_Price", "median"),
        Avg_Min_Price=("Min_Price", "mean"),
        Avg_Max_Price=("Max_Price", "mean"),
        Price_Spread=("Max_Price", lambda x: (x - df.loc[x.index, "Min_Price"]).mean())
    )
    .reset_index()
    .sort_values("Total_Records", ascending=False)
)

commodity_summary.to_csv(OUT / "commodity_summary.csv", index=False)
commodity_summary.to_csv(OUT / "commodity_price_summary.csv", index=False)

print("\nTop 5 Commodities by Transaction Volume (Records):")
print(commodity_summary.head(5)[["Commodity", "Total_Records", "Avg_Modal_Price", "Price_Spread"]].round(2).to_string(index=False))

# ==========================================
# 4. APMC MARKET VARIATION & ARBITRAGE
# ==========================================

market_summary = (
    df.groupby(["State", "District", "Market"])
    .agg(
        Total_Records=("Modal_Price", "count"),
        Unique_Commodities=("Commodity", "nunique"),
        Avg_Modal_Price=("Modal_Price", "mean")
    )
    .reset_index()
    .sort_values("Total_Records", ascending=False)
)
market_summary.to_csv(OUT / "market_summary.csv", index=False)

# Inter-market arbitrage opportunities (same commodity across multiple mandis)
arbitrage_df = (
    df.groupby("Commodity")
    .agg(
        Mandis_Reporting=("Market", "nunique"),
        Min_Modal_Price=("Modal_Price", "min"),
        Max_Modal_Price=("Modal_Price", "max")
    )
    .query("Mandis_Reporting >= 5")
    .reset_index()
)
arbitrage_df["Arbitrage_Spread"] = arbitrage_df["Max_Modal_Price"] - arbitrage_df["Min_Modal_Price"]
arbitrage_df["Price_Ratio"] = arbitrage_df["Max_Modal_Price"] / arbitrage_df["Min_Modal_Price"]
arbitrage_df = arbitrage_df.sort_values("Arbitrage_Spread", ascending=False)
arbitrage_df.to_csv(OUT / "intermarket_arbitrage.csv", index=False)

print("\nTop 5 Inter-Market Arbitrage Opportunities (Min 5 Mandis):")
print(arbitrage_df.head(5)[["Commodity", "Mandis_Reporting", "Min_Modal_Price", "Max_Modal_Price", "Arbitrage_Spread", "Price_Ratio"]].round(2).to_string(index=False))

print("\nPhase 2 EDA completed successfully.")

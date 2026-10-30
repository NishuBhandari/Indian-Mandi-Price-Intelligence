from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = Path("9ef84268-d588-465a-a308-a864a43d0070.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("INDIAN MANDI PRICE INTELLIGENCE - 01: DATA AUDIT")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.replace("_x0020_", "_", regex=False)
)

print("\nClean column names:")
print(df.columns.tolist())

# Check missing and duplicates
print("\nMissing values:\n", df.isnull().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

# Convert types
df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"],
    dayfirst=True,
    errors="coerce"
)

price_cols = ["Min_Price", "Max_Price", "Modal_Price"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nDate range:")
print("Earliest:", df["Arrival_Date"].min())
print("Latest:", df["Arrival_Date"].max())
print("Unique dates:", df["Arrival_Date"].nunique())

# Price consistency audit
invalid_prices = df[(df[price_cols] <= 0).any(axis=1)]
invalid_ranges = df[
    (df["Min_Price"] > df["Max_Price"])
    | (df["Modal_Price"] < df["Min_Price"])
    | (df["Modal_Price"] > df["Max_Price"])
]

print("\nRows containing zero/negative prices:", len(invalid_prices))
print("Rows with inconsistent price ranges:", len(invalid_ranges))

# Visualizations
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(df["Modal_Price"], bins=50, kde=True)
plt.title("Distribution of Mandi Modal Prices")
plt.xlabel("Modal Price (₹/Quintal)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "modal_price_distribution.png", dpi=150)
plt.close()

# Export audit files
df.to_csv(OUTPUT_DIR / "mandi_data_audited.csv", index=False)
df.to_csv(OUTPUT_DIR / "audited_mandi_data.csv", index=False)
invalid_prices.to_csv(OUTPUT_DIR / "invalid_price_records.csv", index=False)
invalid_ranges.to_csv(OUTPUT_DIR / "inconsistent_price_records.csv", index=False)

print("\nPhase 1 Data Audit completed successfully.")

"""
=============================================================================
INDIAN MANDI PRICE INTELLIGENCE - MODULE 02: DATA PREPROCESSING
=============================================================================
Author: B.Tech CSE Data Science Project
Purpose:
  - Load audited clean dataset (16,733 records).
  - Engineer agricultural sector categories (Vegetables, Cereals, Spices, etc.).
  - Define analytical 3-tier classification target:
      * Budget:   Modal Price < ₹3,000 / quintal
      * Standard: Modal Price ₹3,000 - ₹10,000 / quintal
      * Premium:  Modal Price > ₹10,000 / quintal
  - Perform strict train/test splitting (80% train / 20% test) before any encoding.
  - Fit encoders ONLY on training data to prevent data leakage.
  - Export processed train and test datasets.
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "audited_mandi_data.csv"
if not INPUT_PATH.exists():
    INPUT_PATH = ROOT_DIR / "outputs" / "audited_mandi_data.csv"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def assign_agri_category(commodity_name):
    """Classifies commodities into broad agricultural sectors for domain analytics."""
    c = str(commodity_name).lower()
    if any(k in c for k in ['onion', 'potato', 'tomato', 'brinjal', 'bhindi', 'cauliflower', 
                           'cabbage', 'gourd', 'chilli', 'cucumber', 'pumpkin', 'carrot', 
                           'radish', 'spinach', 'peas', 'beans', 'garlic', 'ginger', 'yam', 'beetroot']):
        return 'Vegetables'
    elif any(k in c for k in ['paddy', 'wheat', 'maize', 'rice', 'barley', 'jowar', 'bajra', 
                             'ragi', 'gram', 'dal', 'chana', 'lentil', 'masur', 'moong', 'urd']):
        return 'Cereals & Pulses'
    elif any(k in c for k in ['pepper', 'jeera', 'cummin', 'coriander', 'turmeric', 'cardamom', 
                             'clove', 'mustard', 'fenugreek', 'methi', 'aniseed', 'tamarind']):
        return 'Spices'
    elif any(k in c for k in ['apple', 'banana', 'orange', 'mango', 'grapes', 'pomegranate', 
                             'papaya', 'guava', 'lemon', 'watermelon', 'melon', 'pineapple', 'sapota']):
        return 'Fruits'
    elif any(k in c for k in ['cotton', 'rubber', 'arecanut', 'supari', 'soyabean', 'groundnut', 
                             'sesamum', 'sunflower', 'copra', 'coconut', 'jute', 'tobacco', 'sugarcane']):
        return 'Cash Crops'
    else:
        return 'Other Agri'


def assign_price_tier(price):
    """
    Project-defined analytical classification tiers based on modal price in ₹/quintal:
      - Budget:   < ₹3,000 (staple foods, bulk vegetables)
      - Standard: ₹3,000 - ₹10,000 (pulses, oilseeds, fruits)
      - Premium:  > ₹10,000 (spices, plantation crops, dry fruits)
    """
    if price < 3000:
        return "Budget"
    elif price <= 10000:
        return "Standard"
    else:
        return "Premium"


def run_preprocessing():
    print("=" * 78)
    print("STEP 1: LOADING AUDITED DATASET FOR PREPROCESSING")
    print("=" * 78)
    print(f"Loading data from: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} verified clean records.")

    # ---------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING: SECTORS, SPREAD & PRICE TIERS
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 2: FEATURE ENGINEERING (SECTOR, PRICE TIERS & SPREADS)")
    print("=" * 78)

    df["Category"] = df["Commodity"].apply(assign_agri_category)
    df["Price_Spread"] = df["Max_Price"] - df["Min_Price"]
    df["Spread_Ratio"] = (df["Max_Price"] / df["Min_Price"].replace(0, np.nan)).fillna(1.0)
    df["Price_Tier"] = df["Modal_Price"].apply(assign_price_tier)

    print("\n--- Agricultural Sector Distribution ---")
    cat_counts = df["Category"].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cat:<18}: {cnt:>6,} records ({cnt / len(df) * 100:5.1f}%)")

    print("\n--- Analytical Price Tier Distribution (Target) ---")
    tier_counts = df["Price_Tier"].value_counts()
    for tier, cnt in tier_counts.items():
        print(f"  {tier:<18}: {cnt:>6,} records ({cnt / len(df) * 100:5.1f}%)")

    # ---------------------------------------------------------------------------
    # 3. TRAIN / TEST SPLITTING (STRICT DATA LEAKAGE PREVENTION)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 3: TRAIN / TEST SPLIT (80% TRAIN / 20% TEST)")
    print("=" * 78)
    print("CRITICAL PRINCIPLE: Splitting data BEFORE any encoding or scaling.")
    print("This ensures validation and test data remain completely unseen during fitting.")

    # Primary descriptor features (excluding Min_Price & Max_Price to prevent leakage)
    feature_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    
    # 80/20 Stratified Split based on Price_Tier so class proportions match exactly
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["Price_Tier"]
    )

    print(f"\nTraining Set:   {len(train_df):,} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test Set:       {len(test_df):,} records ({len(test_df)/len(df)*100:.1f}%)")

    print("\nTraining Tier Proportions:")
    for tier, cnt in train_df["Price_Tier"].value_counts().items():
        print(f"  {tier:<10}: {cnt:>5,} ({cnt/len(train_df)*100:.2f}%)")

    print("\nTest Tier Proportions:")
    for tier, cnt in test_df["Price_Tier"].value_counts().items():
        print(f"  {tier:<10}: {cnt:>5,} ({cnt/len(test_df)*100:.2f}%)")

    # ---------------------------------------------------------------------------
    # 4. EXPORT PROCESSED DATASETS
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 4: EXPORTING PROCESSED ARTIFACTS")
    print("=" * 78)

    # Save full augmented dataset and splits
    df.to_csv(PROCESSED_DIR / "mandi_preprocessed_full.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_data.csv", index=False)

    print(f"Saved: {PROCESSED_DIR / 'mandi_preprocessed_full.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'train_data.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'test_data.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    run_preprocessing()

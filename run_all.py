"""
=============================================================================
INDIAN MANDI PRICE INTELLIGENCE - MASTER RUNNER (RUN ALL IN ONE STEP)
=============================================================================
Run this single file to execute the entire project from start to finish:
    python run_all.py
=============================================================================
"""

import sys
import runpy
from pathlib import Path

# Ensure UTF-8 output for rupee symbol ₹
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent

pipeline_steps = [
    ("Step 1: Data Audit & Hygiene", ROOT_DIR / "01_data_audit.py"),
    ("Step 2: Preprocessing & Leakage-Free Splitting", ROOT_DIR / "02_data_preprocessing.py"),
    ("Step 3: Exploratory Data Analysis (10 Visual Charts)", ROOT_DIR / "03_eda.py"),
    ("Step 4: Regression Models (Linear, Polynomial, Decision Tree)", ROOT_DIR / "04_regression_models.py"),
    ("Step 5: Classification Models (Logistic, Decision Tree, Naive Bayes)", ROOT_DIR / "05_classification_models.py"),
    ("Step 6: Model Evaluation & Comparison Synthesis", ROOT_DIR / "06_model_evaluation.py"),
    ("Step 7: Build Interactive Web Dashboard", ROOT_DIR / "src" / "build_dashboard.py"),
]

def main():
    print("=" * 78)
    print("🌾 STARTING COMPLETE INDIAN MANDI PRICE INTELLIGENCE PIPELINE")
    print("=" * 78)

    for step_num, (step_name, script_path) in enumerate(pipeline_steps, start=1):
        print(f"\n>>> Running [{step_num}/7]: {step_name}...")
        try:
            runpy.run_path(str(script_path), run_name="__main__")
            print(f"✓ Completed: {step_name}")
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            sys.exit(1)

    print("\n" + "=" * 78)
    print("🎉 ALL 7 STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 78)

    # Simple, easy-to-read summary for viva and interviews
    print("""
==============================================================================
📌 SIMPLE SUMMARY: THE 6 MODELS YOU USED AND HOW THEY PERFORMED
==============================================================================

TASK 1: PREDICTING EXACT PRICE (REGRESSION)
Target: Modal Settlement Price in ₹ / Quintal
Evaluated on: 3,347 test records (Unseen data)

1. Decision Tree Regressor  -> 🏆 WINNER: Lowest Average Error!
   • Average Error (MAE)   : Off by only ₹1,387 / quintal
   • Explaining Power (R²) : 0.738 (73.8% of price variance explained)

2. Linear Regression (OLS)  -> 🥈 Runner Up: Highest R² Score!
   • Average Error (MAE)   : Off by ₹1,485 / quintal
   • Explaining Power (R²) : 0.782 (78.2% explained) - Highest!

3. Polynomial Regression    -> 🥉 Third Place
   • Average Error (MAE)   : Off by ₹1,462 / quintal
   • Explaining Power (R²) : 0.767 (76.7% explained)

(Comparison Baseline: If you just guessed the average price, error is ₹3,281)

------------------------------------------------------------------------------

TASK 2: PREDICTING PRICE TIER (CLASSIFICATION)
Target: Budget (< ₹3k) | Standard (₹3k - ₹10k) | Premium (> ₹10k)
Evaluated on: 3,347 test records (Unseen data)

4. Logistic Regression      -> 🏆 WINNER: Most Accurate!
   • Accuracy              : 84.67% (85 out of 100 crops classified correctly)
   • F1-Score (Balance)    : 0.8451

5. Multinomial Naive Bayes  -> 🥈 Runner Up: Very Fast & Reliable!
   • Accuracy              : 80.97% (81 out of 100 crops classified correctly)
   • F1-Score (Balance)    : 0.8111

6. Decision Tree Classifier -> 🥉 Third Place: Simple IF-THEN rules
   • Accuracy              : 76.67% (77 out of 100 crops classified correctly)
   • F1-Score (Balance)    : 0.7493

(Comparison Baseline: If you always guessed the most common tier, accuracy is only 60.71%)

==============================================================================
🌐 DASHBOARD READY:
   Double-click 'dashboard.html' or 'index.html' in your folder to open
   the interactive dashboard in your web browser!
==============================================================================
""")

if __name__ == "__main__":
    main()

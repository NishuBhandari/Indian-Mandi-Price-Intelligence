"""
Legacy runner for Regression Models
Delegates to src/04_regression_models.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / "src" / "04_regression_models.py"
    runpy.run_path(str(script_path), run_name="__main__")

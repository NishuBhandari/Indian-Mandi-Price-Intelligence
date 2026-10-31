"""
Root runner for Module 02: Data Preprocessing & Leakage Prevention
Delegates to src/02_data_preprocessing.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / "src" / "02_data_preprocessing.py"
    runpy.run_path(str(script_path), run_name="__main__")

"""
Root runner for Module 01: Data Audit & Hygiene
Delegates to src/01_data_audit.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / "src" / "01_data_audit.py"
    runpy.run_path(str(script_path), run_name="__main__")

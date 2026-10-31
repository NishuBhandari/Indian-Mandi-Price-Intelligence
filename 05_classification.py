"""
Legacy runner for Classification Models
Delegates to src/05_classification_models.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / "src" / "05_classification_models.py"
    runpy.run_path(str(script_path), run_name="__main__")

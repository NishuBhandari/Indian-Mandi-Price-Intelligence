"""
Root runner for Module 06: Model Evaluation & Synthesis
Delegates to src/06_model_evaluation.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / "src" / "06_model_evaluation.py"
    runpy.run_path(str(script_path), run_name="__main__")

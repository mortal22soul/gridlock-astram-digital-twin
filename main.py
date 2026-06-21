"""
Entry point for the ASTraM Digital Twin Command Center.

Usage
-----
    uv run python main.py          # prints launch instructions
    uv run streamlit run app/app.py  # launches the dashboard directly
"""

import os
import sys


def main() -> None:
    print("ASTraM Mobility Digital Twin Command Center")
    print("=" * 44)
    print()
    print("To launch the dashboard, run:")
    print()
    print("    uv run streamlit run app/app.py")
    print()
    print("Models must be present in models/ before starting.")
    print("If missing, execute notebooks/01_eda_and_cleaning.ipynb")
    print("then notebooks/02_xgboost_model_training.ipynb first.")
    print("and notebooks/03_catboost_model_training.ipynb.")


if __name__ == "__main__":
    main()

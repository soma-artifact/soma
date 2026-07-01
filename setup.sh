#!/bin/bash
# setup.sh - Set up virtual environment and install dependencies cleanly.

set -e

echo "=== SOMA Environment Setup ==="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

echo "1. Creating virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "[✓] Virtual environment created successfully."
else
    echo "[✓] Virtual environment already exists."
fi

echo "2. Activating virtual environment..."
source .venv/bin/activate

echo "3. Upgrading pip..."
pip install --upgrade pip

echo "4. Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "5. Verifying package imports..."
python3 -c "
import numpy as np
import pandas as pd
import scipy
import sklearn
import imblearn
import matplotlib
import seaborn
import tabulate
import dit
import xgboost
print('[✓] All packages imported successfully!')
"

echo "=========================================================="
echo "Setup Complete! Activate your environment via:"
echo "    source .venv/bin/activate"
echo "=========================================================="

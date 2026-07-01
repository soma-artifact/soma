#!/bin/bash
# Top-level entry-point to reproduce all SOMA results.
# Ensures environment is set up and runs the python reproduction script.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  SOMA: Entropy-Calibrated Failure Detection Reproduction"
echo "=========================================================="

# Activate virtual environment if it exists, otherwise prompt to run setup.sh
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[!] Virtual environment (.venv) not found."
    echo "[*] Running setup.sh to initialize environment..."
    ./setup.sh
    source .venv/bin/activate
fi

# Run the python reproduction script
python reproduce_all.py

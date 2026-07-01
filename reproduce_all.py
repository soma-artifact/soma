#!/usr/bin/env python3
"""
Reproduce All Results
=====================

Sequentially executes the entire experimental pipeline to reproduce all tables,
figures, and statistical analysis reported in the paper.

Steps:
    1. Primary Datasets Evaluation (AI4I, C-MAPSS, SMD, Synthetic)
    2. NASA PROMISE Datasets Evaluation (CM1, JM1, PC1, MC2)
    3. Multi-Estimator Synergy Ratio Consistency Checks
    4. Efficiency Benchmarks
    5. Figure Generation

Usage:
    python reproduce_all.py
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

def run_command(cmd, desc):
    print("\n" + "=" * 70)
    print(f"  RUNNING: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)
    
    # Locate python in the virtual environment if present
    venv_python = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        cmd[0] = venv_python
    
    t0 = time.time()
    try:
        res = subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        t_elapsed = time.time() - t0
        print(f"\n[✓] Completed: {desc} in {t_elapsed:.1f}s")
        return t_elapsed
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Failed: {desc} (Exit code: {e.returncode})")
        sys.exit(e.returncode)

def main():
    t_start = time.time()
    timings = {}
    
    # 1. Primary evaluations
    timings["primary_eval"] = run_command(
        ["python", "scripts/run_all.py", "--quick"],
        "Primary Datasets Evaluation (Quick Mode)"
    )
    
    # 2. PROMISE evaluations
    timings["promise_eval"] = run_command(
        ["python", "scripts/run_full_experiments.py"],
        "NASA PROMISE Datasets Evaluation"
    )
    
    # 3. Multi-estimator SR consistency
    timings["multi_estimator"] = run_command(
        ["python", "experiments/multi_estimator_sr.py"],
        "Multi-Estimator SR Consistency"
    )
    
    # 4. Efficiency benchmarks
    timings["efficiency"] = run_command(
        ["python", "experiments/benchmark_efficiency.py"],
        "Efficiency Benchmarking"
    )
    
    # 5. Generate figures
    timings["figures"] = run_command(
        ["python", "scripts/generate_figures.py"],
        "Regenerating Figures"
    )
    
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  REPRODUCTION SUMMARY")
    print("=" * 70)
    for desc, t in timings.items():
        print(f"  - {desc:<25}: {t:>6.1f}s")
    print(f"  - Total Execution Time    : {total_time:>6.1f}s")
    print("=" * 70)
    print("\n[✓] All tables, figures, and analyses reproduced successfully.")
    print("    Results are saved under /results/tables/ and /results/figures/\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full Evaluation on NASA PROMISE Datasets
==========================================

Runs the complete SOMA pipeline (PID decomposition + Bi-Level SGD +
baselines) on the NASA PROMISE software defect datasets (JM1, PC1, CM1, MC2).

Usage:
    python scripts/run_full_experiments.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from datasets.cm1.loader import load_cm1_grouped, load_cm1
from datasets.jm1.loader import load_jm1_grouped, load_jm1
from datasets.pc1.loader import load_pc1_grouped, load_pc1
from datasets.mc2.loader import load_mc2_grouped, load_mc2
from scripts.run_all import run_dataset_experiment, save_results_json, print_cross_dataset_summary

def main():
    print("\n" + "="*70)
    print("  PHASE 3: FULL EVALUATION ON PROMISE DATASETS")
    print("="*70)

    loaders = {
        "CM1": (load_cm1, load_cm1_grouped),
        "JM1": (load_jm1, load_jm1_grouped),
        "PC1": (load_pc1, load_pc1_grouped),
        "MC2": (load_mc2, load_mc2_grouped),
    }

    all_experiments = {}

    for ds_name, (load_fn, load_grouped_fn) in loaders.items():
        try:
            X, y, feature_names, groups_idx = load_fn()
            groups_raw, _ = load_grouped_fn()

            res = run_dataset_experiment(
                X=X, y=y,
                groups=groups_idx,
                groups_raw=groups_raw,
                dataset_name=ds_name,
                n_folds=10,
                verbose=True
            )
            all_experiments[ds_name] = res
        except Exception as e:
            print(f"  [ERROR] {ds_name}: {e}")
            import traceback; traceback.print_exc()

    if all_experiments:
        print_cross_dataset_summary(all_experiments)
        out_dir = str(ROOT / "results" / "tables")
        os.makedirs(out_dir, exist_ok=True)
        save_results_json(all_experiments, out_dir)

if __name__ == "__main__":
    main()

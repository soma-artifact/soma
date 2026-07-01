#!/usr/bin/env python3
"""
SOMA Ablation Study Script
==========================

Reproduces the SOMA ablation study (12D Full meta-feature vector vs 3D predictions-only)
and computes a paired t-test over cross-validation folds to demonstrate the statistical
significance of the information-theoretic features.

Usage:
    python scripts/run_ablation.py --quick
"""

import os
import sys
import argparse
import numpy as np
from scipy import stats
from tabulate import tabulate

# Ensure repository root is on the path so packages can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from soma_classifier.bilevel_sgd import evaluate_bilevel_sgd

def main():
    parser = argparse.ArgumentParser(
        description="Run SOMA ablation study (12D vs 3D) and paired t-test."
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["AI4I", "C-MAPSS", "SMD", "Synthetic"],
        choices=["AI4I", "C-MAPSS", "SMD", "Synthetic"],
        help="Datasets to evaluate (default: all four primary datasets)."
    )
    parser.add_argument(
        "--folds", "-f",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)."
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: run with 5 folds instead of 10 for rapid verification."
    )

    args = parser.parse_args()
    n_folds = 5 if args.quick else args.folds
    datasets = args.datasets

    print("=" * 75)
    print(f"  SOMA ABLATION STUDY: 12D (Full) vs 3D (No Information-Theoretic Features)")
    print("=" * 75)

    summary_data = []

    for ds in datasets:
        print(f"\n[*] Loading dataset: {ds}...")

        # Load dataset dynamically
        if ds == "AI4I":
            from datasets.ai4i.loader import load_ai4i
            X, y, _, groups_idx = load_ai4i()
        elif ds == "C-MAPSS":
            from datasets.cmapss.loader import load_cmapss
            X, y, _, groups_idx = load_cmapss()
        elif ds == "SMD":
            from datasets.smd.loader import load_smd
            X, y, _, groups_idx = load_smd()
        elif ds == "Synthetic":
            from datasets.synthetic.loader import generate_cascading_failures
            X, y, _, groups_idx = generate_cascading_failures()

        print(f"[✓] {ds} loaded. Samples={len(y)}, features={X.shape[1]}")

        # 1. Run Full 12D SOMA Model
        print(f"[*] Evaluating Full SOMA (12D meta-features)...")
        full_res = evaluate_bilevel_sgd(
            X, y, groups_idx,
            dataset_name=ds,
            n_folds=n_folds,
            use_entropy=True,
            verbose=False
        )

        # 2. Run Ablated 3D Model (Predictions only)
        print(f"[*] Evaluating Ablated SOMA (3D prediction-only)...")
        ablation_res = evaluate_bilevel_sgd(
            X, y, groups_idx,
            dataset_name=ds,
            n_folds=n_folds,
            use_entropy=False,
            verbose=False
        )

        # 3. Paired t-test over folds
        # A paired t-test is used because both models are evaluated on the exact same
        # cross-validation splits, making their fold-wise performance scores paired observations.
        t_stat, p_val = stats.ttest_rel(full_res.fold_aucs, ablation_res.fold_aucs)
        delta_auc = full_res.auc_mean - ablation_res.auc_mean
        sig = "✓ YES" if p_val < 0.05 else "✗ NO"

        print(f"[✓] Completed {ds}: Δ AUC = {delta_auc:+.4f}, t={t_stat:.3f}, p={p_val:.4f} [Significant={sig}]")

        summary_data.append([
            ds,
            f"{full_res.auc_mean:.4f} ± {full_res.auc_std:.4f}",
            f"{ablation_res.auc_mean:.4f} ± {ablation_res.auc_std:.4f}",
            f"{delta_auc:+.4f}",
            f"{t_stat:.3f}",
            f"{p_val:.4f}",
            sig
        ])

    print("\n" + "=" * 90)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 90)
    headers = ["Dataset", "Full 12D AUC", "Ablated 3D AUC", "Δ AUC", "t-statistic", "p-value", "Significant?"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()

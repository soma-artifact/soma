#!/usr/bin/env python3
"""
SOMA Classifier Pipeline Evaluation Script
==========================================

Runs the full SOMA classifier pipeline (10-fold Cross-Validation) on the four
primary datasets (AI4I, C-MAPSS, SMD, Synthetic) and compares it with
standard baselines (Naive Bayes, Logistic Regression, Random Forest, XGBoost).

Usage:
    python scripts/run_soma_evaluation.py --quick
"""

import os
import sys
import argparse
import numpy as np
from tabulate import tabulate

# Ensure repository root is on the path so packages can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from soma_classifier.bilevel_sgd import evaluate_bilevel_sgd
from scripts.run_all import evaluate_baselines

def main():
    parser = argparse.ArgumentParser(
        description="Run SOMA evaluation on the four primary datasets."
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
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline evaluations (run SOMA only)."
    )

    args = parser.parse_args()
    n_folds = 5 if args.quick else args.folds
    datasets = args.datasets

    print("=" * 70)
    print(f"  SOMA CLASSIFIER EVALUATION ({n_folds}-Fold CV)")
    print("=" * 70)

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

        # Run SOMA evaluation
        print(f"[*] Training SOMA Bi-Level SGD Meta-Classifier on {ds}...")
        soma_res = evaluate_bilevel_sgd(
            X, y, groups_idx,
            dataset_name=ds,
            n_folds=n_folds,
            use_entropy=True,
            verbose=False
        )

        row = [
            ds,
            f"{soma_res.auc_mean:.4f} ± {soma_res.auc_std:.4f}",
            f"{soma_res.f1_mean:.4f}",
            f"{soma_res.mcc_mean:.4f}"
        ]

        # Run Baselines if requested
        if not args.no_baselines:
            print(f"[*] Training baseline models (NB, LR, RF, XGBoost) on {ds}...")
            base_results = evaluate_baselines(X, y, ds, n_folds=n_folds, verbose=False)
            
            xgb = base_results.get("XGBoost (GB)")
            rf = base_results.get("Random Forest")
            lr = base_results.get("Logistic Reg.")
            
            row.extend([
                f"{xgb.auc_mean:.4f} ± {xgb.auc_std:.4f}" if xgb else "N/A",
                f"{rf.auc_mean:.4f} ± {rf.auc_std:.4f}" if rf else "N/A",
                f"{lr.auc_mean:.4f} ± {lr.auc_std:.4f}" if lr else "N/A",
            ])
            
        summary_data.append(row)

    # Print results
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)
    
    headers = ["Dataset", "SOMA AUC", "SOMA F1", "SOMA MCC"]
    if not args.no_baselines:
        headers.extend(["XGBoost AUC", "Random Forest AUC", "Logistic Reg. AUC"])

    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

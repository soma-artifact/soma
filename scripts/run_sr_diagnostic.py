#!/usr/bin/env python3
"""
Synergy Ratio (SR) Diagnostic Script
====================================

Computes the Synergy Ratio and PID atom breakdown for any of the eight datasets:
  - AI4I
  - C-MAPSS
  - SMD
  - Synthetic
  - CM1
  - JM1
  - PC1
  - MC2

Usage:
    python scripts/run_sr_diagnostic.py --dataset AI4I
"""

import os
import sys
import argparse
import numpy as np

# Ensure repository root is on the path so packages can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from sr_computation.pid_decomposition import compute_synergy_diagnostic

def main():
    parser = argparse.ArgumentParser(
        description="Compute the Synergy Ratio (SR) and PID atom breakdown for SOMA datasets."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["all", "AI4I", "C-MAPSS", "SMD", "Synthetic", "CM1", "JM1", "PC1", "MC2"],
        help="The name of the dataset to analyze (default: 'all' to run all datasets)."
    )
    parser.add_argument(
        "--bins", "-b",
        type=int,
        default=8,
        help="Number of bins for quantile discretization (default: 8). See Methodology - quantile binning B=8."
    )
    parser.add_argument(
        "--bootstrap", "-n",
        type=int,
        default=50,
        help="Number of bootstrap iterations for 95% Confidence Interval (default: 50, chosen for execution speed)."
    )
    
    args = parser.parse_args()
    selected_ds = args.dataset

    if selected_ds == "all":
        datasets_to_run = ["AI4I", "C-MAPSS", "SMD", "Synthetic", "CM1", "JM1", "PC1", "MC2"]
    else:
        datasets_to_run = [selected_ds]

    # Try to use dit/BROJA if available, otherwise fallback to I_min
    use_dit = False
    try:
        import dit
        use_dit = True
        print("[✓] 'dit' library found. Using BROJA estimator.")
    except ImportError:
        print("[!] 'dit' library not found. Falling back to Williams-Beer I_min estimator.")

    for ds in datasets_to_run:
        print(f"\n[*] Loading dataset: {ds}...")

        # Load dataset dynamically based on selection
        if ds == "AI4I":
            from datasets.ai4i.loader import load_ai4i_grouped
            groups, y = load_ai4i_grouped()
        elif ds == "C-MAPSS":
            from datasets.cmapss.loader import load_cmapss_grouped
            groups, y = load_cmapss_grouped()
        elif ds == "SMD":
            from datasets.smd.loader import load_smd_grouped
            groups, y = load_smd_grouped()
        elif ds == "Synthetic":
            from datasets.synthetic.loader import generate_grouped
            groups, y = generate_grouped(mode="cascading")
        elif ds == "CM1":
            from datasets.cm1.loader import load_cm1_grouped
            groups, y = load_cm1_grouped()
        elif ds == "JM1":
            from datasets.jm1.loader import load_jm1_grouped
            groups, y = load_jm1_grouped()
        elif ds == "PC1":
            from datasets.pc1.loader import load_pc1_grouped
            groups, y = load_pc1_grouped()
        elif ds == "MC2":
            from datasets.mc2.loader import load_mc2_grouped
            groups, y = load_mc2_grouped()
        else:
            print(f"[!] Unknown dataset: {ds}")
            continue

        print(f"[✓] Dataset {ds} loaded successfully. Samples={len(y)}, failure/defect rate={y.mean():.1%}")

        # Compute synergy ratio and PID atoms
        # We use a default max_samples of 2000 for the PID computation, because:
        # 1. PID calculations scale poorly with sample size (especially when using dit/BROJA)
        # 2. Empirical results demonstrate that the Synergy Ratio stabilizes by N=2000
        print(f"[*] Computing Synergy Ratio and PID atom breakdown (discretization bins={args.bins})...")

        diagnostic = compute_synergy_diagnostic(
            groups, y,
            dataset_name=ds,
            n_bins=args.bins,
            use_dit=use_dit,
            bootstrap_n=args.bootstrap,
            max_samples=2000
        )

        print("\n" + "=" * 60)
        print(f"  RESULTS FOR DATASET: {ds}")
        print("" + "=" * 60)
        print(f"  Synergy Ratio (SR): {diagnostic.synergy_ratio:.4f}")
        if diagnostic.synergy_ratio_ci:
            print(f"  95% Confidence Interval: [{diagnostic.synergy_ratio_ci[0]:.4f}, {diagnostic.synergy_ratio_ci[1]:.4f}]")
        else:
            print("  95% Confidence Interval: N/A")
        
        print("\n  Pairwise PID Breakdown (in bits):")
        for pid in diagnostic.pairwise_pids:
            print(f"    - {pid.source_i} x {pid.source_j}:")
            print(f"      Redundancy: {pid.redundancy:.4f}")
            print(f"      Unique {pid.source_i}: {pid.unique_i:.4f}")
            print(f"      Unique {pid.source_j}: {pid.unique_j:.4f}")
            print(f"      Synergy: {pid.synergy:.4f}")
        
        print("\n  Diagnostic interpretation:")
        if diagnostic.synergy_ratio < 0.05:
            print("    Regime: LOW SYNERGY")
            print("    Interpretation: Individual variables or redundant features dominate. A simple meta-classifier is sufficient.")
        elif diagnostic.synergy_ratio < 0.15:
            print("    Regime: MODERATE SYNERGY")
            print("    Interpretation: Minimal cross-group interaction. Fusion models may give marginal improvements.")
        else:
            print("    Regime: HIGH SYNERGY")
            print("    Interpretation: Failures arise from group interactions. Complex fusion or SOMA meta-classification is highly justified.")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

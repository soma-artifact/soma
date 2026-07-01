#!/usr/bin/env python3
"""
Surprisingness-Weighted Synergy Ratio (SR_φ) Experiment
========================================================

Tests whether weighting each pair's synergy by a *surprisingness factor* φ
can correct anomalous SR values (e.g. AI4I's SR > 1) while preserving the
diagnostic ordering across datasets.

Theory:
    For each pair (Xi, Xj) we define:

        φ(Xi, Xj; Y) = [H(Y) − H(Y | Xi, Xj)] / H(Y)

    φ ∈ [0, 1]:
        φ → 1  means observing Xi,Xj together nearly resolves Y
               → synergy is genuine
        φ → 0  means the pair tells us nothing new about Y
               → synergy is noise/artifact

    The modified Synergy Ratio becomes:

        SR_φ = Σ_{i<j} [Syn(Xi, Xj; Y) · φ(Xi, Xj; Y)]  /  Σ_k I(Xk; Y)

    Only the numerator changes; the denominator is unchanged.

    This is theoretically grounded in Williams & Beer (2010), who introduced
    the concept of surprise (log(1/p(s))) inside specific information.  We
    lift that idea to the pairwise synergy level.

Usage:
    python experiments/run_surprisingness_sr.py

Note:
    This experiment is standalone and does NOT modify any existing code
    in soma/ or the existing experiment results.

References:
    Williams, P.L. & Beer, R.D. (2010). Nonnegative decomposition of
        multivariate information. arXiv:1004.2515.
"""

import os
import sys
import json
import numpy as np
from collections import OrderedDict
from tabulate import tabulate

# ── Path setup ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from soma_classifier.entropy_features import discretize_features, mutual_information_discrete
from sr_computation.pid_decomposition import compute_pairwise_pid
from datasets.ai4i.loader import load_ai4i_grouped
from datasets.cmapss.loader import load_cmapss_grouped
from datasets.smd.loader import load_smd_grouped
from datasets.synthetic.loader import generate_grouped

np.random.seed(42)

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Information-Theoretic Helpers
# ═══════════════════════════════════════════════════════════════════════════

def shannon_entropy(Y: np.ndarray) -> float:
    """
    Compute Shannon entropy H(Y) in bits.

    H(Y) = − Σ_y P(y) · log₂ P(y)
    """
    n = len(Y)
    _, counts = np.unique(Y, return_counts=True)
    probs = counts / n
    # Filter zeros for log safety
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def conditional_entropy_joint(Xi: np.ndarray, Xj: np.ndarray,
                               Y: np.ndarray, n_bins: int = 8) -> float:
    """
    Compute H(Y | Xi, Xj) — conditional entropy of Y given the joint
    observation of discretized (Xi, Xj).

    H(Y | Xi, Xj) = Σ_{(xi,xj)} P(xi,xj) · H(Y | Xi=xi, Xj=xj)

    where H(Y | Xi=xi, Xj=xj) = − Σ_y P(y | xi,xj) · log₂ P(y | xi,xj)

    Args:
        Xi: Feature array for group i, shape (n_samples, n_features_i).
        Xj: Feature array for group j, shape (n_samples, n_features_j).
        Y:  Binary labels, shape (n_samples,).
        n_bins: Discretization bins.

    Returns:
        Conditional entropy H(Y | Xi, Xj) in bits.
    """
    if Xi.ndim == 1:
        Xi = Xi.reshape(-1, 1)
    if Xj.ndim == 1:
        Xj = Xj.reshape(-1, 1)

    n = len(Y)

    # Discretize both groups
    Xi_d = discretize_features(Xi, n_bins=n_bins)
    Xj_d = discretize_features(Xj, n_bins=n_bins)

    # Hash each group to a single integer state
    def hash_cols(X_d):
        state = np.zeros(len(X_d), dtype=np.int64)
        for col_idx in range(X_d.shape[1]):
            col = X_d[:, col_idx].astype(np.int64)
            state = state * (col.max() + 1) + col
        return state

    xi_state = hash_cols(Xi_d)
    xj_state = hash_cols(Xj_d)

    # Create a joint state from (xi_state, xj_state)
    # Use a large multiplier to avoid collisions
    max_xj = xj_state.max() + 1
    joint_state = xi_state * max_xj + xj_state

    # For each unique joint state, compute the local entropy of Y
    unique_joints = np.unique(joint_state)
    h_cond = 0.0

    for js in unique_joints:
        mask = (joint_state == js)
        count_js = np.sum(mask)
        p_js = count_js / n  # P(xi, xj)

        # Y values within this joint state
        y_local = Y[mask]
        _, y_counts = np.unique(y_local, return_counts=True)
        y_probs = y_counts / count_js

        # Local entropy H(Y | Xi=xi, Xj=xj)
        y_probs = y_probs[y_probs > 0]
        h_local = -np.sum(y_probs * np.log2(y_probs))

        h_cond += p_js * h_local

    return h_cond


def surprisingness_factor(Xi: np.ndarray, Xj: np.ndarray,
                           Y: np.ndarray, n_bins: int = 8) -> float:
    """
    Compute the surprisingness factor φ(Xi, Xj; Y).

        φ = [H(Y) − H(Y | Xi, Xj)] / H(Y)

    Returns a value in [0, 1].  If H(Y) = 0 (degenerate), returns 0.
    """
    h_y = shannon_entropy(Y)
    if h_y < 1e-12:
        return 0.0

    h_y_given_pair = conditional_entropy_joint(Xi, Xj, Y, n_bins=n_bins)
    phi = (h_y - h_y_given_pair) / h_y

    # Clip to [0, 1] for numerical safety
    return float(np.clip(phi, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all_datasets() -> dict:
    """Load all 4 datasets as {name: (groups_dict, Y)}."""
    datasets = OrderedDict()

    print("  Loading datasets...")

    groups, y = load_ai4i_grouped()
    datasets["AI4I"] = (groups, y)
    print(f"    ✓ AI4I — {len(y):,} samples")

    groups, y = load_cmapss_grouped()
    datasets["C-MAPSS"] = (groups, y)
    print(f"    ✓ C-MAPSS — {len(y):,} samples")

    groups, y = generate_grouped(mode="cascading")
    datasets["Synthetic (Cascading)"] = (groups, y)
    print(f"    ✓ Synthetic — {len(y):,} samples")

    groups, y = load_smd_grouped()
    datasets["SMD"] = (groups, y)
    print(f"    ✓ SMD — {len(y):,} samples")

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment():
    """Run the full Surprisingness-Weighted SR experiment."""

    print(f"\n{'#'*64}")
    print(f"  EXPERIMENT: Surprisingness-Weighted Synergy Ratio (SR_φ)")
    print(f"{'#'*64}\n")

    datasets = load_all_datasets()

    n_bins = 8
    max_samples = 2000  # Match the existing PID pipeline's subsampling

    all_results = OrderedDict()

    for ds_name, (groups, Y) in datasets.items():
        print(f"\n{'─'*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'─'*60}")

        group_names = list(groups.keys())
        group_arrays = list(groups.values())

        # ── Subsample (same strategy as pid_decomposition.py) ────────
        n_total = len(Y)
        if max_samples > 0 and n_total > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_total, size=max_samples, replace=False)
            Y_sub = Y[idx]
            group_arrays_sub = [arr[idx] for arr in group_arrays]
        else:
            Y_sub = Y
            group_arrays_sub = group_arrays

        # ── H(Y) ─────────────────────────────────────────────────────
        h_y = shannon_entropy(Y_sub.astype(int))
        print(f"\n  H(Y) = {h_y:.4f} bits")

        # ── Per-group MI (denominator) ────────────────────────────────
        mi_per_group = {}
        for name, X_g in zip(group_names, group_arrays_sub):
            if X_g.ndim == 1:
                X_g = X_g.reshape(-1, 1)
            X_g_d = discretize_features(X_g, n_bins=n_bins)
            mi = mutual_information_discrete(X_g_d, Y_sub.astype(int))
            mi_per_group[name] = mi

        total_mi = sum(mi_per_group.values())
        print(f"  Σ I(Xk; Y) = {total_mi:.4f} bits")
        for g, mi in mi_per_group.items():
            print(f"    I({g}; Y) = {mi:.4f}")

        # ── Pairwise PID + Surprisingness ─────────────────────────────
        pairs = [(0, 1), (0, 2), (1, 2)]
        pair_details = []
        total_syn_original = 0.0
        total_syn_weighted = 0.0

        print(f"\n  Per-Pair Analysis:")
        print(f"  {'Pair':<28} {'Syn':>8} {'φ':>8} {'Syn·φ':>10}")
        print(f"  {'─'*56}")

        for i, j in pairs:
            # Original PID
            pid = compute_pairwise_pid(
                group_arrays_sub[i], group_arrays_sub[j], Y_sub,
                name_i=group_names[i], name_j=group_names[j],
                n_bins=n_bins, use_dit=False,
                max_samples=0  # already subsampled
            )

            # Surprisingness factor
            phi = surprisingness_factor(
                group_arrays_sub[i], group_arrays_sub[j],
                Y_sub, n_bins=n_bins
            )

            syn_weighted = pid.synergy * phi
            total_syn_original += pid.synergy
            total_syn_weighted += syn_weighted

            pair_label = f"{group_names[i]} × {group_names[j]}"
            print(f"  {pair_label:<28} {pid.synergy:>8.4f} {phi:>8.4f} {syn_weighted:>10.4f}")

            # H(Y|Xi,Xj) for reporting
            h_cond = conditional_entropy_joint(
                group_arrays_sub[i], group_arrays_sub[j],
                Y_sub, n_bins=n_bins
            )

            pair_details.append({
                "pair": f"{group_names[i]}×{group_names[j]}",
                "synergy_original": float(pid.synergy),
                "redundancy": float(pid.redundancy),
                "unique_i": float(pid.unique_i),
                "unique_j": float(pid.unique_j),
                "H_Y": float(h_y),
                "H_Y_given_pair": float(h_cond),
                "phi": float(phi),
                "synergy_weighted": float(syn_weighted),
            })

        # ── SR computations ───────────────────────────────────────────
        sr_original = total_syn_original / total_mi if total_mi > 0 else 0.0
        sr_phi = total_syn_weighted / total_mi if total_mi > 0 else 0.0

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  SR (original)  = {sr_original:>8.4f}               │")
        print(f"  │  SR_φ (weighted) = {sr_phi:>8.4f}               │")
        if sr_original > 0:
            reduction = (1 - sr_phi / sr_original) * 100
            print(f"  │  Reduction       = {reduction:>7.1f}%               │")
        print(f"  └─────────────────────────────────────────┘")

        # Interpretation
        if sr_original > 1.0 and sr_phi <= 1.0:
            print(f"  ✓ FIXED: SR pulled from {sr_original:.3f} back to {sr_phi:.3f} (below 1.0)")
        elif sr_phi < 0.05:
            print(f"  → LOW SR_φ — simple model sufficient (consistent with original)")
        elif sr_phi < 0.15:
            print(f"  → MODERATE SR_φ — fusion optional")
        else:
            print(f"  → HIGH SR_φ — fusion justified")

        all_results[ds_name] = {
            "H_Y": float(h_y),
            "mi_per_group": mi_per_group,
            "total_mi": float(total_mi),
            "sr_original": float(sr_original),
            "sr_phi": float(sr_phi),
            "total_synergy_original": float(total_syn_original),
            "total_synergy_weighted": float(total_syn_weighted),
            "pair_details": pair_details,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Cross-Dataset Summary Table
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*64}")
    print(f"  CROSS-DATASET COMPARISON: SR vs SR_φ")
    print(f"{'='*64}\n")

    headers = ["Dataset", "SR (original)", "SR_φ (weighted)", "Reduction %", "φ̄ (mean)", "Status"]
    rows = []

    for ds_name, res in all_results.items():
        sr_orig = res["sr_original"]
        sr_new = res["sr_phi"]
        reduction = (1 - sr_new / sr_orig) * 100 if sr_orig > 0 else 0.0
        mean_phi = np.mean([p["phi"] for p in res["pair_details"]])

        if sr_orig > 1.0 and sr_new <= 1.0:
            status = "✓ FIXED"
        elif sr_orig > 1.0 and sr_new > 1.0:
            status = "⚠ STILL >1"
        elif sr_new < 0.05:
            status = "Low"
        elif sr_new < 0.15:
            status = "Moderate"
        else:
            status = "High"

        rows.append([
            ds_name,
            f"{sr_orig:.4f}",
            f"{sr_new:.4f}",
            f"{reduction:.1f}%",
            f"{mean_phi:.4f}",
            status,
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ── Key Finding ─────────────────────────────────────────────────
    ai4i = all_results.get("AI4I", {})
    print(f"\n  KEY FINDING:")
    print(f"  ─────────────")
    if ai4i:
        if ai4i["sr_phi"] <= 1.0:
            print(f"  ✓ AI4I SR corrected: {ai4i['sr_original']:.4f} → {ai4i['sr_phi']:.4f}")
            print(f"    The surprisingness weighting successfully discounts inflated synergy")
            print(f"    from entangled features where Y remains unpredictable.")
        else:
            print(f"  ⚠ AI4I SR_φ = {ai4i['sr_phi']:.4f} — still above 1.0.")
            print(f"    The φ-weighting reduces it but the entanglement is too strong")
            print(f"    for this correction alone.")

    print(f"\n  Theoretical grounding:")
    print(f"    φ derives from Williams & Beer's specific-information surprise concept,")
    print(f"    lifted from individual outcomes to pairwise synergy contributions.")
    print(f"    It asks: is the synergy genuinely *new* information for Y?")

    # ── Save JSON ───────────────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "surprisingness_sr_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → Results saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in Sanity Check (the 8-row example from the proposal)
# ═══════════════════════════════════════════════════════════════════════════

def run_sanity_check():
    """
    Verify φ computation against the hand-worked 8-row example.

    Data:
        X1 (Temp)   X2 (Pressure)   Y (Failure)
        Low(0)      Low(0)          0
        Low(0)      Low(0)          0
        Low(0)      High(1)         0
        Low(0)      High(1)         1
        High(1)     Low(0)          0
        High(1)     Low(0)          1
        High(1)     High(1)         1
        High(1)     High(1)         1

    Expected:
        H(Y) = 1.0 bit
        H(Y|X1,X2) = 0.5 bits
        φ = 0.5
    """
    print(f"\n{'─'*60}")
    print(f"  SANITY CHECK: 8-row Temperature/Pressure Example")
    print(f"{'─'*60}")

    X1 = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)
    X2 = np.array([0, 0, 1, 1, 0, 0, 1, 1]).reshape(-1, 1)
    Y  = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    h_y = shannon_entropy(Y)
    h_cond = conditional_entropy_joint(X1, X2, Y, n_bins=2)
    phi = surprisingness_factor(X1, X2, Y, n_bins=2)

    print(f"  H(Y)          = {h_y:.4f} bits  (expected: 1.0000)")
    print(f"  H(Y|X1,X2)    = {h_cond:.4f} bits  (expected: 0.5000)")
    print(f"  φ(X1,X2; Y)   = {phi:.4f}       (expected: 0.5000)")

    # Assertions
    assert abs(h_y - 1.0) < 1e-6, f"H(Y) should be 1.0, got {h_y}"
    assert abs(h_cond - 0.5) < 1e-6, f"H(Y|X1,X2) should be 0.5, got {h_cond}"
    assert abs(phi - 0.5) < 1e-6, f"φ should be 0.5, got {phi}"
    print(f"  ✓ All sanity checks passed!")


if __name__ == "__main__":
    run_sanity_check()
    run_experiment()

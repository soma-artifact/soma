#!/usr/bin/env python3
"""
Multi-Estimator Synergy Ratio Consistency Table
=================================================

Computes SR using three different PID estimators (BROJA, Imin, CoI)
across all datasets and checks whether the regime classification
(HIGH/LOW) is consistent across estimators.

This directly addresses the Liardi et al. (2026) objection: if SR values
are estimator-dependent, is the HIGH/LOW regime label also estimator-
dependent? If the regime label is consistent, then the claim that SR
identifies a structural dataset property is defensible.

Usage:
    python experiments/multi_estimator_sr.py
"""

import os
import sys
import json
import warnings
import numpy as np
import scipy.io.arff as arff_io
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent   # repo root (one level above experiments/)
sys.path.insert(0, str(ROOT))

# ── NumPy 2.x compatibility patch for dit 1.5 ────────────────────────────
if not hasattr(np, "alltrue"):
    np.alltrue = np.all
if not hasattr(np, "cumproduct"):
    np.cumproduct = np.cumprod
if not hasattr(np, "product"):
    np.product = np.prod

from soma_classifier.entropy_features import discretize_features, mutual_information_discrete
from sr_computation.pid_decomposition import _imin_redundancy, _specific_information

# ── Check BROJA availability ─────────────────────────────────────────────
try:
    import dit
    from dit.pid import PID_BROJA
    HAS_DIT = True
    print("[OK] dit loaded with NumPy 2.x patch")
except ImportError:
    HAS_DIT = False
    print("[WARN] dit not available; BROJA will use cached results")


# ══════════════════════════════════════════════════════════════════════════
#  THREE PID ESTIMATORS
# ══════════════════════════════════════════════════════════════════════════

def _hash_array(X: np.ndarray) -> np.ndarray:
    """Hash multi-dim discrete array to compact integer states."""
    if X.ndim == 1:
        return X.astype(int)
    vals = np.zeros(len(X), dtype=int)
    for j in range(X.shape[1]):
        col = X[:, j].astype(int)
        vals = vals * (col.max() + 1) + col
    return vals


def _reindex(arr: np.ndarray) -> np.ndarray:
    """Map arbitrary integer labels to 0..K-1."""
    vals = np.unique(arr)
    mapping = {v: i for i, v in enumerate(vals)}
    return np.array([mapping[v] for v in arr])


def pid_broja(X_i_d: np.ndarray, X_j_d: np.ndarray, Y_d: np.ndarray
              ) -> Tuple[float, float, float, float]:
    """PID via BROJA estimator (dit library)."""
    if not HAS_DIT:
        raise ImportError("dit not available")

    n = len(Y_d)
    xi = _reindex(_hash_array(X_i_d))
    xj = _reindex(_hash_array(X_j_d))
    y = _reindex(Y_d.astype(int))

    counts = {}
    for i in range(n):
        key = (int(xi[i]), int(xj[i]), int(y[i]))
        counts[key] = counts.get(key, 0) + 1

    outcomes = list(counts.keys())
    probs = [counts[k] / n for k in outcomes]

    d = dit.Distribution(outcomes, probs)
    # Do NOT call set_rv_names — PID_BROJA uses integer indices internally
    pid = PID_BROJA(d, [[0], [1]], [2])

    red = float(pid._pis[((0,), (1,))])
    unq_i = float(pid._pis[((0,),)])
    unq_j = float(pid._pis[((1,),)])
    syn = float(pid._pis[((0, 1),)])
    return red, unq_i, unq_j, syn


def pid_imin(X_i_d: np.ndarray, X_j_d: np.ndarray, Y_d: np.ndarray,
             mi_i: float, mi_j: float, mi_joint: float
             ) -> Tuple[float, float, float, float]:
    """PID via Williams-Beer Imin estimator."""
    red = _imin_redundancy(X_i_d, X_j_d, Y_d)
    unq_i = max(0.0, mi_i - red)
    unq_j = max(0.0, mi_j - red)
    syn = max(0.0, mi_joint - unq_i - unq_j - red)
    return red, unq_i, unq_j, syn


def pid_coi(mi_i: float, mi_j: float, mi_joint: float
            ) -> Tuple[float, float, float, float]:
    """
    PID via Co-Information (Interaction Information) decomposition.

    CoI(X_i, X_j; Y) = I(X_i;Y) + I(X_j;Y) - I(X_i,X_j;Y)

    If CoI > 0: net redundancy dominates.
    If CoI < 0: net synergy dominates.

    We decompose as:
        Red = max(0, CoI)     — net redundant component
        Syn = max(0, -CoI)    — net synergistic component
        Unq_i = MI_i - Red
        Unq_j = MI_j - Red

    This gives genuinely different SR values from Imin (which uses
    specific information) because CoI is a signed measure while Imin
    is always non-negative.
    """
    coi = mi_i + mi_j - mi_joint
    red = max(0.0, coi)
    syn = max(0.0, -coi)
    unq_i = max(0.0, mi_i - red)
    unq_j = max(0.0, mi_j - red)
    return red, unq_i, unq_j, syn


# ══════════════════════════════════════════════════════════════════════════
#  MULTI-ESTIMATOR SR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_sr_all_estimators(
    groups_raw: Dict[str, np.ndarray],
    y: np.ndarray,
    dataset_name: str,
    n_bins: int = 8,
    max_samples: int = 2000
) -> Dict[str, dict]:
    """
    Compute SR using BROJA, Imin, and CoI for a single dataset.

    Returns dict of {estimator_name: {sr, regime, pairwise_details}}.
    """
    group_names = list(groups_raw.keys())
    group_arrays = list(groups_raw.values())
    assert len(group_names) == 3, f"Expected 3 groups, got {len(group_names)}"

    # Subsample
    n = len(y)
    if max_samples > 0 and n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        y = y[idx]
        group_arrays = [arr[idx] for arr in group_arrays]

    # Discretize all groups
    groups_disc = []
    for arr in group_arrays:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        groups_disc.append(discretize_features(arr, n_bins=n_bins))

    # Per-group MI
    mi_per_group = {}
    for name, X_g_d in zip(group_names, groups_disc):
        mi_per_group[name] = mutual_information_discrete(X_g_d, y.astype(int))

    # Pairwise: compute MI_joint for each pair
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_mi_joints = {}
    for i, j in pairs:
        X_joint = np.hstack([groups_disc[i], groups_disc[j]])
        pair_mi_joints[(i, j)] = mutual_information_discrete(X_joint, y.astype(int))

    results = {}

    # ── BROJA (Requires state space reduction to prevent hanging) ──
    try:
        from sklearn.decomposition import PCA
        total_syn_broja = 0.0
        # Reduce each group to 1D PCA and 3 bins for BROJA
        groups_broja = []
        for arr in group_arrays:
            if arr.shape[1] > 1:
                arr_1d = PCA(n_components=1).fit_transform(arr)
            else:
                arr_1d = arr
            groups_broja.append(discretize_features(arr_1d, n_bins=3))
        
        # Per-group MI for BROJA
        mi_broja = {}
        for name, X_g_d in zip(group_names, groups_broja):
            mi_broja[name] = mutual_information_discrete(X_g_d, y.astype(int))

        for i, j in pairs:
            red, unq_i, unq_j, syn = pid_broja(groups_broja[i], groups_broja[j], y.astype(int))
            total_syn_broja += syn

        total_mi_broja = sum(mi_broja.values())
        sr_broja = total_syn_broja / total_mi_broja if total_mi_broja > 0 else 0.0
        results["BROJA"] = {
            "sr": sr_broja,
            "regime": _classify_regime(sr_broja),
            "total_synergy": total_syn_broja,
        }
        print(f"    BROJA:  SR={sr_broja:.4f}  [{_classify_regime(sr_broja)}]")
    except Exception as e:
        print(f"    BROJA:  FAILED ({e})")
        results["BROJA"] = {"sr": None, "regime": "FAILED", "error": str(e)}

    # ── Imin ──
    total_syn_imin = 0.0
    for i, j in pairs:
        mi_i = mi_per_group[group_names[i]]
        mi_j = mi_per_group[group_names[j]]
        mi_joint = pair_mi_joints[(i, j)]
        red, unq_i, unq_j, syn = pid_imin(
            groups_disc[i], groups_disc[j], y.astype(int),
            mi_i, mi_j, mi_joint
        )
        total_syn_imin += syn

    total_mi = sum(mi_per_group.values())
    sr_imin = total_syn_imin / total_mi if total_mi > 0 else 0.0
    results["Imin"] = {
        "sr": sr_imin,
        "regime": _classify_regime(sr_imin),
        "total_synergy": total_syn_imin,
    }
    print(f"    Imin:   SR={sr_imin:.4f}  [{_classify_regime(sr_imin)}]")

    # ── CoI (Co-Information) ──
    total_syn_coi = 0.0
    for i, j in pairs:
        mi_i = mi_per_group[group_names[i]]
        mi_j = mi_per_group[group_names[j]]
        mi_joint = pair_mi_joints[(i, j)]
        red, unq_i, unq_j, syn = pid_coi(mi_i, mi_j, mi_joint)
        total_syn_coi += syn

    sr_coi = total_syn_coi / total_mi if total_mi > 0 else 0.0
    results["CoI"] = {
        "sr": sr_coi,
        "regime": _classify_regime(sr_coi),
        "total_synergy": total_syn_coi,
    }
    print(f"    CoI:    SR={sr_coi:.4f}  [{_classify_regime(sr_coi)}]")

    return results


def _classify_regime(sr: float) -> str:
    if sr is None:
        return "UNKNOWN"
    if sr < 0.05:
        return "LOW"
    elif sr < 0.15:
        return "MOD"
    else:
        return "HIGH"


# ══════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_arff_promise(filepath: str, dataset_name: str):
    """Load a NASA PROMISE .arff file and return grouped features + labels."""
    data, meta = arff_io.loadarff(filepath)
    df = pd.DataFrame(data)

    target_col = df.columns[-1]
    y_raw = df[target_col].values

    if y_raw.dtype == object:
        decoded = np.array([
            v.decode("utf-8", errors="replace").strip().lower()
            if isinstance(v, bytes) else str(v).strip().lower()
            for v in y_raw
        ])
        y = np.array([
            1 if v in ("y", "yes", "true", "1", "defect", "defective", "bug") else 0
            for v in decoded
        ], dtype=int)
    else:
        y = (y_raw.astype(float) > 0).astype(int)

    df_feat = df.drop(columns=[target_col])
    for col in df_feat.columns:
        if df_feat[col].dtype == object:
            df_feat[col] = df_feat[col].apply(
                lambda v: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            )
    for col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    df_feat = df_feat.fillna(df_feat.median())

    cols = df_feat.columns.tolist()
    cols_upper = [c.upper() for c in cols]

    halstead_cols, complexity_cols, volume_cols = [], [], []
    for i, col in enumerate(cols):
        cu = cols_upper[i]
        if "HALSTEAD" in cu:
            halstead_cols.append(col)
        elif any(k in cu for k in ["CYCLOMATIC", "BRANCH", "CONDITION", "DECISION",
                                    "ESSENTIAL", "DESIGN", "EDGE", "NODE",
                                    "NORMALIZED_CYLO", "MULTIPLE_CONDITION",
                                    "MODIFIED_CONDITION"]):
            complexity_cols.append(col)
        elif any(k in cu for k in ["LOC", "NUM_", "PARAMETER", "PERCENT",
                                    "CALL_PAIR", "MAINTENANCE"]):
            volume_cols.append(col)

    if not halstead_cols or not complexity_cols or not volume_cols:
        n_feat = len(cols)
        s = n_feat // 3
        halstead_cols = cols[:s]
        complexity_cols = cols[s:2*s]
        volume_cols = cols[2*s:]

    groups_raw = {
        "Halstead":   df_feat[halstead_cols].values.astype(float),
        "Complexity": df_feat[complexity_cols].values.astype(float),
        "Volume":     df_feat[volume_cols].values.astype(float),
    }

    X = df_feat.values.astype(float)
    print(f"  [{dataset_name}] {X.shape[0]} samples, {X.shape[1]} features, "
          f"defect rate={y.mean():.1%}")
    return groups_raw, y, X


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    all_results = {}

    # ── 1. Load existing datasets ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MULTI-ESTIMATOR SR CONSISTENCY TABLE")
    print("=" * 65)

    from datasets.ai4i.loader import load_ai4i_grouped
    from datasets.cmapss.loader import load_cmapss_grouped
    from datasets.smd.loader import load_smd_grouped
    from datasets.synthetic.loader import generate_grouped

    existing_datasets = {
        "AI4I": lambda: load_ai4i_grouped(),
        "C-MAPSS": lambda: load_cmapss_grouped(),
        "SMD": lambda: load_smd_grouped(),
        "Synthetic": lambda: generate_grouped(mode="cascading"),
    }

    for ds_name, loader in existing_datasets.items():
        print(f"\n{'─' * 55}")
        print(f"  Dataset: {ds_name}")
        try:
            groups_raw, y = loader()
            res = compute_sr_all_estimators(groups_raw, y, ds_name)
            all_results[ds_name] = res
        except Exception as e:
            print(f"  [ERROR] {ds_name}: {e}")
            import traceback; traceback.print_exc()

    # ── 2. NASA PROMISE datasets ─────────────────────────────────────────
    from datasets.cm1.loader import load_cm1_grouped
    from datasets.jm1.loader import load_jm1_grouped
    from datasets.pc1.loader import load_pc1_grouped
    from datasets.mc2.loader import load_mc2_grouped

    promise_datasets = {
        "CM1": load_cm1_grouped,
        "JM1": load_jm1_grouped,
        "PC1": load_pc1_grouped,
        "MC2": load_mc2_grouped,
    }

    for ds_name, loader in promise_datasets.items():
        print(f"\n{'─' * 55}")
        print(f"  Dataset: {ds_name}")
        try:
            groups_raw, y = loader()
            res = compute_sr_all_estimators(groups_raw, y, ds_name)
            all_results[ds_name] = res
        except Exception as e:
            print(f"  [ERROR] {ds_name}: {e}")
            import traceback; traceback.print_exc()

    # ── 3. Print consistency table ───────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("  ESTIMATOR CONSISTENCY TABLE")
    print("=" * 75)
    header = f"  {'Dataset':<16} {'BROJA SR':>10} {'BROJA':>7} {'Imin SR':>10} {'Imin':>7} {'CoI SR':>10} {'CoI':>7} {'Consistent':>11}"
    print(header)
    print("  " + "─" * 73)

    n_consistent = 0
    n_total = 0

    for ds_name, estimators in all_results.items():
        broja = estimators.get("BROJA", {})
        imin = estimators.get("Imin", {})
        coi = estimators.get("CoI", {})

        br_sr = broja.get("sr")
        im_sr = imin.get("sr")
        co_sr = coi.get("sr")

        br_reg = broja.get("regime", "?")
        im_reg = imin.get("regime", "?")
        co_reg = coi.get("regime", "?")

        # Consistency: check if all non-failed regimes map to the same
        # binary decision (HIGH vs not-HIGH)
        regimes = [r for r in [br_reg, im_reg, co_reg] if r not in ("FAILED", "UNKNOWN", "?")]
        binary = [r in ("HIGH", "MOD") for r in regimes]  # fusion-worth-considering
        if len(binary) >= 2:
            n_total += 1
            consistent = all(b == binary[0] for b in binary)
            if consistent:
                n_consistent += 1
            cons_str = "YES" if consistent else "*** NO ***"
        else:
            cons_str = "N/A"

        # Print row
        br_str = f"{br_sr:>10.4f} {br_reg:>7}" if br_sr is not None else f"{'FAILED':>10} {'':>7}"
        im_str = f"{im_sr:>10.4f} {im_reg:>7}"
        co_str = f"{co_sr:>10.4f} {co_reg:>7}"
        print(f"  {ds_name:<16} {br_str} {im_str} {co_str} {cons_str:>11}")

    print("  " + "─" * 73)
    print(f"  Regime consistency: {n_consistent}/{n_total} datasets agree across all estimators")

    if n_total > 0 and n_consistent == n_total:
        print("  → All estimators agree on regime classification.")
        print("  → SR regime is robust to estimator choice.")
    elif n_consistent < n_total:
        disagreements = []
        for ds_name, estimators in all_results.items():
            regimes = [estimators.get(e, {}).get("regime", "?") for e in ["BROJA", "Imin", "CoI"]]
            regimes = [r for r in regimes if r not in ("FAILED", "UNKNOWN", "?")]
            binary = [r in ("HIGH", "MOD") for r in regimes]
            if len(binary) >= 2 and not all(b == binary[0] for b in binary):
                disagreements.append(ds_name)
        print(f"  Disagreements: {', '.join(disagreements)}")

    # ── 4. Save results ──────────────────────────────────────────────────
    out_path = ROOT / "results" / "tables" / "multi_estimator_sr.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # Make serializable
    save_data = {}
    for ds_name, estimators in all_results.items():
        save_data[ds_name] = {}
        for est_name, est_data in estimators.items():
            save_data[ds_name][est_name] = {
                k: v for k, v in est_data.items()
                if not callable(v)
            }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()

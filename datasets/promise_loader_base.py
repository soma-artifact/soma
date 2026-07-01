#!/usr/bin/env python3
"""
Base loader for NASA PROMISE datasets in ARFF format.
"""

import numpy as np
import scipy.io.arff as arff_io
import pandas as pd
from typing import Tuple, Dict, List

def load_arff_promise(filepath: str, dataset_name: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Load a NASA PROMISE .arff file and return grouped features, labels, full X, feature names, and group indices.
    
    Semantic Groups (based on Halstead, Complexity, and Volume metrics):
        Halstead: HALSTEAD_* metrics
        Complexity: CYCLOMATIC_*, BRANCH_COUNT, CONDITION_COUNT, DECISION_*,
                    ESSENTIAL_*, DESIGN_*, EDGE_COUNT, NODE_COUNT
        Volume: LOC_*, PARAMETER_COUNT, NUM_*, PERCENT_COMMENTS

    Args:
        filepath: Path to the ARFF file.
        dataset_name: Name of the dataset.

    Returns:
        groups_raw: dict of {group_name: np.ndarray}
        y: binary labels (0=clean, 1=defect)
        X: full feature matrix
        feature_names: list of feature names
        groups_idx: dict of {group_name: list of column indices in X}
    """
    data, meta = arff_io.loadarff(filepath)
    df = pd.DataFrame(data)

    # 1. Extract label before column-wide decode (preserves bytes dtype)
    target_col = df.columns[-1]
    y_raw_bytes = df[target_col].values

    if y_raw_bytes.dtype == object:
        # bytes categorical: b'Y'/b'N' or b'true'/b'false'
        decoded = np.array([
            v.decode("utf-8", errors="replace").strip().lower()
            if isinstance(v, bytes) else str(v).strip().lower()
            for v in y_raw_bytes
        ])
        y = np.array([
            1 if v in ("y", "yes", "true", "1", "defect", "defective", "bug") else 0
            for v in decoded
        ], dtype=int)
    else:
        y = (y_raw_bytes.astype(float) > 0).astype(int)

    # 2. Decode feature columns
    df_feat = df.drop(columns=[target_col])
    for col in df_feat.columns:
        if df_feat[col].dtype == object:
            df_feat[col] = df_feat[col].apply(
                lambda v: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            )

    # 3. Convert to float and impute missing values with median
    for col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    df_feat = df_feat.fillna(df_feat.median())

    cols = df_feat.columns.tolist()
    cols_upper = [c.upper() for c in cols]

    # 4. Semantic grouping based on attribute names
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

    # Fallback to even split if groupings are empty
    if not halstead_cols or not complexity_cols or not volume_cols:
        n_feat = len(cols)
        s = n_feat // 3
        halstead_cols  = cols[:s]
        complexity_cols = cols[s:2*s]
        volume_cols    = cols[2*s:]

    groups_raw = {
        "Halstead":   df_feat[halstead_cols].values.astype(float),
        "Complexity": df_feat[complexity_cols].values.astype(float),
        "Volume":     df_feat[volume_cols].values.astype(float),
    }

    X = df_feat.values.astype(float)
    feature_names = cols
    
    groups_idx = {
        "Halstead": [feature_names.index(c) for c in halstead_cols],
        "Complexity": [feature_names.index(c) for c in complexity_cols],
        "Volume": [feature_names.index(c) for c in volume_cols],
    }

    return groups_raw, y, X, feature_names, groups_idx

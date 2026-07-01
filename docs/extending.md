# How to Extend SOMA

This guide explains how to add new datasets or models to the SOMA framework without modifying existing code.

---

## Adding a New Dataset

### Step 1: Create a Loader

Create a new file in `soma/datasets/`, e.g. `soma/datasets/my_dataset_loader.py`:

```python
import os
import numpy as np
from typing import Tuple, Dict, List


# Define semantic feature groups
GROUPS = {
    "Group_A": ["feature_1", "feature_2"],
    "Group_B": ["feature_3", "feature_4"],
    "Group_C": ["feature_5", "feature_6"],
}


def load_my_dataset(
    data_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Load your dataset.
    
    Returns:
        X: Feature array (n_samples, n_features)
        y: Binary labels (n_samples,)
        feature_names: Column names
        groups_idx: Dict mapping group names to column indices
    """
    if data_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(os.path.dirname(base), "data", "my_dataset")

    # Load your data here
    # ...
    
    return X, y, feature_names, groups_idx


def load_my_dataset_grouped(
    data_dir: str = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return grouped arrays for PID analysis."""
    X, y, _, groups_idx = load_my_dataset(data_dir)
    groups = {name: X[:, idx] for name, idx in groups_idx.items()}
    return groups, y
```

### Step 2: Place Your Data

Put your data file in `data/my_dataset/`.

### Step 3: Register in run_all.py

Add your dataset to the `run_all_experiments()` function in `experiments/run_all.py`:

```python
elif ds_name == "MyDataset":
    from soma.datasets.my_dataset_loader import load_my_dataset, load_my_dataset_grouped
    X, y, _, groups_idx = load_my_dataset()
    groups_raw, _ = load_my_dataset_grouped()
```

### Step 4: Run

```bash
python experiments/run_all.py --dataset MyDataset
```

---

## Adding a New Baseline Model

In `experiments/run_all.py`, add to the `baselines` dict in `evaluate_baselines()`:

```python
baselines = {
    # ... existing baselines ...
    "My New Model": MyModel(param1=value1, random_state=42),
}
```

The framework will automatically handle CV splits, SMOTE, scaling, and metric collection.

---

## Key Principle: Don't Touch Existing Code

When extending:
- ✅ Add new files in `soma/datasets/`
- ✅ Add new entries to experiment runners
- ✅ Create new experiment scripts in `experiments/`
- ❌ Don't modify `soma/core/` (the core algorithms)
- ❌ Don't change existing dataset loaders
- ❌ Don't alter hyperparameters in existing experiments

import os
from datasets.promise_loader_base import load_arff_promise

def load_cm1():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, "CM1.arff")
    groups_raw, y, X, feature_names, groups_idx = load_arff_promise(filepath, "CM1")
    return X, y, feature_names, groups_idx

def load_cm1_grouped():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, "CM1.arff")
    groups_raw, y, X, feature_names, groups_idx = load_arff_promise(filepath, "CM1")
    return groups_raw, y

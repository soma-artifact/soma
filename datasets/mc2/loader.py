import os
from datasets.promise_loader_base import load_arff_promise

def load_mc2():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, "MC2.arff")
    groups_raw, y, X, feature_names, groups_idx = load_arff_promise(filepath, "MC2")
    return X, y, feature_names, groups_idx

def load_mc2_grouped():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, "MC2.arff")
    groups_raw, y, X, feature_names, groups_idx = load_arff_promise(filepath, "MC2")
    return groups_raw, y

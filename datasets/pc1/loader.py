import os
from datasets.promise_loader_base import load_arff_promise

def load_pc1():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, "PC1.arff")
    groups_raw, y, X, feature_names, groups_idx = load_arff_promise(filepath, "PC1")
    return X, y, feature_names, groups_idx

def load_pc1_grouped():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir_path, "PC1.arff")
    groups_raw, y, X, feature_names, groups_idx = load_arff_promise(filepath, "PC1")
    return groups_raw, y

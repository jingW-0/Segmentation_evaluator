from typing import Dict, List
import numpy as np


def get_labels_from_volume(volume: np.ndarray) -> List[int]:
    """Return sorted non-zero integer labels present in the volume."""
    return sorted(set(np.unique(volume.astype(int)).tolist()) - {0})


def build_label_names(labels: List[int], names: Dict[int, str] = None) -> Dict[int, str]:
    """Map label ints to display names. Unknown labels get 'Class N'."""
    names = names or {}
    return {lbl: names.get(lbl, f"Class {lbl}") for lbl in labels}

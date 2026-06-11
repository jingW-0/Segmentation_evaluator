import numpy as np
from typing import Dict, List


def compute_metrics(gt: np.ndarray, pred: np.ndarray, class_labels: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class Dice, IoU, Precision, Recall over full 3D volumes.

    Args:
        gt:   integer label array (H, W, D)
        pred: integer label array (H, W, D)
        class_labels: list of class ints to evaluate; defaults to all non-zero labels in gt

    Returns:
        {class_id: {"dice": float, "iou": float, "precision": float, "recall": float}}
    """
    if class_labels is None:
        class_labels = sorted(set(np.unique(gt).tolist()) - {0})

    results = {}
    for cls in class_labels:
        g = gt == cls
        p = pred == cls

        tp = np.logical_and(g, p).sum()
        fp = np.logical_and(~g, p).sum()
        fn = np.logical_and(g, ~p).sum()

        dice      = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou       = tp / (tp + fp + fn)            if (tp + fp + fn) > 0      else 0.0
        precision = tp / (tp + fp)                 if (tp + fp) > 0           else 0.0
        recall    = tp / (tp + fn)                 if (tp + fn) > 0           else 0.0

        results[cls] = {
            "dice":      float(dice),
            "iou":       float(iou),
            "precision": float(precision),
            "recall":    float(recall),
        }

    return results

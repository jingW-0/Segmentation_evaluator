import numpy as np


def normalize_zscore(volume: np.ndarray) -> np.ndarray:
    mean, std = volume.mean(), volume.std()
    if std == 0:
        return volume - mean
    return (volume - mean) / std


def normalize_minmax(volume: np.ndarray) -> np.ndarray:
    lo, hi = volume.min(), volume.max()
    if hi == lo:
        return np.zeros_like(volume)
    return (volume - lo) / (hi - lo)

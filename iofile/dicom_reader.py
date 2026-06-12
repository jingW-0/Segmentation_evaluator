import os
import numpy as np
import pydicom


def load_dicom_series(folder: str) -> np.ndarray:
    """Load a DICOM series from a folder and return a (H, W, D) float32 array."""
    slices = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            ds = pydicom.dcmread(fpath)
            if hasattr(ds, "InstanceNumber"):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM files found in: {folder}")

    slices.sort(key=lambda s: int(s.InstanceNumber))
    volume = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)
    return volume  # (H, W, D)

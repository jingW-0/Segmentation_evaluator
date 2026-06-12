import nibabel as nib
import numpy as np


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file, reorient to RAS canonical, and return a (X, Y, Z) float32 array."""
    img = nib.load(path)
    # Reorient to RAS so axes are consistent regardless of scanner convention
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data

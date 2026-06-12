import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from iofile.dicom_reader import load_dicom_series
from iofile.nifti_reader import load_nifti


class NiftiReaderTests(unittest.TestCase):
    def test_load_nifti_canonicalizes_and_returns_float32_array(self):
        loaded_img = SimpleNamespace(dataobj=np.array([1], dtype=np.int16))
        canonical_img = SimpleNamespace(dataobj=np.array([[1, 2], [3, 4]], dtype=np.int16))

        with patch("iofile.nifti_reader.nib.load", return_value=loaded_img) as load_mock:
            with patch("iofile.nifti_reader.nib.as_closest_canonical", return_value=canonical_img) as canonical_mock:
                result = load_nifti("scan.nii.gz")

        load_mock.assert_called_once_with("scan.nii.gz")
        canonical_mock.assert_called_once_with(loaded_img)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, canonical_img.dataobj.astype(np.float32))


class DicomReaderTests(unittest.TestCase):
    def test_load_dicom_series_sorts_by_instance_number_and_stacks_slices(self):
        datasets = {
            "slice3.dcm": SimpleNamespace(InstanceNumber="3", pixel_array=np.full((2, 2), 30)),
            "slice1.dcm": SimpleNamespace(InstanceNumber="1", pixel_array=np.full((2, 2), 10)),
            "slice2.dcm": SimpleNamespace(InstanceNumber="2", pixel_array=np.full((2, 2), 20)),
            "metadata.txt": SimpleNamespace(pixel_array=np.full((2, 2), 99)),
        }

        def fake_dcmread(path):
            name = os.path.basename(path)
            if name == "broken.dcm":
                raise ValueError("not a dicom")
            return datasets[name]

        with patch("iofile.dicom_reader.os.listdir", return_value=["slice3.dcm", "broken.dcm", "slice1.dcm", "metadata.txt", "slice2.dcm"]):
            with patch("iofile.dicom_reader.pydicom.dcmread", side_effect=fake_dcmread):
                result = load_dicom_series("series")

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (2, 2, 3))
        np.testing.assert_array_equal(result[:, :, 0], np.full((2, 2), 10, dtype=np.float32))
        np.testing.assert_array_equal(result[:, :, 1], np.full((2, 2), 20, dtype=np.float32))
        np.testing.assert_array_equal(result[:, :, 2], np.full((2, 2), 30, dtype=np.float32))

    def test_load_dicom_series_raises_when_no_valid_slices_are_found(self):
        with patch("iofile.dicom_reader.os.listdir", return_value=["notes.txt"]):
            with patch("iofile.dicom_reader.pydicom.dcmread", return_value=SimpleNamespace()):
                with self.assertRaisesRegex(ValueError, "No valid DICOM files found"):
                    load_dicom_series("empty")


if __name__ == "__main__":
    unittest.main()

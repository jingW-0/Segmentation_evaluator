import unittest

import numpy as np

from inference.preprocessing import normalize_minmax, normalize_zscore


class PreprocessingTests(unittest.TestCase):
    def test_normalize_zscore_centers_and_scales_volume(self):
        volume = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = normalize_zscore(volume)

        self.assertAlmostEqual(float(result.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(result.std()), 1.0, places=6)

    def test_normalize_zscore_handles_constant_volume(self):
        volume = np.full((2, 2), 7.0, dtype=np.float32)

        np.testing.assert_array_equal(normalize_zscore(volume), np.zeros_like(volume))

    def test_normalize_minmax_scales_to_unit_range(self):
        volume = np.array([2.0, 4.0, 6.0], dtype=np.float32)

        np.testing.assert_allclose(normalize_minmax(volume), np.array([0.0, 0.5, 1.0], dtype=np.float32))

    def test_normalize_minmax_handles_constant_volume(self):
        volume = np.full((2, 3), 5.0, dtype=np.float32)

        np.testing.assert_array_equal(normalize_minmax(volume), np.zeros_like(volume))


if __name__ == "__main__":
    unittest.main()

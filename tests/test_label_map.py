import unittest

import numpy as np

from utils.label_map import build_label_names, get_labels_from_volume


class LabelMapTests(unittest.TestCase):
    def test_get_labels_from_volume_returns_sorted_non_zero_integer_labels(self):
        volume = np.array(
            [
                [2.9, 0.0, 1.2],
                [2.1, 3.0, 1.9],
            ],
            dtype=np.float32,
        )

        self.assertEqual(get_labels_from_volume(volume), [1, 2, 3])

    def test_build_label_names_uses_overrides_and_default_names(self):
        result = build_label_names([1, 2], names={2: "Tumor"})

        self.assertEqual(result, {1: "Class 1", 2: "Tumor"})


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from metrics.overlap import compute_metrics


class ComputeMetricsTests(unittest.TestCase):
    def test_computes_expected_metrics_for_non_background_labels(self):
        gt = np.array(
            [
                [1, 1, 0],
                [2, 0, 2],
            ]
        )
        pred = np.array(
            [
                [1, 0, 1],
                [2, 2, 0],
            ]
        )

        results = compute_metrics(gt, pred)

        self.assertEqual(set(results), {1, 2})
        for class_metrics in results.values():
            self.assertAlmostEqual(class_metrics["dice"], 0.5)
            self.assertAlmostEqual(class_metrics["iou"], 1 / 3)
            self.assertAlmostEqual(class_metrics["precision"], 0.5)
            self.assertAlmostEqual(class_metrics["recall"], 0.5)

    def test_uses_explicit_class_labels_even_when_missing_from_ground_truth(self):
        gt = np.zeros((2, 2), dtype=np.int32)
        pred = np.ones((2, 2), dtype=np.int32)

        results = compute_metrics(gt, pred, class_labels=[1])

        self.assertEqual(
            results[1],
            {
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            },
        )

    def test_returns_empty_results_when_ground_truth_has_only_background(self):
        gt = np.zeros((2, 2, 2), dtype=np.int32)
        pred = np.ones((2, 2, 2), dtype=np.int32)

        self.assertEqual(compute_metrics(gt, pred), {})


if __name__ == "__main__":
    unittest.main()

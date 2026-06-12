import unittest
from types import SimpleNamespace

import numpy as np

from inference.onnx_predictor import OnnxPredictor


class FakeSession:
    def __init__(self, output):
        self.output = output
        self.feed = None

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def run(self, output_names, feed):
        self.feed = feed
        return [self.output]


class OnnxPredictorTests(unittest.TestCase):
    def test_predict_requires_loaded_model(self):
        with self.assertRaisesRegex(RuntimeError, "Model not loaded"):
            OnnxPredictor().predict(np.zeros((2, 2, 1), dtype=np.float32))

    def test_predict_adds_batch_and_channel_axes_and_argmaxes_5d_output(self):
        output = np.array(
            [
                [
                    [[[0.1], [0.8]], [[0.2], [0.4]]],
                    [[[0.9], [0.2]], [[0.8], [0.6]]],
                ]
            ],
            dtype=np.float32,
        )
        session = FakeSession(output)
        predictor = OnnxPredictor()
        predictor._session = session
        volume = np.arange(4, dtype=np.float64).reshape(2, 2, 1)

        result = predictor.predict(volume)

        self.assertEqual(session.feed["image"].shape, (1, 1, 2, 2, 1))
        self.assertEqual(session.feed["image"].dtype, np.float32)
        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, np.array([[[1], [0]], [[1], [1]]], dtype=np.int32))

    def test_predict_returns_first_output_as_labels_for_non_channel_output(self):
        output = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        predictor = OnnxPredictor()
        predictor._session = FakeSession(output)

        result = predictor.predict(np.zeros((1, 2, 2), dtype=np.float32))

        np.testing.assert_array_equal(result, np.array([[[1, 2], [3, 4]]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()

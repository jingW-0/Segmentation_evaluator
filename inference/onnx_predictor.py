import numpy as np
from .base_predictor import BasePredictor


class OnnxPredictor(BasePredictor):
    def __init__(self):
        self._session = None

    def load(self, model_path: str) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def predict(self, volume: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Expect model input: (1, 1, H, W, D)
        inp = volume[np.newaxis, np.newaxis].astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: inp})[0]

        # output shape: (1, C, H, W, D) — argmax over channel dim
        if output.ndim == 5:
            return np.argmax(output[0], axis=0).astype(np.int32)
        return output[0].astype(np.int32)

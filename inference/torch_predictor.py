import numpy as np
from .base_predictor import BasePredictor


class TorchPredictor(BasePredictor):
    def __init__(self):
        self._model = None

    def load(self, model_path: str) -> None:
        import torch
        self._model = torch.load(model_path, map_location="cpu")
        self._model.eval()

    def predict(self, volume: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        # Expect model input: (1, 1, H, W, D)
        inp = torch.from_numpy(volume[np.newaxis, np.newaxis].astype(np.float32))
        with torch.no_grad():
            output = self._model(inp)

        # output shape: (1, C, H, W, D) — argmax over channel dim
        if output.ndim == 5:
            return output[0].argmax(dim=0).numpy().astype(np.int32)
        return output[0].numpy().astype(np.int32)

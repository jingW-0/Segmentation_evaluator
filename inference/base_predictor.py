from abc import ABC, abstractmethod
import numpy as np


class BasePredictor(ABC):
    """All inference backends must implement this interface."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model weights from path."""

    @abstractmethod
    def predict(self, volume: np.ndarray) -> np.ndarray:
        """
        Run inference on a (H, W, D) float32 volume.
        Returns an integer label array of the same shape (H, W, D).
        """

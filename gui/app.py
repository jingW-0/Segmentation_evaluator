import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStackedWidget, QMessageBox
)
from PyQt6.QtCore import Qt

from .file_panel import FilePanel
from .viewer_panel import ViewerPanel
from .results_panel import ResultsPanel
from .inference_panel import InferencePanel
from ..metrics.overlap import compute_metrics
from ..utils.label_map import get_labels_from_volume, build_label_names


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Image Segmentation Evaluator")
        self.resize(1200, 800)

        self._image = None
        self._gt    = None
        self._pred  = None

        # Panels
        self._file_panel      = FilePanel()
        self._viewer_panel    = ViewerPanel()
        self._results_panel   = ResultsPanel()
        self._inference_panel = InferencePanel()

        # Prediction source stack: index 0 = nothing, index 1 = inference panel
        self._pred_stack = QStackedWidget()
        self._pred_stack.addWidget(QWidget())          # placeholder for file mode
        self._pred_stack.addWidget(self._inference_panel)

        # Left column: file controls + inference panel (when in model mode)
        left = QVBoxLayout()
        left.addWidget(self._file_panel)
        left.addWidget(self._pred_stack)
        left.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(340)

        # Right column: viewer on top, results on bottom
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self._viewer_panel)
        right_splitter.addWidget(self._results_panel)
        right_splitter.setSizes([500, 300])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_splitter)
        splitter.setSizes([340, 860])

        central = QWidget()
        central.setLayout(QHBoxLayout())
        central.layout().addWidget(splitter)
        self.setCentralWidget(central)

        # Wire signals
        self._file_panel.image_loaded.connect(self._on_image)
        self._file_panel.gt_loaded.connect(self._on_gt)
        self._file_panel.pred_loaded.connect(self._on_pred)
        self._file_panel.mode_changed.connect(self._on_mode)
        self._file_panel.evaluate_btn.clicked.connect(self._evaluate)
        self._inference_panel.prediction_ready.connect(self._on_pred)

    def _on_image(self, vol: np.ndarray):
        self._image = vol
        self._viewer_panel.set_image(vol)
        self._inference_panel.set_volume(vol)

    def _on_gt(self, vol: np.ndarray):
        self._gt = vol
        self._viewer_panel.set_gt(vol)

    def _on_pred(self, vol: np.ndarray):
        self._pred = vol
        self._viewer_panel.set_pred(vol)

    def _on_mode(self, mode: str):
        self._pred_stack.setCurrentIndex(1 if mode == "model" else 0)

    def _evaluate(self):
        if self._gt is None or self._pred is None:
            QMessageBox.warning(self, "Missing data", "Please load both Ground Truth and Prediction before evaluating.")
            return

        gt   = self._gt.astype(np.int32)
        pred = self._pred.astype(np.int32)

        if gt.shape != pred.shape:
            QMessageBox.warning(self, "Shape mismatch",
                                f"GT shape {gt.shape} != Prediction shape {pred.shape}")
            return

        results     = compute_metrics(gt, pred)
        labels      = get_labels_from_volume(gt)
        label_names = build_label_names(labels)
        self._results_panel.update_results(results, label_names)

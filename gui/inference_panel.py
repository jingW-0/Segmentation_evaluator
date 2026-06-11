import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QProgressBar,
    QGroupBox, QRadioButton
)
from PyQt6.QtCore import QThread, pyqtSignal

from ..inference.preprocessing import normalize_zscore, normalize_minmax


class _InferenceWorker(QThread):
    finished = pyqtSignal(object)  # emits np.ndarray
    error = pyqtSignal(str)

    def __init__(self, predictor, volume):
        super().__init__()
        self._predictor = predictor
        self._volume = volume

    def run(self):
        try:
            result = self._predictor.predict(self._volume)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class InferencePanel(QWidget):
    prediction_ready = pyqtSignal(object)  # emits np.ndarray mask

    def __init__(self):
        super().__init__()
        self._predictor = None
        self._volume = None
        self._worker = None

        group = QGroupBox("Model Inference")
        inner = QVBoxLayout(group)

        # Backend selector
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self._backend = QComboBox()
        self._backend.addItems(["ONNX", "PyTorch"])
        self._backend.currentTextChanged.connect(self._on_backend_change)
        backend_row.addWidget(self._backend)
        backend_row.addStretch()
        inner.addLayout(backend_row)

        # Model path
        model_row = QHBoxLayout()
        self._model_path_lbl = QLabel("No model loaded")
        self._model_path_lbl.setWordWrap(True)
        btn_load = QPushButton("Load Model…")
        btn_load.clicked.connect(self._load_model)
        model_row.addWidget(self._model_path_lbl, stretch=1)
        model_row.addWidget(btn_load)
        inner.addLayout(model_row)

        # Preprocessing
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalization:"))
        self._norm_zscore = QRadioButton("Z-score")
        self._norm_minmax = QRadioButton("Min-max")
        self._norm_zscore.setChecked(True)
        norm_row.addWidget(self._norm_zscore)
        norm_row.addWidget(self._norm_minmax)
        norm_row.addStretch()
        inner.addLayout(norm_row)

        # Num classes
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Num classes:"))
        self._num_classes = QSpinBox()
        self._num_classes.setRange(2, 256)
        self._num_classes.setValue(2)
        cls_row.addWidget(self._num_classes)
        cls_row.addStretch()
        inner.addLayout(cls_row)

        # Run button + progress
        self._btn_run = QPushButton("▶ Run Inference")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)

        inner.addWidget(self._btn_run)
        inner.addWidget(self._progress)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addStretch()

    def set_volume(self, volume: np.ndarray):
        self._volume = volume
        self._update_run_btn()

    def _on_backend_change(self, backend: str):
        self._predictor = None
        self._model_path_lbl.setText("No model loaded")
        self._update_run_btn()

    def _load_model(self):
        backend = self._backend.currentText()
        if backend == "ONNX":
            filt = "ONNX model (*.onnx)"
        else:
            filt = "PyTorch model (*.pt *.pth)"

        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", filt)
        if not path:
            return

        try:
            if backend == "ONNX":
                from ..inference.onnx_predictor import OnnxPredictor
                self._predictor = OnnxPredictor()
            else:
                from ..inference.torch_predictor import TorchPredictor
                self._predictor = TorchPredictor()

            self._predictor.load(path)
            self._model_path_lbl.setText(path)
        except Exception as e:
            self._model_path_lbl.setText(f"Error: {e}")
            self._predictor = None

        self._update_run_btn()

    def _update_run_btn(self):
        self._btn_run.setEnabled(self._predictor is not None and self._volume is not None)

    def _run(self):
        if self._volume is None or self._predictor is None:
            return

        volume = self._volume.copy()
        if self._norm_zscore.isChecked():
            volume = normalize_zscore(volume)
        else:
            volume = normalize_minmax(volume)

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)

        self._worker = _InferenceWorker(self._predictor, volume)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, mask: np.ndarray):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self.prediction_ready.emit(mask)

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._model_path_lbl.setText(f"Inference error: {msg}")

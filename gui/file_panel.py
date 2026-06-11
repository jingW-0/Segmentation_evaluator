import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QButtonGroup, QRadioButton, QGroupBox
)
from PyQt6.QtCore import pyqtSignal

from ..io.nifti_reader import load_nifti
from ..io.dicom_reader import load_dicom_series


def _load_volume(parent) -> np.ndarray | None:
    """Show a file/folder dialog and return a volume, or None on cancel/error."""
    choice, _ = QFileDialog.getOpenFileName(
        parent, "Open NIfTI file", "",
        "NIfTI files (*.nii *.nii.gz);;All files (*)"
    )
    if choice:
        return load_nifti(choice)

    # If user cancelled, offer DICOM folder
    folder = QFileDialog.getExistingDirectory(parent, "Or select DICOM folder")
    if folder:
        return load_dicom_series(folder)

    return None


class FilePanel(QWidget):
    image_loaded = pyqtSignal(object)    # np.ndarray
    gt_loaded    = pyqtSignal(object)    # np.ndarray
    pred_loaded  = pyqtSignal(object)    # np.ndarray
    # signals for switching prediction mode
    mode_changed = pyqtSignal(str)       # "file" or "model"

    def __init__(self):
        super().__init__()

        # Image
        btn_img = QPushButton("Load Image")
        self._img_lbl = QLabel("—")
        btn_img.clicked.connect(self._load_image)

        # GT
        btn_gt = QPushButton("Load Ground Truth")
        self._gt_lbl = QLabel("—")
        btn_gt.clicked.connect(self._load_gt)

        # Prediction mode toggle
        mode_group = QGroupBox("Prediction source")
        self._rb_file  = QRadioButton("Load mask file")
        self._rb_model = QRadioButton("Run model inference")
        self._rb_file.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._rb_file)
        bg.addButton(self._rb_model)
        self._rb_file.toggled.connect(self._on_mode_toggle)
        mode_inner = QHBoxLayout(mode_group)
        mode_inner.addWidget(self._rb_file)
        mode_inner.addWidget(self._rb_model)

        # Pred file loader (visible in file mode)
        self._btn_pred = QPushButton("Load Prediction Mask")
        self._pred_lbl = QLabel("—")
        self._btn_pred.clicked.connect(self._load_pred)

        # Evaluate button
        self._btn_eval = QPushButton("▶  Evaluate")
        self._btn_eval.setObjectName("evalBtn")

        row1 = QHBoxLayout()
        row1.addWidget(btn_img)
        row1.addWidget(self._img_lbl, stretch=1)

        row2 = QHBoxLayout()
        row2.addWidget(btn_gt)
        row2.addWidget(self._gt_lbl, stretch=1)

        row3 = QHBoxLayout()
        row3.addWidget(self._btn_pred)
        row3.addWidget(self._pred_lbl, stretch=1)

        layout = QVBoxLayout(self)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(mode_group)
        layout.addLayout(row3)
        layout.addWidget(self._btn_eval)

    def _load_image(self):
        vol = _load_volume(self)
        if vol is not None:
            self._img_lbl.setText(f"shape: {vol.shape}")
            self.image_loaded.emit(vol)

    def _load_gt(self):
        vol = _load_volume(self)
        if vol is not None:
            self._gt_lbl.setText(f"shape: {vol.shape}")
            self.gt_loaded.emit(vol)

    def _load_pred(self):
        vol = _load_volume(self)
        if vol is not None:
            self._pred_lbl.setText(f"shape: {vol.shape}")
            self.pred_loaded.emit(vol)

    def _on_mode_toggle(self, file_mode: bool):
        self._btn_pred.setVisible(file_mode)
        self._pred_lbl.setVisible(file_mode)
        self.mode_changed.emit("file" if file_mode else "model")

    @property
    def evaluate_btn(self):
        return self._btn_eval

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox, QTabWidget
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


ORIENTATIONS = ["Axial", "Coronal", "Sagittal"]


class SliceCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.axis("off")

    def update_slice(self, image_slice, gt_slice=None, pred_slice=None, show_gt=True, show_pred=True):
        self.ax.clear()
        self.ax.axis("off")

        # Normalize image to [0,1] for display
        img = image_slice.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)

        self.ax.imshow(img, cmap="gray", origin="upper")

        if show_gt and gt_slice is not None:
            mask = np.ma.masked_where(gt_slice == 0, gt_slice)
            self.ax.imshow(mask, cmap="Greens", alpha=0.4, origin="upper",
                           vmin=0, vmax=max(gt_slice.max(), 1))

        if show_pred and pred_slice is not None:
            mask = np.ma.masked_where(pred_slice == 0, pred_slice)
            self.ax.imshow(mask, cmap="Reds", alpha=0.4, origin="upper",
                           vmin=0, vmax=max(pred_slice.max(), 1))

        self.draw()


class ViewerPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._image = None   # (H, W, D)
        self._gt = None
        self._pred = None

        self._tabs = QTabWidget()
        self._canvases = {}
        self._sliders = {}
        self._labels = {}

        for orient in ORIENTATIONS:
            tab = QWidget()
            canvas = SliceCanvas()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setValue(0)
            lbl = QLabel("0 / 0")

            slider.valueChanged.connect(lambda v, o=orient: self._on_slide(o, v))

            row = QHBoxLayout()
            row.addWidget(slider)
            row.addWidget(lbl)

            layout = QVBoxLayout(tab)
            layout.addWidget(canvas)
            layout.addLayout(row)

            self._tabs.addTab(tab, orient)
            self._canvases[orient] = canvas
            self._sliders[orient] = slider
            self._labels[orient] = lbl

        self._cb_gt = QCheckBox("Show GT overlay")
        self._cb_gt.setChecked(True)
        self._cb_gt.stateChanged.connect(self._refresh_current)

        self._cb_pred = QCheckBox("Show Pred overlay")
        self._cb_pred.setChecked(True)
        self._cb_pred.stateChanged.connect(self._refresh_current)

        cb_row = QHBoxLayout()
        cb_row.addWidget(self._cb_gt)
        cb_row.addWidget(self._cb_pred)
        cb_row.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addLayout(cb_row)

        self._tabs.currentChanged.connect(self._refresh_current)

    def set_image(self, volume: np.ndarray):
        self._image = volume
        self._reset_sliders()
        self._refresh_current()

    def set_gt(self, volume: np.ndarray):
        self._gt = volume
        self._refresh_current()

    def set_pred(self, volume: np.ndarray):
        self._pred = volume
        self._refresh_current()

    def _reset_sliders(self):
        if self._image is None:
            return
        H, W, D = self._image.shape
        depths = {"Axial": D, "Coronal": W, "Sagittal": H}
        for orient, depth in depths.items():
            s = self._sliders[orient]
            s.setMaximum(depth - 1)
            s.setValue(depth // 2)
            self._labels[orient].setText(f"{depth // 2} / {depth - 1}")

    def _current_orient(self) -> str:
        return ORIENTATIONS[self._tabs.currentIndex()]

    def _on_slide(self, orient: str, value: int):
        if self._image is None:
            return
        depth = self._sliders[orient].maximum() + 1
        self._labels[orient].setText(f"{value} / {depth - 1}")
        if self._current_orient() == orient:
            self._draw(orient, value)

    def _refresh_current(self, *_):
        orient = self._current_orient()
        self._draw(orient, self._sliders[orient].value())

    def _draw(self, orient: str, idx: int):
        if self._image is None:
            return

        img_s = self._get_slice(self._image, orient, idx)
        gt_s  = self._get_slice(self._gt,    orient, idx) if self._gt   is not None else None
        pr_s  = self._get_slice(self._pred,  orient, idx) if self._pred is not None else None

        self._canvases[orient].update_slice(
            img_s, gt_s, pr_s,
            show_gt=self._cb_gt.isChecked(),
            show_pred=self._cb_pred.isChecked(),
        )

    @staticmethod
    def _get_slice(volume: np.ndarray, orient: str, idx: int) -> np.ndarray:
        # After RAS reorientation: X=R, Y=A, Z=S
        # Transpose so cols=X (L-R) and rows=Y (A-P), then flip rows so Anterior is at top
        if orient == "Axial":
            return np.flipud(volume[:, :, idx].T)
        elif orient == "Coronal":
            return np.flipud(volume[:, idx, :].T)
        else:  # Sagittal
            return np.flipud(volume[idx, :, :].T)

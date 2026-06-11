from typing import Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QSizePolicy, QTabWidget
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


METRICS = ["dice", "iou", "precision", "recall"]
HEADERS = ["Class", "Dice", "IoU", "Precision", "Recall"]


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()

        self._table = QTableWidget(0, len(HEADERS))
        self._table.setHorizontalHeaderLabels(HEADERS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)

        tabs = QTabWidget()
        tabs.addTab(self._table, "Metrics Table")
        tabs.addTab(self._canvas, "Chart")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)

    def update_results(self, results: Dict[int, Dict[str, float]], label_names: Dict[int, str] = None):
        label_names = label_names or {}
        classes = sorted(results.keys())

        self._table.setRowCount(len(classes))
        for row, cls in enumerate(classes):
            name = label_names.get(cls, f"Class {cls}")
            vals = results[cls]
            self._table.setItem(row, 0, QTableWidgetItem(name))
            for col, m in enumerate(METRICS, start=1):
                item = QTableWidgetItem(f"{vals[m]:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)

        self._draw_chart(classes, results, label_names)

    def _draw_chart(self, classes, results, label_names):
        self._ax.clear()

        n = len(classes)
        x = np.arange(n)
        width = 0.2

        for i, m in enumerate(METRICS):
            vals = [results[c][m] for c in classes]
            self._ax.bar(x + i * width, vals, width, label=m.capitalize())

        self._ax.set_xticks(x + width * 1.5)
        self._ax.set_xticklabels([label_names.get(c, f"Class {c}") for c in classes], rotation=15)
        self._ax.set_ylim(0, 1.05)
        self._ax.set_ylabel("Score")
        self._ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
        self._fig.tight_layout()
        self._canvas.draw()

"""The Analysis menu: each analysis asks for its parameters and returns a new cube.

To add an analysis, write the numerical part in :mod:`hsview.analysis`, then
add an :class:`AnalysisCommand` to ``ANALYSES`` whose ``run`` takes the main
window, asks the user for parameters and returns the processed cube (or
``None`` if cancelled). The command automatically gets a menu item, a
toolbar-placeable action and a configurable shortcut; the window keeps the
previous cube so the result can be undone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ..analysis import SVDBackground, fit_svd_background, remove_svd_background
from ..io import Cube


@dataclass(frozen=True)
class AnalysisCommand:
    id: str
    text: str
    icon: str
    tip: str
    run: Callable[["object"], Cube | None]  # MainWindow -> new cube or None


class SVDBackgroundDialog(QDialog):
    """Choose how many of the largest SVD components are background."""

    def __init__(self, model: SVDBackground, parent=None, k: int = 1, add_mean: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Remove background (SVD)")
        self.model = model
        self.fractions = model.variance_fraction
        n = len(self.fractions)

        lay = QVBoxLayout(self)
        intro = QLabel("The mean spectrum is subtracted and the SVD of the cube (pixels × bands) is computed. "
                       "The largest components are treated as background and projected out of every spectrum.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.figure = Figure(figsize=(4.5, 2.4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        lay.addWidget(self.canvas, 1)

        form = QFormLayout()
        self.k = QSpinBox()
        self.k.setRange(0, max(0, n - 1))
        self.k.setValue(min(k, self.k.maximum()))
        self.k.setToolTip("Number of the largest components removed as background")
        form.addRow("Background components (k)", self.k)
        self.summary = QLabel()
        form.addRow("", self.summary)
        self.add_mean = QCheckBox("Add the mean spectrum back")
        self.add_mean.setToolTip("Off: the result is the residual around zero. "
                                 "On: the mean spectrum is added back so values stay in the original range.")
        self.add_mean.setChecked(add_mean)
        form.addRow("", self.add_mean)
        lay.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.k.valueChanged.connect(self._update)
        self._update()

    def _update(self):
        k = self.k.value()
        removed = float(self.fractions[:k].sum()) * 100
        self.summary.setText(f"Removes {removed:.2f} % of the variance around the mean")
        ax = self.ax
        ax.clear()
        shown = min(len(self.fractions), max(20, k + 5))
        idx = np.arange(1, shown + 1)
        frac = np.maximum(self.fractions[:shown] * 100, 1e-6)
        colors = ["#c0504d" if i <= k else "#4f81bd" for i in idx]
        ax.bar(idx, frac, color=colors)
        ax.set_yscale("log")
        ax.set_xlabel("Component")
        ax.set_ylabel("% of variance")
        ax.set_title("Red: removed as background", fontsize=9)
        self.canvas.draw_idle()


def run_svd_background(window) -> Cube | None:
    cube = window.cube
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        model = fit_svd_background(cube)
    finally:
        QApplication.restoreOverrideCursor()
    s = window.settings
    dlg = SVDBackgroundDialog(model, window, k=s.value("analysis/svd_k", 1, type=int),
                              add_mean=s.value("analysis/svd_add_mean", False, type=bool))
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    k, add_mean = dlg.k.value(), dlg.add_mean.isChecked()
    s.setValue("analysis/svd_k", k)
    s.setValue("analysis/svd_add_mean", add_mean)
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        return remove_svd_background(cube, k, add_mean, model)
    finally:
        QApplication.restoreOverrideCursor()


ANALYSES = [
    AnalysisCommand("analysis_svd_background", "Remove background (SVD)…", "background",
                    "Subtract the mean and project out the largest SVD components", run_svd_background),
]

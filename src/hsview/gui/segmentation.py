"""The Segment menu: parameter dialogs for unsupervised and supervised segmentation.

The numerical code is in :mod:`hsview.analysis.segmentation`. A segmentation
does not change the cube; it is shown as a label overlay (supervised results
use the annotation class colours) and can be saved with *Save segmentation…*.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ..analysis.segmentation import (
    INTENSITY_MODES,
    SUPERVISED_METHODS,
    Segmentation,
    segment_supervised,
    segment_unsupervised,
    training_label_image,
)
from ..annotations import CLASS_NAMES

SMOOTHING = [(0, "Off"), (3, "3 × 3"), (5, "5 × 5"), (7, "7 × 7"), (9, "9 × 9")]


class _SegmentDialog(QDialog):
    """Feature options shared by both dialogs, remembered in the settings."""

    prefix = "segment"

    def __init__(self, n_bands: int, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.lay = QVBoxLayout(self)
        self.form = QFormLayout()
        self.components = QSpinBox()
        self.components.setRange(0, n_bands)
        self.components.setSpecialValueText("All bands")
        self.components.setValue(min(self._get("components", 10, int), n_bands))
        self.components.setToolTip("Cluster or classify on this many principal components of the cube "
                                   "(faster and less noisy); 'All bands' uses the spectra directly")
        self.intensity = QComboBox()
        for key, text in INTENSITY_MODES.items():
            self.intensity.addItem(text, key)
        self.intensity.setCurrentIndex(max(0, self.intensity.findData(self._get("intensity", "use", str))))
        self.intensity.setToolTip(
            "Use intensity: brightness counts, so the same material in light and shadow can split.\n"
            "Ignore intensity: every spectrum is scaled to unit length, so only its shape decides the class\n"
            "(spectral angle); a pixel gets the same class at any brightness.\n"
            "Ignore intensity and offset: also subtracts each spectrum's own mean first, so a constant\n"
            "offset such as stray light or a dark level is ignored too (correlation between spectra).")
        self.smooth = QComboBox()
        for size, text in SMOOTHING:
            self.smooth.addItem(text, size)
        self.smooth.setCurrentIndex(max(0, self.smooth.findData(self._get("smooth", 0, int))))
        self.smooth.setToolTip("Majority filter applied to the result to remove isolated pixels")

    def _get(self, key, default, typ):
        if self.settings is None:
            return default
        return self.settings.value(f"{self.prefix}/{key}", default, type=typ)

    def _set(self, key, value):
        if self.settings is not None:
            self.settings.setValue(f"{self.prefix}/{key}", value)

    def _add_common_rows(self):
        self.form.addRow("Principal components", self.components)
        self.form.addRow("Intensity", self.intensity)
        self.form.addRow("Smoothing", self.smooth)

    def _finish_layout(self):
        self.lay.addLayout(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.lay.addWidget(buttons)

    def common_params(self) -> dict:
        p = {"n_components": self.components.value(), "intensity": self.intensity.currentData(),
             "smooth": int(self.smooth.currentData())}
        self._set("components", p["n_components"])
        self._set("intensity", p["intensity"])
        self._set("smooth", p["smooth"])
        return p


class UnsupervisedDialog(_SegmentDialog):
    prefix = "segment_unsupervised"

    def __init__(self, n_bands: int, settings=None, parent=None):
        super().__init__(n_bands, settings, parent)
        self.setWindowTitle("Segment (unsupervised)")
        intro = QLabel("Group the pixels into regions of similar spectra with k-means clustering. "
                       "No annotations are needed.")
        intro.setWordWrap(True)
        self.lay.addWidget(intro)
        self.clusters = QSpinBox()
        self.clusters.setRange(2, 64)
        self.clusters.setValue(self._get("clusters", 6, int))
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.iterations.setValue(self._get("iterations", 50, int))
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.seed.setValue(self._get("seed", 0, int))
        self.seed.setToolTip("Random seed for the starting cluster centres; change it to try another start")
        self.form.addRow("Number of regions", self.clusters)
        self._add_common_rows()
        self.form.addRow("Maximum iterations", self.iterations)
        self.form.addRow("Random seed", self.seed)
        self._finish_layout()

    def params(self) -> dict:
        p = self.common_params()
        p.update(n_clusters=self.clusters.value(), max_iter=self.iterations.value(), seed=self.seed.value())
        self._set("clusters", p["n_clusters"])
        self._set("iterations", p["max_iter"])
        self._set("seed", p["seed"])
        return p


class SupervisedDialog(_SegmentDialog):
    prefix = "segment_supervised"

    def __init__(self, n_bands: int, training: np.ndarray, n_annotations: dict, settings=None, parent=None):
        super().__init__(n_bands, settings, parent)
        self.setWindowTitle("Segment from annotations")
        intro = QLabel("Classify every pixel into one of the annotated classes. Points give one training pixel, "
                       "rectangles and polygons every pixel inside them.")
        intro.setWordWrap(True)
        self.lay.addWidget(intro)
        counts = np.bincount(training.ravel(), minlength=len(CLASS_NAMES) + 1)
        rows = [f"{CLASS_NAMES[c]}: {n_annotations.get(c, 0)} annotation(s), {counts[c + 1]} pixel(s)"
                for c in range(len(CLASS_NAMES)) if counts[c + 1] or n_annotations.get(c)]
        self.n_classes = int((counts[1:] > 0).sum())
        summary = QLabel("Training data\n  " + ("\n  ".join(rows) if rows else "none"))
        self.lay.addWidget(summary)
        self.method = QComboBox()
        for key, text in SUPERVISED_METHODS.items():
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(0, self.method.findData(self._get("method", "nearest_mean", str))))
        self.method.setToolTip("Nearest class mean: each pixel goes to the class with the closest mean spectrum.\n"
                               "LDA: linear discriminant with a shared covariance; better when classes overlap.\n"
                               "k-NN: majority vote of the closest training pixels; needs several pixels per class.")
        self.neighbors = QSpinBox()
        self.neighbors.setRange(1, 50)
        self.neighbors.setValue(self._get("neighbors", 5, int))
        self.form.addRow("Method", self.method)
        self.form.addRow("Neighbours (k-NN)", self.neighbors)
        self._add_common_rows()
        self.warning = QLabel()
        self.warning.setStyleSheet("color: #c0504d")
        self.warning.setWordWrap(True)
        self._finish_layout()
        self.lay.insertWidget(self.lay.count() - 1, self.warning)
        self.method.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self):
        self.neighbors.setEnabled(self.method.currentData() == "knn")
        ok = self.n_classes >= 2
        self.warning.setText("" if ok else "Annotate at least two classes (pick the class with keys 1–7) "
                                           "inside the image to train on.")
        self.warning.setVisible(not ok)
        self.ok_button.setEnabled(ok)

    def params(self) -> dict:
        p = self.common_params()
        p.update(method=self.method.currentData(), k_neighbors=self.neighbors.value())
        self._set("method", p["method"])
        self._set("neighbors", p["k_neighbors"])
        return p


def _busy(fn, *args, **kwargs):
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        return fn(*args, **kwargs)
    finally:
        QApplication.restoreOverrideCursor()


def run_unsupervised(window) -> Segmentation | None:
    dlg = UnsupervisedDialog(window.cube.n_bands, window.settings, window)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return _busy(segment_unsupervised, window.cube, **dlg.params())


def run_supervised(window) -> Segmentation | None:
    annotations = list(window.store)
    training = training_label_image(annotations, window.cube.shape[:2])
    n_ann = {}
    for a in annotations:
        n_ann[a.cls] = n_ann.get(a.cls, 0) + 1
    dlg = SupervisedDialog(window.cube.n_bands, training, n_ann, window.settings, window)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return _busy(segment_supervised, window.cube, annotations, **dlg.params())

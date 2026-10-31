"""Processed images (segmentations, score maps) drawn over the cube display."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..io import is_label_image, load_overlay
from .image_view import ImageView

OVERLAY_FILTER = "Images and arrays (*.npy *.tif *.tiff *.png *.jpg *.bmp *.h5 *.hdf5 *.pkl);;All files (*)"


def label_colors(n: int) -> np.ndarray:
    """Distinct RGB colours for labels 1..n (index 0 is unused/transparent)."""
    from matplotlib import colormaps

    base = colormaps["tab20"].colors
    cols = np.array([base[(i * 2) % 20 if i < 10 else (i * 2 + 1) % 20] for i in range(max(n, 1))])
    return np.vstack([[0, 0, 0], cols[:n]])


def colorize(arr: np.ndarray, as_labels: bool) -> np.ndarray:
    """RGBA uint8 rendering of an overlay array."""
    if arr.ndim == 3:  # an RGB(A) image is shown as-is
        rgb = arr[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(255 * (rgb - rgb.min()) / max(np.ptp(rgb), 1e-12), 0, 255).astype(np.uint8)
        return np.dstack([rgb, np.full(arr.shape[:2], 255, np.uint8)])
    if as_labels:
        labels = arr.astype(np.int64)
        values = np.unique(labels)
        values = values[values != 0]
        lut_rgb = label_colors(len(values))
        index = np.searchsorted(values, labels) + 1
        index[labels == 0] = 0
        rgba = np.zeros(arr.shape + (4,), np.uint8)
        rgba[..., :3] = (lut_rgb[index] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(labels == 0, 0, 255)
        return rgba
    from matplotlib import colormaps

    finite = arr[np.isfinite(arr)]
    lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0, 1))
    norm = np.clip((arr - lo) / (hi - lo if hi > lo else 1), 0, 1)
    rgba = (colormaps["magma"](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[~np.isfinite(arr), 3] = 0
    return rgba


@dataclass
class Overlay:
    key: str
    name: str
    data: np.ndarray
    as_labels: bool
    opacity: float = 0.5
    visible: bool = True

    def label_color(self, value: int):
        values = np.unique(self.data)
        values = values[values != 0]
        cols = label_colors(len(values))
        idx = int(np.searchsorted(values, value)) + 1
        return tuple(cols[idx]) if idx < len(cols) else (0.5, 0.5, 0.5)


class OverlayPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, view: ImageView, settings=None, parent=None):
        super().__init__(parent)
        self.view = view
        self.settings = settings
        self.shape: tuple[int, int] | None = None
        self.overlays: dict[str, Overlay] = {}
        self._counter = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentItemChanged.connect(self._on_current)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.valueChanged.connect(self._on_opacity)
        row.addWidget(self.opacity, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.add_btn = QPushButton("Add…")
        self.add_btn.clicked.connect(self.add_dialog)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_current)
        row.addWidget(self.add_btn)
        row.addWidget(self.remove_btn)
        lay.addLayout(row)
        self._sync_buttons()

    def set_shape(self, shape):
        self.shape = shape
        self.clear()

    def clear(self):
        self.view.clear_overlays()
        self.overlays.clear()
        self.list.clear()
        self._sync_buttons()
        self.changed.emit()

    def add_dialog(self):
        last = self.settings.value("paths/last_dir", "") if self.settings else ""
        path, _ = QFileDialog.getOpenFileName(self, "Add overlay", last, OVERLAY_FILTER)
        if path:
            self.add_file(path)

    def add_file(self, path) -> Overlay | None:
        try:
            arr = load_overlay(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open overlay", str(exc))
            return None
        if self.settings:
            self.settings.setValue("paths/last_dir", str(Path(path).parent))
        return self.add_array(arr, Path(path).stem)

    def add_array(self, arr: np.ndarray, name: str, as_labels: bool | None = None) -> Overlay | None:
        arr = np.asarray(arr)
        if self.shape is not None and arr.shape[:2] != tuple(self.shape):
            QMessageBox.warning(self, "Size mismatch", f"Overlay '{name}' is {arr.shape[1]}×{arr.shape[0]} but the "
                                                       f"image is {self.shape[1]}×{self.shape[0]}.")
            return None
        if as_labels is None:
            as_labels = is_label_image(arr)
        self._counter += 1
        ov = Overlay(f"ov{self._counter}", name, arr, as_labels)
        self.overlays[ov.key] = ov
        self.view.set_overlay(ov.key, colorize(arr, as_labels), ov.opacity)
        item = QListWidgetItem(f"{name}  ({'labels' if as_labels else 'image'})")
        item.setData(Qt.ItemDataRole.UserRole, ov.key)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self._sync_buttons()
        self.changed.emit()
        return ov

    def current(self) -> Overlay | None:
        item = self.list.currentItem()
        return self.overlays.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def label_overlay(self) -> Overlay | None:
        """The selected overlay if it is a visible segmentation, else the last visible one."""
        cur = self.current()
        if cur is not None and cur.as_labels and cur.visible:
            return cur
        for ov in reversed(self.overlays.values()):
            if ov.as_labels and ov.visible:
                return ov
        return None

    def remove_current(self):
        item = self.list.currentItem()
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        self.view.remove_overlay(key)
        self.overlays.pop(key, None)
        self.list.takeItem(self.list.row(item))
        self._sync_buttons()
        self.changed.emit()

    def _on_item_changed(self, item):
        ov = self.overlays.get(item.data(Qt.ItemDataRole.UserRole))
        if ov is not None:
            ov.visible = item.checkState() == Qt.CheckState.Checked
            self.view.set_overlay_visible(ov.key, ov.visible)
            self.changed.emit()

    def _on_current(self, *_):
        ov = self.current()
        if ov is not None:
            self.opacity.blockSignals(True)
            self.opacity.setValue(int(ov.opacity * 100))
            self.opacity.blockSignals(False)
        self._sync_buttons()

    def _on_opacity(self, v):
        ov = self.current()
        if ov is not None:
            ov.opacity = v / 100
            self.view.set_overlay_opacity(ov.key, ov.opacity)

    def _sync_buttons(self):
        has = self.current() is not None
        self.remove_btn.setEnabled(has)
        self.opacity.setEnabled(has)

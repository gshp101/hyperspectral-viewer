"""Controls that decide what the canvas shows.

Three band channels and a reference image can each be routed to the red,
green or blue display channel, or shown alone as a gray (or, for the
reference, colour) image; the reference can also be blended over the band
display. A channel's source is a single band (optionally averaged over a
window of neighbouring bands), a sum/mean of bands, or a spectral filter. Contrast and bias work like the original viewer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..io import Cube, LoadError, load_filter, read_image
from ..processing import SCALINGS, apply_filter, band_image, compose, contrast_bias_lut, parse_band_spec, spectral_energy

ROUTES = ("None", "Red", "Green", "Blue", "Gray")
REF_ROUTES = ("None", "Show", "Overlay")
SOURCES = ("Band", "Band sum", "Filter")
SCALING_LABELS = {"global": "Shared min–max", "channel": "Per-channel min–max", "percentile": "Per-channel 2–98 %"}
DEFAULT_CONTRAST, DEFAULT_BIAS = 45, 0


def band_window(center: int, width: int, n_bands: int) -> range:
    """The ``width`` bands centred on ``center``, shifted to stay inside 0..n_bands-1."""
    width = max(1, min(int(width), n_bands))
    start = min(max(0, int(center) - (width - 1) // 2), n_bands - width)
    return range(start, start + width)


def _hslider(lo, hi, value):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(value)
    return s


class BandChannel(QGroupBox):
    """One hyperspectral channel: where its image comes from and which display channel it feeds."""

    changed = pyqtSignal()
    bandChanged = pyqtSignal(int)  # the band slider moved (for the wavelength readout)

    def __init__(self, title: str, panel: "DisplayPanel"):
        super().__init__(title)
        self.panel = panel
        self.filter: np.ndarray | None = None
        self.filter_name = ""
        self._cache = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        row = QHBoxLayout()
        self.source = QComboBox()
        self.source.addItems(SOURCES)
        self.route = QComboBox()
        self.route.addItems(ROUTES)
        self.route.setToolTip("Display channel this image feeds; Gray shows it alone")
        row.addWidget(self.source, 1)
        row.addWidget(QLabel("→"))
        row.addWidget(self.route, 1)
        lay.addLayout(row)

        self.stack = QStackedWidget()
        # Band
        w = QWidget()
        r = QHBoxLayout()
        r.setContentsMargins(0, 0, 0, 0)
        self.slider = _hslider(0, 0, 0)
        self.spin = QSpinBox()
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self._on_band)
        r.addWidget(self.slider, 1)
        r.addWidget(self.spin)
        g = QVBoxLayout()
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(2)
        g.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel("Width"))
        self.width_slider = _hslider(1, 1, 1)
        self.width_slider.setToolTip("Average this many bands centred on the selected band")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 1)
        self.width_spin.setToolTip(self.width_slider.toolTip())
        self.width_slider.valueChanged.connect(self.width_spin.setValue)
        self.width_spin.valueChanged.connect(self.width_slider.setValue)
        self.width_spin.valueChanged.connect(self._invalidate)
        r.addWidget(self.width_slider, 1)
        r.addWidget(self.width_spin)
        g.addLayout(r)
        w.setLayout(g)
        self.stack.addWidget(w)
        # Band sum
        w = QWidget()
        r = QHBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        self.spec = QLineEdit()
        self.spec.setPlaceholderText("10-20, 35  or  650-700nm")
        self.spec.editingFinished.connect(self._invalidate)
        self.sum_mode = QComboBox()
        self.sum_mode.addItems(["sum", "mean"])
        self.sum_mode.currentIndexChanged.connect(self._invalidate)
        r.addWidget(self.spec, 1)
        r.addWidget(self.sum_mode)
        self.stack.addWidget(w)
        # Filter
        w = QWidget()
        r = QHBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        self.filter_label = QLabel("No filter")
        self.filter_btn = QPushButton("Load…")
        self.filter_btn.clicked.connect(self.load_filter_dialog)
        r.addWidget(self.filter_label, 1)
        r.addWidget(self.filter_btn)
        self.stack.addWidget(w)
        lay.addWidget(self.stack)
        self.info = QLabel("")
        self.info.setStyleSheet("color: gray")
        lay.addWidget(self.info)

        self.source.currentIndexChanged.connect(self._on_source)
        self.route.currentIndexChanged.connect(lambda _: panel._route_changed(self))

    # ---- state
    def set_cube(self, cube: Cube | None):
        self._cache = None
        self.filter, self.filter_name = None, ""
        self.filter_label.setText("No filter")
        n = cube.n_bands if cube else 1
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setRange(0, n - 1)
            w.blockSignals(False)
        for w in (self.width_slider, self.width_spin):
            w.blockSignals(True)
            w.setRange(1, n)
            w.blockSignals(False)
        if self.source.currentText() == "Filter":
            self.source.setCurrentIndex(0)
        self._update_info()

    def set_band(self, b: int):
        self.source.setCurrentIndex(0)
        self.spin.setValue(int(b))

    def set_width(self, width: int):
        self.width_spin.setValue(int(width))

    def bands(self) -> range:
        """Bands averaged in Band mode."""
        n = self.panel.cube.n_bands if self.panel.cube else 1
        return band_window(self.spin.value(), self.width_spin.value(), n)

    def set_spec(self, spec: str, mode: str = "sum"):
        self.source.blockSignals(True)
        self.source.setCurrentIndex(1)
        self.stack.setCurrentIndex(1)
        self.source.blockSignals(False)
        self.spec.setText(spec)
        self.sum_mode.setCurrentText(mode)
        self._invalidate()

    def set_filter(self, filt: np.ndarray, name: str):
        self.filter, self.filter_name = filt, name
        self.filter_label.setText(name)
        self.source.setCurrentIndex(2)
        self._invalidate()

    def step_band(self, delta: int):
        self.source.setCurrentIndex(0)
        self.spin.setValue(max(0, min(self.spin.maximum(), self.spin.value() + delta)))

    # ---- events
    def _on_source(self, idx):
        self.stack.setCurrentIndex(idx)
        if SOURCES[idx] == "Filter" and self.filter is None:
            self.load_filter_dialog()
            if self.filter is None:
                self.source.setCurrentIndex(0)
                return
        self._invalidate()

    def _on_band(self, b):
        self.bandChanged.emit(b)
        self._invalidate()

    def _invalidate(self):
        self._cache = None
        self._update_info()
        self.changed.emit()

    def _update_info(self):
        cube = self.panel.cube
        if cube is None:
            self.info.setText("")
            return
        src = self.source.currentText()
        if src == "Band":
            text = cube.band_label(self.spin.value())
            bands = self.bands()
            if len(bands) > 1:
                text += f", mean of {len(bands)} bands {bands[0]}–{bands[-1]}"
            self.info.setText(text)
        elif src == "Band sum":
            try:
                n = len(parse_band_spec(self.spec.text(), cube)) if self.spec.text().strip() else 0
                self.info.setText(f"{self.sum_mode.currentText()} of {n} bands" if n else "Enter bands to combine")
            except ValueError as exc:
                self.info.setText(str(exc))
        else:
            self.info.setText("" if self.filter is None else
                              f"filter {self.filter.shape[1] if self.filter.ndim == 2 else 1} component(s)")

    def load_filter_dialog(self):
        cube = self.panel.cube
        if cube is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load filter", self.panel.last_dir(), "Filter files (*.pkl);;All files (*)")
        if not path:
            return
        try:
            filt = load_filter(path, cube.n_bands)
        except LoadError as exc:
            QMessageBox.warning(self, "Bad filter file", str(exc))
            return
        self.panel.remember_dir(path)
        self.set_filter(filt, Path(path).name)

    # ---- output
    def image(self) -> np.ndarray | None:
        cube = self.panel.cube
        if cube is None:
            return None
        if self._cache is None:
            src = self.source.currentText()
            if src == "Band":
                bands = self.bands()
                if len(bands) == 1:
                    self._cache = np.asarray(cube.data[:, :, bands[0]], dtype=np.float64)
                else:
                    self._cache = band_image(cube, bands, "mean")
            elif src == "Band sum":
                if not self.spec.text().strip():
                    return None
                try:
                    bands = parse_band_spec(self.spec.text(), cube)
                except ValueError:
                    return None
                self._cache = band_image(cube, bands, self.sum_mode.currentText())
            elif self.filter is not None:
                self._cache = apply_filter(cube, self.filter, self.panel.energy() if self.filter.ndim == 2 else None)
        return self._cache


class ReferenceChannel(QGroupBox):
    """A co-registered photo of the scene (the file's 'lbl'/'webcam' image, or one loaded from disk)."""

    changed = pyqtSignal()

    def __init__(self, panel: "DisplayPanel"):
        super().__init__("Reference image")
        self.panel = panel
        self.img: np.ndarray | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        row = QHBoxLayout()
        self.label = QLabel("None")
        self.route = QComboBox()
        self.route.addItems(REF_ROUTES)
        row.addWidget(self.label, 1)
        row.addWidget(self.route)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.load_btn = QPushButton("Load…")
        self.load_btn.clicked.connect(self.load_dialog)
        self.reverse_btn = QPushButton("Reverse RGB")
        self.reverse_btn.setToolTip("Swap red and blue (for BGR images)")
        self.reverse_btn.clicked.connect(self.reverse_rgb)
        row.addWidget(self.load_btn)
        row.addWidget(self.reverse_btn)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity"))
        self.opacity = _hslider(0, 100, 50)
        self.opacity.setToolTip("How strongly the reference image covers the bands when routed to Overlay")
        self.opacity.valueChanged.connect(lambda _: self.changed.emit())
        row.addWidget(self.opacity, 1)
        lay.addLayout(row)
        self.route.setToolTip("Show replaces the band display; Overlay blends the image over it")
        self.route.currentIndexChanged.connect(lambda _: panel._route_changed(self))
        self._sync()

    def _sync(self):
        has = self.img is not None
        self.route.setEnabled(has)
        self.reverse_btn.setEnabled(has and self.img.ndim == 3)
        self.opacity.setEnabled(has)
        if not has and self.route.currentIndex():
            self.route.setCurrentIndex(0)

    def set_image(self, img: np.ndarray | None, name: str = ""):
        self.img = None if img is None else np.asarray(img)
        self.label.setText(name if img is not None else "None")
        self._sync()
        self.changed.emit()

    def reverse_rgb(self):
        if self.img is not None and self.img.ndim == 3:
            self.img = np.ascontiguousarray(self.img[:, :, ::-1])
            self.changed.emit()

    def load_dialog(self):
        cube = self.panel.cube
        path, _ = QFileDialog.getOpenFileName(self, "Open reference image", self.panel.last_dir(),
                                              "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open image", str(exc))
            return
        if cube is not None and img.shape[:2] != cube.shape[:2]:
            QMessageBox.warning(self, "Size mismatch", f"The image is {img.shape[1]}×{img.shape[0]} but the cube is "
                                                       f"{cube.shape[1]}×{cube.shape[0]}.")
            return
        self.panel.remember_dir(path)
        self.set_image(img, Path(path).name)
        self.route.setCurrentIndex(1)

    def image(self):
        return self.img


class DisplayPanel(QWidget):
    """All display controls; emits ``imageChanged`` with the uint8 image to show."""

    imageChanged = pyqtSignal(object)
    bandReadout = pyqtSignal(str)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.cube: Cube | None = None
        self.settings = settings
        self._energy = None
        self._routing = False
        self._last_composite = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.channels = [BandChannel(f"Band channel {i + 1}", self) for i in range(3)]
        for ch in self.channels:
            ch.changed.connect(self.schedule)
            ch.bandChanged.connect(lambda b, ch=ch: self._readout(b))
            lay.addWidget(ch)
        self.reference = ReferenceChannel(self)
        self.reference.changed.connect(self.schedule)
        lay.addWidget(self.reference)

        box = QGroupBox("Contrast")
        form = QFormLayout(box)
        self.scaling = QComboBox()
        for key in SCALINGS:
            self.scaling.addItem(SCALING_LABELS[key], key)
        self.scaling.currentIndexChanged.connect(self.schedule)
        form.addRow("Scaling", self.scaling)
        self.contrast = _hslider(0, 89, DEFAULT_CONTRAST)
        self.bias = _hslider(-128, 127, DEFAULT_BIAS)
        self.contrast.valueChanged.connect(self._apply_lut)
        self.bias.valueChanged.connect(self._apply_lut)
        form.addRow("Contrast", self.contrast)
        form.addRow("Bias", self.bias)
        reset = QPushButton("Reset contrast")
        reset.clicked.connect(self.reset_contrast)
        form.addRow(reset)
        lay.addWidget(box)
        lay.addStretch(1)

        self._timer = QTimer(self, singleShot=True, interval=15)
        self._timer.timeout.connect(self.update_image)
        self.setEnabled(False)

    # ------------------------------------------------------------ helpers
    def last_dir(self) -> str:
        return self.settings.value("paths/last_dir", "") if self.settings else ""

    def remember_dir(self, path):
        if self.settings:
            self.settings.setValue("paths/last_dir", str(Path(path).parent))

    def energy(self):
        if self._energy is None and self.cube is not None:
            self._energy = spectral_energy(self.cube)
        return self._energy

    def _readout(self, b):
        if self.cube is not None:
            self.bandReadout.emit(self.cube.band_label(b))

    # ------------------------------------------------------------- cube
    def set_cube(self, cube: Cube | None):
        self.cube = cube
        self._energy = None
        self.setEnabled(cube is not None)
        self._routing = True
        for ch in self.channels:
            ch.set_cube(cube)
            ch.route.setCurrentIndex(0)
        name = "from file" if cube is not None and cube.reference is not None else ""
        self.reference.set_image(None if cube is None else cube.reference, name)
        self.reference.route.setCurrentIndex(0)
        self._routing = False
        if cube is None:
            self.imageChanged.emit(None)
            return
        if cube.reference is not None and cube.band_names in (("red", "green", "blue"), ("gray",)):
            self.reference.route.setCurrentIndex(1)  # plain photos show as themselves
        else:
            for i, ch in enumerate(self.channels):
                ch.spin.setValue(min(cube.n_bands - 1, (i + 1) * cube.n_bands // 4))
            self.channels[0].spin.setValue(cube.n_bands // 2)
            self.channels[0].route.setCurrentText("Gray")
        self._readout(self.channels[0].spin.value())
        self.update_image()

    # --------------------------------------------------------- routing
    def _route_changed(self, source):
        """Keep routing consistent: one source per colour, Gray/Show is exclusive, never nothing."""
        if self._routing:
            return
        self._routing = True
        try:
            # The reference's Overlay blends over whatever the bands show, so it never conflicts.
            sources = [*self.channels, self.reference]
            choice = source.route.currentText()
            exclusive = choice in ("Gray", "Show")
            if choice not in ("None", "Overlay"):
                for other in sources:
                    if other is source:
                        continue
                    o = other.route.currentText()
                    if o == "Overlay" and choice != "Show":
                        continue
                    if exclusive or o in ("Gray", "Show") or o == choice:
                        other.route.setCurrentIndex(0)
            if self.reference.route.currentText() != "Show" and all(c.route.currentIndex() == 0 for c in self.channels):
                self.channels[0].route.setCurrentText("Gray")
        finally:
            self._routing = False
        self.schedule()

    def show_band(self, b: int, channel: int = 0):
        self.channels[channel].set_band(b)
        self.channels[channel].route.setCurrentText("Gray")

    def show_rgb(self, specs, mode="sum"):
        """specs: three band selections for red, green, blue ('' leaves a channel empty)."""
        self._routing = True
        try:
            for ch in self.channels:
                ch.route.setCurrentIndex(0)
            self.reference.route.setCurrentIndex(0)
            for color, (ch, spec) in enumerate(zip(self.channels, specs), start=1):
                if spec:
                    ch.set_spec(spec, mode)
                    ch.route.setCurrentIndex(color)
        finally:
            self._routing = False
        self.schedule()

    def true_colour(self):
        if self.cube is None or self.cube.wavelengths is None:
            return
        self.show_rgb(["620-670nm", "530-570nm", "450-490nm"])

    def step_band(self, delta: int):
        """Move the band of the first channel showing a single band."""
        for ch in self.channels:
            if ch.route.currentIndex() and ch.source.currentText() == "Band":
                ch.step_band(delta)
                return
        self.channels[0].step_band(delta)
        if all(c.route.currentIndex() == 0 for c in self.channels) and self.reference.route.currentIndex() == 0:
            self.channels[0].route.setCurrentText("Gray")

    # --------------------------------------------------------- output
    def schedule(self):
        self._timer.start()

    def composite(self) -> np.ndarray | None:
        if self.cube is None:
            return None
        scaling = self.scaling.currentData()
        ref_route = self.reference.route.currentText() if self.reference.img is not None else "None"
        if ref_route == "Show":
            return compose(gray=self.reference.img, scaling=scaling)
        base = self._band_composite(scaling)
        if ref_route == "Overlay":
            ref = compose(gray=self.reference.img, scaling=scaling)
            if base is None:
                return ref
            alpha = self.reference.opacity.value() / 100
            if ref.ndim == 2:
                ref = ref[:, :, None]
            if base.ndim == 2:
                base = base[:, :, None]
            out = (1 - alpha) * base + alpha * ref
            out = np.rint(out).astype(np.uint8)
            return np.ascontiguousarray(out if out.shape[2] == 3 else out[:, :, 0])
        return base

    def _band_composite(self, scaling):
        rgb = {}
        for ch in self.channels:
            route = ch.route.currentText()
            if route == "None":
                continue
            img = ch.image()
            if img is None:
                continue
            if route == "Gray":
                return compose(gray=img, scaling=scaling)
            rgb[ROUTES.index(route) - 1] = img
        return compose(rgb=rgb, scaling=scaling)

    def update_image(self):
        self._last_composite = self.composite()
        self._apply_lut()

    def _apply_lut(self):
        img = self._last_composite
        if img is not None and (self.contrast.value() != DEFAULT_CONTRAST or self.bias.value() != DEFAULT_BIAS):
            img = contrast_bias_lut(self.contrast.value(), self.bias.value())[img]
        self.imageChanged.emit(img)

    def reset_contrast(self):
        self.contrast.setValue(DEFAULT_CONTRAST)
        self.bias.setValue(DEFAULT_BIAS)

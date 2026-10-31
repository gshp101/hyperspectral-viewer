"""Spectrum plots: the live panel and the per-class plot windows."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QVBoxLayout, QWidget

from ..annotations import CLASS_NAMES, CLASS_PLOT_COLORS, AnnotationStore, mean_spectra
from ..io import Cube
from ..processing import label_spectra, pixel_spectrum


class SpectrumPanel(QWidget):
    """Live plot of annotation spectra, the pixel under the cursor and segmentation class means."""

    def __init__(self, store: AnnotationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.cube: Cube | None = None
        self.cursor = None
        self.band_marker = None
        self.label_source = None  # callable returning the overlay to summarise
        self._label_cache = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.follow_cursor = QCheckBox("Cursor")
        self.follow_cursor.setToolTip("Show the spectrum of the pixel under the mouse")
        self.follow_cursor.setChecked(True)
        self.show_std = QCheckBox("±1σ")
        self.show_std.setToolTip("Shade one standard deviation around region means")
        self.show_std.setChecked(True)
        self.by_class = QCheckBox("By class")
        self.by_class.setToolTip("One mean spectrum per annotation class instead of one line per annotation")
        self.show_labels = QCheckBox("Overlay classes")
        self.show_labels.setToolTip("Mean spectrum of each class in the selected segmentation overlay")
        for cb in (self.follow_cursor, self.show_std, self.by_class, self.show_labels):
            cb.toggled.connect(self.schedule)
            row.addWidget(cb)
        row.addStretch(1)
        lay.addLayout(row)
        self.figure = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(180)
        self.ax = self.figure.add_subplot(111)
        lay.addWidget(self.canvas, 1)

        self._timer = QTimer(self, singleShot=True, interval=40)
        self._timer.timeout.connect(self.redraw)
        store.subscribe(self.schedule)

    def set_cube(self, cube: Cube | None):
        self.cube = cube
        self.cursor = None
        self._label_cache.clear()
        self.redraw()

    def set_cursor(self, x: float, y: float):
        pos = None if np.isnan(x) else (int(np.floor(x)), int(np.floor(y)))
        if pos != self.cursor:
            self.cursor = pos
            if self.follow_cursor.isChecked():
                self.schedule()

    def set_band_marker(self, b: int | None):
        self.band_marker = b
        self.schedule()

    def overlays_changed(self):
        self._label_cache.clear()
        self.schedule()

    def schedule(self):
        self._timer.start()

    def redraw(self):
        ax = self.ax
        ax.clear()
        cube = self.cube
        if cube is None:
            ax.text(0.5, 0.5, "Open a cube to see spectra", ha="center", va="center", transform=ax.transAxes,
                    color="0.5")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        x = cube.band_axis
        n = 0

        if self.show_labels.isChecked() and self.label_source is not None:
            ov = self.label_source()
            if ov is not None and ov.data.shape == cube.shape[:2]:
                if ov.key not in self._label_cache:
                    self._label_cache[ov.key] = label_spectra(cube, ov.data)
                for v, spec in self._label_cache[ov.key].items():
                    ax.plot(x, spec, "--", lw=1.2, color=ov.label_color(v), label=f"{ov.name} {v}")
                    n += 1

        stats = mean_spectra(cube, self.store)
        if self.by_class.isChecked():
            for cls in sorted({a.cls for a, *_ in stats}):
                rows = [(a, m, s, k) for a, m, s, k in stats if a.cls == cls]
                pix = sum(k for *_, k in rows)
                mean = sum(m * k for _, m, _, k in rows) / pix
                ax.plot(x, mean, color=CLASS_PLOT_COLORS[cls], lw=2,
                        label=f"{CLASS_NAMES[cls]} ({len(rows)} ann., n={pix})")
                n += 1
        else:
            for a, mean, std, k in stats:
                col = CLASS_PLOT_COLORS[a.cls % len(CLASS_PLOT_COLORS)]
                if a.kind == "point":
                    ax.plot(x, mean, color=col, lw=1.2, label=f"{a.label} ({int(a.coords[0][0])}, {int(a.coords[0][1])})")
                else:
                    ax.plot(x, mean, color=col, lw=2, label=f"{a.label} mean, n={k}")
                    if self.show_std.isChecked() and k > 1:
                        ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.15, lw=0)
                n += 1

        if self.follow_cursor.isChecked() and self.cursor is not None:
            s = pixel_spectrum(cube, *self.cursor)
            if s is not None:
                ax.plot(x, s, color="0.35", lw=1, ls=":", label=f"cursor ({self.cursor[0]}, {self.cursor[1]})")
                n += 1

        if self.band_marker is not None and 0 <= self.band_marker < cube.n_bands:
            ax.axvline(x[self.band_marker], color="0.6", lw=0.8)
        ax.set_xlabel(cube.band_axis_label)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        if n:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=False)
        self.canvas.draw_idle()


class ClassPlotDialog(QDialog):
    """One subplot per class with every spectrum of that class, like the original viewer's Plot menu."""

    def __init__(self, cube: Cube, groups: dict[int, list[np.ndarray]], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 300 + 160 * max(0, (len(groups) - 1) // 2))
        fig = Figure(layout="constrained")
        canvas = FigureCanvasQTAgg(fig)
        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(canvas, self))
        lay.addWidget(canvas, 1)
        classes = sorted(groups)
        cols = 1 if len(classes) == 1 else 2
        rows = int(np.ceil(len(classes) / cols))
        x = cube.band_axis
        axes = fig.subplots(rows, cols, squeeze=False, sharex=True)
        for ax, cls in zip(axes.ravel(), classes):
            for spec in groups[cls]:
                ax.plot(x, spec, color=CLASS_PLOT_COLORS[cls], lw=1, alpha=0.8)
            ax.set_title(f"{CLASS_NAMES[cls]} ({len(groups[cls])})", fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(classes):]:
            ax.set_axis_off()
        for ax in axes[-1]:
            ax.set_xlabel(cube.band_axis_label)
        self.figure = fig

"""The napari dock widget: band display, RGB composites, annotations, spectra, overlays."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import annotations as ann_io
from .io import Cube, is_label_image, load_cube, load_overlay
from .processing import (
    band_image,
    label_spectra,
    parse_band_spec,
    pixel_spectrum,
    region_spectrum,
    rgb_composite,
)

# Tableau-like palette; annotation N always gets colour N so plot and canvas match.
PALETTE = np.array(
    [
        [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
        [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.74, 0.74, 0.13],
        [0.09, 0.75, 0.81], [1.00, 0.90, 0.20],
    ]
)
POINTS_LAYER = "points"
SHAPES_LAYER = "regions"
BAND_LAYER = "band"
RGB_LAYER = "RGB"

CUBE_FILTER = "Hyperspectral cubes (*.hdr *.img *.dat *.raw *.bil *.bip *.bsq *.npy *.h5 *.hdf5 *.he5 *.tif *.tiff *.pkl *.pickle *.p);;All files (*)"
OVERLAY_FILTER = "Images (*.npy *.tif *.tiff *.png *.jpg *.bmp *.h5 *.hdf5);;All files (*)"


def _color(i: int) -> np.ndarray:
    return np.append(PALETTE[i % len(PALETTE)], 1.0)


class HyperspectralWidget(QWidget):
    def __init__(self, viewer, cube: Cube | None = None):
        super().__init__()
        self.viewer = viewer
        self.cube: Cube | None = None
        self._cursor_pos = None
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_file_box())
        layout.addWidget(self._build_band_box())
        layout.addWidget(self._build_rgb_box())
        layout.addWidget(self._build_annotation_box())
        layout.addWidget(self._build_spectrum_box(), stretch=1)

        # Replotting is debounced so dragging a vertex or the band slider stays smooth.
        self._plot_timer = QTimer(self, singleShot=True, interval=60)
        self._plot_timer.timeout.connect(self.update_spectrum_plot)
        self._band_timer = QTimer(self, singleShot=True, interval=30)
        self._band_timer.timeout.connect(self._show_single_band)

        viewer.cursor.events.position.connect(self._on_cursor)
        self._set_enabled(False)
        if cube is not None:
            self.set_cube(cube)

    # ------------------------------------------------------------------ UI
    def _build_file_box(self):
        box = QGroupBox("Data")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        btn = QPushButton("Open cube…")
        btn.clicked.connect(self._open_cube_dialog)
        row.addWidget(btn)
        self.overlay_btn = QPushButton("Add overlay…")
        self.overlay_btn.setToolTip("Load a processed image (segmentation, classification, score map) on top of the cube")
        self.overlay_btn.clicked.connect(self._open_overlay_dialog)
        row.addWidget(self.overlay_btn)
        lay.addLayout(row)
        self.info_label = QLabel("No cube loaded")
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)
        return box

    def _build_band_box(self):
        box = QGroupBox("Band view")
        lay = QVBoxLayout(box)
        modes = QHBoxLayout()
        self.single_radio = QRadioButton("Single band")
        self.sum_radio = QRadioButton("Summed bands")
        self.single_radio.setChecked(True)
        group = QButtonGroup(box)
        group.addButton(self.single_radio)
        group.addButton(self.sum_radio)
        modes.addWidget(self.single_radio)
        modes.addWidget(self.sum_radio)
        lay.addLayout(modes)

        row = QHBoxLayout()
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_spin = QSpinBox()
        self.band_slider.valueChanged.connect(self.band_spin.setValue)
        self.band_spin.valueChanged.connect(self.band_slider.setValue)
        self.band_spin.valueChanged.connect(lambda _: self._band_timer.start())
        row.addWidget(self.band_slider, 1)
        row.addWidget(self.band_spin)
        lay.addLayout(row)
        self.band_wl_label = QLabel("")
        lay.addWidget(self.band_wl_label)

        row = QHBoxLayout()
        self.sum_edit = QLineEdit()
        self.sum_edit.setPlaceholderText("e.g. 10-20, 35  or  650-700nm")
        self.sum_edit.returnPressed.connect(self._show_summed_bands)
        self.sum_mode = QComboBox()
        self.sum_mode.addItems(["sum", "mean"])
        sum_btn = QPushButton("Show")
        sum_btn.clicked.connect(self._show_summed_bands)
        row.addWidget(self.sum_edit, 1)
        row.addWidget(self.sum_mode)
        row.addWidget(sum_btn)
        lay.addLayout(row)

        self.single_radio.toggled.connect(self._on_band_mode)
        self._on_band_mode()
        self.band_box = box
        return box

    def _build_rgb_box(self):
        box = QGroupBox("RGB composite (each channel is a band sum)")
        form = QFormLayout(box)
        self.rgb_edits = []
        for name in ("Red", "Green", "Blue"):
            e = QLineEdit()
            e.setPlaceholderText("bands, blank = empty channel")
            e.returnPressed.connect(self.show_rgb)
            form.addRow(name, e)
            self.rgb_edits.append(e)
        row = QHBoxLayout()
        self.truecolor_btn = QPushButton("True colour preset")
        self.truecolor_btn.clicked.connect(self._truecolor_preset)
        show = QPushButton("Show RGB")
        show.clicked.connect(self.show_rgb)
        row.addWidget(self.truecolor_btn)
        row.addWidget(show)
        form.addRow(row)
        self.rgb_box = box
        return box

    def _build_annotation_box(self):
        box = QGroupBox("Annotate")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        for text, fn in (("Point", self.start_points), ("Rectangle", lambda: self.start_shapes("add_rectangle")),
                         ("Polygon", lambda: self.start_shapes("add_polygon")), ("Select", self.start_select)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        hint = QLabel("Polygon: click vertices, double-click or Esc to finish. Delete removes the selection.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px")
        lay.addWidget(hint)
        row = QHBoxLayout()
        for text, fn in (("Save…", self._save_annotations_dialog), ("Load…", self._load_annotations_dialog),
                         ("Export spectra CSV…", self._export_csv_dialog)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.annotation_box = box
        return box

    def _build_spectrum_box(self):
        box = QGroupBox("Spectra")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.follow_cursor = QCheckBox("Follow cursor")
        self.follow_cursor.setChecked(True)
        self.show_std = QCheckBox("±1σ for regions")
        self.show_std.setChecked(True)
        self.show_labels = QCheckBox("Overlay class means")
        self.show_labels.setToolTip("Plot the mean spectrum of every class in the selected labels overlay")
        for cb in (self.follow_cursor, self.show_std, self.show_labels):
            cb.toggled.connect(lambda _: self._plot_timer.start())
            row.addWidget(cb)
        lay.addLayout(row)
        self.figure = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(220)
        self.ax = self.figure.add_subplot(111)
        lay.addWidget(self.canvas, 1)
        self._label_cache = {}
        return box

    def _set_enabled(self, on: bool):
        for w in (self.band_box, self.rgb_box, self.annotation_box, self.overlay_btn):
            w.setEnabled(on)

    # ------------------------------------------------------------ data
    def _open_cube_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open hyperspectral cube", "", CUBE_FILTER)
        if path:
            self.open_cube(path)

    def open_cube(self, path):
        try:
            cube = load_cube(path)
        except Exception as exc:  # surface reader errors to the user rather than the console
            QMessageBox.critical(self, "Could not open cube", str(exc))
            return
        self.set_cube(cube)

    def set_cube(self, cube: Cube):
        self.cube = cube
        self._label_cache.clear()
        r, c, b = cube.shape
        wl = ""
        if cube.wavelengths is not None:
            wl = f", {cube.wavelengths[0]:.4g}–{cube.wavelengths[-1]:.4g} {cube.wavelength_units}".rstrip()
        self.info_label.setText(f"<b>{cube.name}</b><br>{r} × {c} px, {b} bands{wl}, {cube.data.dtype}")
        self.band_slider.setRange(0, b - 1)
        self.band_spin.setRange(0, b - 1)
        self.truecolor_btn.setEnabled(cube.wavelengths is not None)
        self._set_enabled(True)
        self._ensure_annotation_layers()
        self.band_spin.setValue(b // 2)
        self.single_radio.setChecked(True)
        self._band_timer.stop()
        self._show_single_band()
        for name, arr in cube.overlays.items():
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            self.add_overlay(arr, name=name)
        self.viewer.reset_view()
        self.update_spectrum_plot()

    def _open_overlay_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open overlay image", "", OVERLAY_FILTER)
        if path:
            self.add_overlay(path)

    def add_overlay(self, path_or_array, name: str | None = None, as_labels: bool | None = None):
        """Add a processed image above the cube display and below the annotations."""
        if isinstance(path_or_array, (str, Path)):
            name = name or Path(path_or_array).stem
            try:
                arr = load_overlay(path_or_array)
            except Exception as exc:
                QMessageBox.critical(self, "Could not open overlay", str(exc))
                return None
        else:
            arr = np.asarray(path_or_array)
            name = name or "overlay"
        if self.cube is not None and arr.shape[:2] != self.cube.shape[:2]:
            QMessageBox.warning(self, "Size mismatch",
                                f"Overlay is {arr.shape[:2]} but the cube is {self.cube.shape[:2]}; it will not line up.")
        if as_labels is None:
            as_labels = is_label_image(arr)
        if as_labels:
            layer = self.viewer.add_labels(arr.astype(np.int32), name=name, opacity=0.5)
            layer.events.data.connect(lambda _: (self._label_cache.pop(id(layer), None), self._plot_timer.start()))
        elif arr.ndim == 3:
            layer = self.viewer.add_image(arr, name=name, rgb=True, opacity=0.5)
        else:
            layer = self.viewer.add_image(arr, name=name, colormap="magma", opacity=0.5, blending="additive")
        self._keep_annotations_on_top()
        return layer

    # ------------------------------------------------------- band display
    def _on_band_mode(self):
        single = self.single_radio.isChecked()
        self.band_slider.setEnabled(single)
        self.band_spin.setEnabled(single)
        self.sum_edit.setEnabled(not single)
        self.sum_mode.setEnabled(not single)
        if single and self.cube is not None:
            self._show_single_band()

    def _show_single_band(self):
        if self.cube is None or not self.single_radio.isChecked():
            return
        b = self.band_spin.value()
        if self.cube.wavelengths is not None:
            self.band_wl_label.setText(f"{self.cube.wavelengths[b]:.4g} {self.cube.wavelength_units}")
        self._set_image(BAND_LAYER, np.asarray(self.cube.data[:, :, b], dtype=np.float32), f"band {b}")

    def _show_summed_bands(self):
        if self.cube is None:
            return
        self.sum_radio.setChecked(True)
        try:
            bands = parse_band_spec(self.sum_edit.text(), self.cube)
        except ValueError as exc:
            QMessageBox.warning(self, "Band selection", str(exc))
            return
        img = band_image(self.cube, bands, self.sum_mode.currentText())
        self.band_wl_label.setText(f"{self.sum_mode.currentText()} of {len(bands)} bands")
        self._set_image(BAND_LAYER, img.astype(np.float32), f"{self.sum_mode.currentText()} {self.sum_edit.text()}")

    def show_single_band(self, band: int):
        self.single_radio.setChecked(True)
        self.band_spin.setValue(band)
        self._band_timer.stop()
        self._show_single_band()

    def show_summed_bands(self, spec: str, mode: str = "sum"):
        self.sum_edit.setText(spec)
        self.sum_mode.setCurrentText(mode)
        self._show_summed_bands()

    def _truecolor_preset(self):
        for e, (lo, hi) in zip(self.rgb_edits, ((620, 670), (530, 570), (450, 490))):
            e.setText(f"{lo}-{hi}nm")
        self.show_rgb()

    def show_rgb(self, specs=None):
        if self.cube is None:
            return
        if specs is not None:
            for e, s in zip(self.rgb_edits, specs):
                e.setText(s or "")
        channels = []
        try:
            for e in self.rgb_edits:
                t = e.text().strip()
                channels.append(band_image(self.cube, parse_band_spec(t, self.cube)) if t else None)
        except ValueError as exc:
            QMessageBox.warning(self, "RGB band selection", str(exc))
            return
        if all(c is None for c in channels):
            QMessageBox.warning(self, "RGB composite", "Enter bands for at least one channel")
            return
        self._set_image(RGB_LAYER, rgb_composite(channels), "", rgb=True)

    def _set_image(self, name, data, title, rgb=False):
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = data
            if not rgb:
                finite = data[np.isfinite(data)]
                if finite.size:
                    lo, hi = np.percentile(finite, [1, 99])
                    layer.contrast_limits_range = (float(finite.min()), float(max(finite.max(), finite.min() + 1e-9)))
                    layer.contrast_limits = (float(lo), float(max(hi, lo + 1e-9)))
            layer.visible = True
        else:
            if rgb:
                layer = self.viewer.add_image(data, name=name, rgb=True)
            else:
                lo, hi = np.percentile(data[np.isfinite(data)], [1, 99]) if np.isfinite(data).any() else (0, 1)
                layer = self.viewer.add_image(data, name=name, colormap="gray", contrast_limits=(lo, max(hi, lo + 1e-9)))
            # Base images go to the bottom of the stack.
            self.viewer.layers.move(self.viewer.layers.index(layer), 0)
        if title:
            self.viewer.title = f"hsview — {self.cube.name} — {title}"
        # Make the image just shown the visible one when both band and RGB exist.
        other = RGB_LAYER if name == BAND_LAYER else BAND_LAYER
        if other in self.viewer.layers:
            self.viewer.layers[other].visible = False

    # ---------------------------------------------------------- annotations
    def _ensure_annotation_layers(self):
        layers = self.viewer.layers
        if POINTS_LAYER not in layers:
            pts = self.viewer.add_points(np.empty((0, 2)), ndim=2, name=POINTS_LAYER, size=7, border_color="white",
                                         border_width=0.15, features={"id": np.empty(0, dtype=int)})
            pts.feature_defaults["id"] = -1
            pts.events.data.connect(self._on_points_changed)
        if SHAPES_LAYER not in layers:
            sh = self.viewer.add_shapes(ndim=2, name=SHAPES_LAYER, edge_width=2, face_color=[0, 0, 0, 0],
                                        features={"id": np.empty(0, dtype=int)})
            sh.feature_defaults["id"] = -1
            sh.events.data.connect(self._on_shapes_changed)
        self._keep_annotations_on_top()

    def _keep_annotations_on_top(self):
        layers = self.viewer.layers
        for name in (SHAPES_LAYER, POINTS_LAYER):
            if name in layers:
                layers.move(layers.index(name), len(layers))

    @property
    def points_layer(self):
        return self.viewer.layers[POINTS_LAYER]

    @property
    def shapes_layer(self):
        return self.viewer.layers[SHAPES_LAYER]

    def _next_id(self) -> int:
        ids = np.concatenate([self.points_layer.features["id"].to_numpy(), self.shapes_layer.features["id"].to_numpy()])
        return int(ids.max()) + 1 if ids.size else 0

    def _assign_ids(self, layer):
        """Give new annotations a unique id; colour follows the id so it is stable across deletions."""
        # napari copies the selected item's features into new items, so a new
        # annotation shows up as a duplicate id (or -1), not reliably as a sentinel.
        f = layer.features
        ids = f["id"].to_numpy()
        _, first = np.unique(ids, return_index=True)
        new = np.setdiff1d(np.arange(len(ids)), first)
        new = np.union1d(new, np.nonzero(ids < 0)[0])
        if new.size:
            other = self.shapes_layer if layer is self.points_layer else self.points_layer
            keep = np.delete(ids, new)
            all_ids = np.concatenate([keep, other.features["id"].to_numpy(), [-1]])
            start = int(all_ids.max()) + 1
            f.loc[f.index[new], "id"] = np.arange(start, start + new.size)
        return np.array([_color(i) for i in f["id"].to_numpy()]).reshape(-1, 4)

    def _on_points_changed(self, event=None):
        if getattr(event, "action", "added") not in ("added", "removed", "changed") or self._syncing:
            return
        layer = self.points_layer
        if len(layer.data):
            self._syncing = True
            try:
                layer.face_color = self._assign_ids(layer)
            finally:
                self._syncing = False
        self._plot_timer.start()

    def _on_shapes_changed(self, event=None):
        if getattr(event, "action", "added") not in ("added", "removed", "changed") or self._syncing:
            return
        layer = self.shapes_layer
        if len(layer.data):
            self._syncing = True
            try:
                colors = self._assign_ids(layer)
                layer.edge_color = colors
                face = colors.copy()
                face[:, 3] = 0.15
                layer.face_color = face
            finally:
                self._syncing = False
        self._plot_timer.start()

    def start_points(self):
        self.viewer.layers.selection.active = self.points_layer
        self.points_layer.mode = "add"

    def start_shapes(self, mode):
        self.viewer.layers.selection.active = self.shapes_layer
        self.shapes_layer.mode = mode

    def start_select(self):
        active = self.viewer.layers.selection.active
        if active not in (self.points_layer, self.shapes_layer):
            active = self.shapes_layer
            self.viewer.layers.selection.active = active
        active.mode = "select"

    def current_annotations(self) -> dict:
        p, s = self.points_layer, self.shapes_layer
        return ann_io.annotations_to_dict(p.data, p.features["id"].to_numpy(), s.data, s.shape_type,
                                          s.features["id"].to_numpy(), self.cube)

    def load_annotation_dict(self, data: dict, replace: bool = True):
        p, s = self.points_layer, self.shapes_layer
        self._syncing = True
        try:
            if replace:
                p.data = np.empty((0, 2))
                s.data = []
            offset = 0 if replace else self._next_id()
            for pt in data["points"]:
                p.feature_defaults["id"] = pt["id"] + offset
                p.add([[pt["row"], pt["col"]]])
            p.feature_defaults["id"] = -1
            for sh in data["shapes"]:
                s.feature_defaults["id"] = sh["id"] + offset
                v = np.asarray(sh["vertices"], dtype=float)
                if sh["type"] == "rectangle":
                    s.add_rectangles(v)
                elif sh["type"] == "ellipse":
                    s.add_ellipses(v)
                else:
                    s.add_polygons([v])
            s.feature_defaults["id"] = -1
        finally:
            self._syncing = False
        self._on_points_changed()
        self._on_shapes_changed()

    def _save_annotations_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save annotations", f"{self.cube.name}_annotations.json", "JSON (*.json)")
        if path:
            ann_io.save_annotations(path, self.current_annotations())

    def _load_annotations_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load annotations", "", "JSON (*.json)")
        if path:
            try:
                self.load_annotation_dict(ann_io.load_annotations(path))
            except Exception as exc:
                QMessageBox.critical(self, "Could not load annotations", str(exc))

    def _export_csv_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export spectra", f"{self.cube.name}_spectra.csv", "CSV (*.csv)")
        if path:
            n = ann_io.export_spectra_csv(path, self.cube, self.current_annotations())
            self.viewer.status = f"Wrote {n} spectra to {path}"

    # -------------------------------------------------------------- spectra
    def _on_cursor(self, event=None):
        if self.cube is None or not self.follow_cursor.isChecked():
            return
        pos = self.viewer.cursor.position
        new = (int(round(pos[-2])), int(round(pos[-1])))
        if new != self._cursor_pos:
            self._cursor_pos = new
            self._plot_timer.start()

    def _labels_layer(self):
        """The overlay labels layer to summarise: the active one, else the topmost."""
        from napari.layers import Labels

        active = self.viewer.layers.selection.active
        if isinstance(active, Labels):
            return active
        for layer in reversed(self.viewer.layers):
            if isinstance(layer, Labels):
                return layer
        return None

    def update_spectrum_plot(self):
        ax = self.ax
        ax.clear()
        cube = self.cube
        if cube is None:
            self.canvas.draw_idle()
            return
        x = cube.band_axis
        n_lines = 0

        if self.show_labels.isChecked():
            lab = self._labels_layer()
            if lab is not None and lab.data.shape == cube.shape[:2]:
                key = id(lab)
                if key not in self._label_cache:
                    self._label_cache[key] = label_spectra(cube, np.asarray(lab.data))
                for v, spec in self._label_cache[key].items():
                    ax.plot(x, spec, "--", lw=1.2, color=lab.get_color(v), label=f"class {v}")
                    n_lines += 1

        if POINTS_LAYER in self.viewer.layers:
            p = self.points_layer
            for pt, i in zip(p.data, p.features["id"].to_numpy()):
                s = pixel_spectrum(cube, pt[-2], pt[-1])
                if s is not None:
                    ax.plot(x, s, color=_color(i), lw=1.5, label=f"Pt {i} ({int(round(pt[-2]))}, {int(round(pt[-1]))})")
                    n_lines += 1

        if SHAPES_LAYER in self.viewer.layers:
            sh = self.shapes_layer
            for verts, kind, i in zip(sh.data, sh.shape_type, sh.features["id"].to_numpy()):
                if kind in ("line", "path"):
                    continue
                mean, std, n = region_spectrum(cube, verts)
                if mean is None:
                    continue
                col = _color(i)
                ax.plot(x, mean, color=col, lw=2, label=f"{'Rect' if kind == 'rectangle' else 'Poly'} {i} mean, n={n}")
                if self.show_std.isChecked():
                    ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.15, lw=0)
                n_lines += 1

        if self.follow_cursor.isChecked() and self._cursor_pos is not None:
            s = pixel_spectrum(cube, *self._cursor_pos)
            if s is not None:
                ax.plot(x, s, color="0.4", lw=1, ls=":", label=f"cursor {self._cursor_pos}")
                n_lines += 1

        # Mark the band(s) currently displayed.
        if self.single_radio.isChecked():
            ax.axvline(x[self.band_spin.value()], color="0.6", lw=0.8)

        ax.set_xlabel(cube.band_axis_label)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        if n_lines:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=False)
        self.canvas.draw_idle()

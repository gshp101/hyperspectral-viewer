"""The main window: canvas in the middle, display/overlay/spectra docks, menus and toolbars."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QWidgetAction,
)

from .. import __version__
from .. import annotations as ann_io
from ..annotations import CLASS_COLORS, CLASS_NAMES, AnnotationStore
from ..io import Cube, LoadError, cube_file_filter, load_cube, resolve_image_id
from .actions import ActionRegistry
from .display_panel import DisplayPanel
from .icons import swatch_icon
from .image_view import ImageView
from .overlay_panel import OverlayPanel
from .spectrum_panel import ClassPlotDialog, SpectrumPanel
from .toolbar import ToolbarManager

ANNOTATION_FILTER = "Annotations (*.pkl *.json);;Pickle (*.pkl);;JSON (*.json);;All files (*)"


class ClassSelectorAction(QWidgetAction):
    """Toolbar combo box for the current annotation class (can sit on several toolbars)."""

    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.window = window

    def createWidget(self, parent):
        combo = QComboBox(parent)
        for i, name in enumerate(CLASS_NAMES):
            combo.addItem(swatch_icon(CLASS_COLORS[i]), name)
        combo.setCurrentIndex(self.window.view.current_class)
        combo.setToolTip("Class for new annotations (keys 1–7); also recolours selected annotations")
        combo.currentIndexChanged.connect(self.window.set_class)
        return combo

    def sync(self, cls: int):
        for w in self.createdWidgets():
            w.blockSignals(True)
            w.setCurrentIndex(cls)
            w.blockSignals(False)


class BandReadoutAction(QWidgetAction):
    """Toolbar label showing the band and wavelength last chosen."""

    def __init__(self, window):
        super().__init__(window)
        self.text_value = "Band: --"

    def createWidget(self, parent):
        label = QLabel(self.text_value, parent)
        label.setMinimumWidth(150)
        label.setContentsMargins(6, 0, 6, 0)
        return label

    def set_text(self, text: str):
        self.text_value = text
        for w in self.createdWidgets():
            w.setText(text)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.settings = settings if settings is not None else QSettings("hsview", "hsview")
        self.cube: Cube | None = None
        self.annotation_path: str | None = None
        self.store = AnnotationStore()
        self.setWindowTitle("hsview")
        self.resize(1400, 900)

        self.view = ImageView(self.store)
        self.setCentralWidget(self.view)
        self.display = DisplayPanel(self.settings)
        self.overlays = OverlayPanel(self.view, self.settings)
        self.spectra = SpectrumPanel(self.store)
        self.spectra.label_source = self.overlays.label_overlay

        self.dock_display = self._dock("Display", self.display, Qt.DockWidgetArea.LeftDockWidgetArea, scroll=True)
        self.dock_overlays = self._dock("Overlays", self.overlays, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_spectra = self._dock("Spectra", self.spectra, Qt.DockWidgetArea.RightDockWidgetArea)
        self.resizeDocks([self.dock_display, self.dock_spectra], [330, 460], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_display, self.dock_overlays], [700, 160], Qt.Orientation.Vertical)

        self.registry = ActionRegistry(self, self.settings)
        self.class_selector = ClassSelectorAction(self)
        self.band_readout = BandReadoutAction(self)
        self.registry.add_widget_item("class_selector", self.class_selector)
        self.registry.add_widget_item("band_readout", self.band_readout)
        self._connect_actions()
        self._build_menus()
        self.toolbars = ToolbarManager(self, self.registry, self.settings)
        self.toolbars.apply()

        self.status = QLabel("")
        self.statusBar().addPermanentWidget(self.status)

        self.display.imageChanged.connect(self._show_image)
        self.display.bandReadout.connect(self._on_band_readout)
        self.view.cursorMoved.connect(self._on_cursor)
        self.view.modeChanged.connect(self._on_mode)
        self.view.scene().selectionChanged.connect(self._update_actions)
        self.overlays.changed.connect(self.spectra.overlays_changed)
        self.store.subscribe(self._update_actions)

        state = self.settings.value("window/state")
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        self.registry["mode_navigate"].setChecked(True)
        self.registry["class_0"].setChecked(True)
        self._update_actions()

    # ----------------------------------------------------------------- setup
    def _dock(self, title, widget, area, scroll=False):
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower()}")
        if scroll:
            sa = QScrollArea()
            sa.setWidget(widget)
            sa.setWidgetResizable(True)
            sa.setFrameShape(QScrollArea.Shape.NoFrame)
            widget = sa
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _connect_actions(self):
        r = self.registry
        slots = {
            "open_cube": self.open_cube_dialog,
            "open_by_id": self.open_by_id_dialog,
            "open_reference": lambda: self.display.reference.load_dialog(),
            "add_overlay": lambda: self.overlays.add_dialog(),
            "load_filter": lambda: self.display.channels[0].load_filter_dialog(),
            "load_annotations": self.load_annotations_dialog,
            "save_annotations": self.save_annotations,
            "save_annotations_as": self.save_annotations_as,
            "save_spectra": self.save_spectra_dialog,
            "export_csv": self.export_csv_dialog,
            "quit": self.close,
            "undo": self.store.undo,
            "redo": self.store.redo,
            "delete_selected": self.view.delete_selected,
            "select_all": self._select_all,
            "clear_annotations": self.clear_annotations,
            "customize": self.toolbars_customize,
            "zoom_in": self.view.zoom_in,
            "zoom_out": self.view.zoom_out,
            "zoom_fit": self.view.zoom_fit,
            "zoom_actual": self.view.zoom_actual,
            "prev_band": lambda: self.display.step_band(-1),
            "next_band": lambda: self.display.step_band(1),
            "true_colour": self.display.true_colour,
            "reverse_rgb": self.display.reference.reverse_rgb,
            "reset_contrast": self.display.reset_contrast,
            "plot_points": lambda: self.plot_by_class("point"),
            "plot_regions": lambda: self.plot_by_class("region"),
            "about": self.about,
        }
        for action_id, slot in slots.items():
            r.connect(action_id, lambda *_, f=slot: f())
        for mode in ("navigate", "select", "point", "rect", "polygon"):
            r[f"mode_{mode}"].triggered.connect(lambda _, m=mode: self.view.set_mode(m))
        for i in range(len(CLASS_NAMES)):
            r[f"class_{i}"].triggered.connect(lambda _, c=i: self.set_class(c))
        for action_id, dock in (("toggle_display", self.dock_display), ("toggle_spectra", self.dock_spectra),
                                ("toggle_overlays", self.dock_overlays)):
            act = r[action_id]
            act.setChecked(True)
            act.toggled.connect(dock.setVisible)
            dock.visibilityChanged.connect(lambda vis, a=act: (a.blockSignals(True), a.setChecked(vis),
                                                                a.blockSignals(False)))

    def _build_menus(self):
        r = self.registry
        mb = self.menuBar()
        menus = {
            "&File": ["open_cube", "open_by_id", "open_reference", "add_overlay", "load_filter", None,
                      "load_annotations", "save_annotations", "save_annotations_as", None,
                      "save_spectra", "export_csv", None, "quit"],
            "&Edit": ["undo", "redo", None, "select_all", "delete_selected", "clear_annotations", None, "customize"],
            "&Mode": ["mode_navigate", "mode_select", "mode_point", "mode_rect", "mode_polygon"],
            "&Class": [f"class_{i}" for i in range(len(CLASS_NAMES))],
            "&View": ["zoom_in", "zoom_out", "zoom_fit", "zoom_actual", None, "prev_band", "next_band",
                      "true_colour", "reverse_rgb", "reset_contrast", None,
                      "toggle_display", "toggle_overlays", "toggle_spectra"],
            "&Plot": ["plot_points", "plot_regions"],
            "&Help": ["about"],
        }
        for title, ids in menus.items():
            menu = mb.addMenu(title)
            for action_id in ids:
                if action_id is None:
                    menu.addSeparator()
                else:
                    menu.addAction(r[action_id])
            if title == "&View":
                menu.addSeparator()
                self.toolbar_menu = menu.addMenu("Toolbars")
                self.toolbar_menu.aboutToShow.connect(self._fill_toolbar_menu)

    def _fill_toolbar_menu(self):
        self.toolbar_menu.clear()
        for tb in self.toolbars.toolbars:
            self.toolbar_menu.addAction(tb.toggleViewAction())
        self.toolbar_menu.addSeparator()
        self.toolbar_menu.addAction(self.registry["customize"])

    def createPopupMenu(self):
        menu = super().createPopupMenu()
        menu.addSeparator()
        menu.addAction(self.registry["customize"])
        return menu

    def toolbars_customize(self):
        self.toolbars.customize()

    # ---------------------------------------------------------------- state
    def _update_actions(self):
        r = self.registry
        has_cube = self.cube is not None
        for action_id in ("open_reference", "add_overlay", "load_filter", "load_annotations", "save_annotations",
                          "save_annotations_as", "save_spectra", "export_csv", "plot_points", "plot_regions",
                          "zoom_in", "zoom_out", "zoom_fit", "zoom_actual", "prev_band", "next_band",
                          "reset_contrast", "clear_annotations", "select_all"):
            r[action_id].setEnabled(has_cube)
        r["true_colour"].setEnabled(has_cube and self.cube.wavelengths is not None)
        r["reverse_rgb"].setEnabled(self.display.reference.img is not None and self.display.reference.img.ndim == 3)
        r["undo"].setEnabled(self.store.can_undo)
        r["redo"].setEnabled(self.store.can_redo)
        try:
            r["delete_selected"].setEnabled(bool(self.view.selected_ids()))
        except RuntimeError:  # scene already gone during shutdown
            pass
        title = "hsview"
        if self.cube is not None:
            title = f"{self.cube.name} — hsview"
            if self.store.modified and len(self.store):
                title = "• " + title
        self.setWindowTitle(title)

    def _show_image(self, img):
        self.view.set_image(img)
        # Mark the displayed band on the spectrum plot when a single band is shown.
        ch = next((c for c in self.display.channels if c.route.currentIndex() and c.source.currentText() == "Band"), None)
        self.spectra.set_band_marker(ch.spin.value() if ch else None)
        self._update_actions()

    def _on_band_readout(self, text):
        self.band_readout.set_text(text)

    def _on_cursor(self, x, y):
        self.spectra.set_cursor(x, y)
        if math.isnan(x) or self.cube is None:
            self.status.setText("")
            return
        c, r = int(x), int(y)
        if 0 <= r < self.cube.shape[0] and 0 <= c < self.cube.shape[1]:
            self.status.setText(f"x {c}  y {r}   zoom {self.view.zoom:.2f}×")

    def _on_mode(self, mode):
        act = self.registry[f"mode_{mode}"]
        if not act.isChecked():
            act.setChecked(True)

    def set_class(self, cls: int):
        cls = int(cls)
        self.view.current_class = cls
        self.registry[f"class_{cls}"].setChecked(True)
        self.class_selector.sync(cls)
        ids = self.view.selected_ids()
        if ids:  # choosing a class with annotations selected recolours them
            self.store.update({i: {"cls": cls} for i in ids})

    def _select_all(self):
        self.view.set_mode("select")
        self.view.select_ids([a.id for a in self.store])

    # ------------------------------------------------------------- cubes
    def _last_dir(self):
        return self.settings.value("paths/last_dir", "")

    def _remember(self, path):
        self.settings.setValue("paths/last_dir", str(Path(path).parent))

    def open_cube_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open hyperspectral cube", self._last_dir(), cube_file_filter())
        if path:
            self.open_cube(path)

    def open_by_id_dialog(self):
        image_id, ok = QInputDialog.getInt(self, "Open by database id", "Image id:", 1, 0, 10 ** 9)
        if not ok:
            return
        try:
            path = resolve_image_id(image_id, self.settings.value("paths/image_dir") or None)
        except LoadError as exc:
            directory = QFileDialog.getExistingDirectory(self, f"{exc}. Choose the image directory", self._last_dir())
            if not directory:
                return
            self.settings.setValue("paths/image_dir", directory)
            try:
                path = resolve_image_id(image_id, directory)
            except LoadError as exc2:
                QMessageBox.warning(self, "Open by id", str(exc2))
                return
        self.open_cube(path)

    def open_cube(self, path, **kwargs) -> bool:
        if not self.confirm_discard_annotations():
            return False
        try:
            cube = load_cube(path, **kwargs)
        except Exception as exc:  # readers raise many kinds of errors; show them all
            QMessageBox.warning(self, "Could not open file", str(exc))
            return False
        self._remember(path)
        self.set_cube(cube)
        return True

    def set_cube(self, cube: Cube):
        self.cube = cube
        self.annotation_path = None
        self.store.replace([], undoable=False)
        self.store._undo.clear()
        self.overlays.set_shape(cube.shape[:2])
        self.spectra.set_cube(cube)
        self.display.set_cube(cube)
        self.view.zoom_fit()
        for key in ("wavelength_warning", "reference_warning"):
            if key in cube.metadata:
                self.statusBar().showMessage(cube.metadata[key], 8000)
        self._update_actions()

    # -------------------------------------------------------- annotations
    def confirm_discard_annotations(self) -> bool:
        if not self.store.modified or not len(self.store):
            return True
        answer = QMessageBox.question(
            self, "Unsaved annotations", "Save the current annotations first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save_annotations()
        return answer == QMessageBox.StandardButton.Discard

    def load_annotations_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load annotations", self._last_dir(), ANNOTATION_FILTER)
        if path:
            self.load_annotations(path)

    def load_annotations(self, path) -> bool:
        try:
            items = ann_io.load_annotations(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load annotations", str(exc))
            return False
        self.store.replace(items)
        self.store.modified = False
        self.annotation_path = str(path)
        self._update_actions()
        return True

    def save_annotations(self) -> bool:
        if self.annotation_path:
            return self._write_annotations(self.annotation_path)
        return self.save_annotations_as()

    def save_annotations_as(self) -> bool:
        default = self.annotation_path or str(Path(self._last_dir()) / f"{self.cube.name}_annotations.pkl")
        path, _ = QFileDialog.getSaveFileName(self, "Save annotations", default, ANNOTATION_FILTER)
        return bool(path) and self._write_annotations(path)

    def _write_annotations(self, path) -> bool:
        try:
            ann_io.save_annotations(path, self.store, self.cube)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save annotations", str(exc))
            return False
        self.annotation_path = str(path)
        self.store.modified = False
        self._remember(path)
        self.statusBar().showMessage(f"Saved {len(self.store)} annotations to {path}", 5000)
        self._update_actions()
        return True

    def clear_annotations(self):
        if len(self.store) and QMessageBox.question(self, "Clear annotations",
                                                    "Remove all annotations?") == QMessageBox.StandardButton.Yes:
            self.store.clear()

    def save_spectra_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save annotation spectra",
                                              str(Path(self._last_dir()) / f"{self.cube.name}_spectra.pkl"),
                                              "Pickle (*.pkl)")
        if path:
            ann_io.save_annotation_spectra(path, self.cube, self.store)
            self.statusBar().showMessage(f"Saved spectra to {path}", 5000)

    def export_csv_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export spectra",
                                              str(Path(self._last_dir()) / f"{self.cube.name}_spectra.csv"),
                                              "CSV (*.csv)")
        if path:
            n = ann_io.export_spectra_csv(path, self.cube, self.store)
            self.statusBar().showMessage(f"Wrote {n} spectra to {path}", 5000)

    def plot_by_class(self, which: str):
        groups: dict[int, list[np.ndarray]] = {}
        for a in self.store:
            if (which == "point") != (a.kind == "point"):
                continue
            px = a.pixels(self.cube)
            if len(px):
                groups.setdefault(a.cls, []).append(px.mean(axis=0))
        if not groups:
            QMessageBox.information(self, "Nothing to plot",
                                    "There are no points." if which == "point" else "There are no rectangles or polygons.")
            return None
        title = "Point spectra by class" if which == "point" else "Region mean spectra by class"
        dlg = ClassPlotDialog(self.cube, groups, f"{title} — {self.cube.name}", self)
        dlg.show()
        return dlg

    # --------------------------------------------------------------- misc
    def about(self):
        QMessageBox.about(self, "About hsview",
                          f"<b>hsview {__version__}</b><br>View and annotate hyperspectral cubes.<br><br>"
                          "Right-click a toolbar to customize it.")

    def closeEvent(self, event):
        if not self.confirm_discard_annotations():
            event.ignore()
            return
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)

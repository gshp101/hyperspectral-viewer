"""The main window: canvas in the middle, display/overlay/spectra docks, menus and toolbars."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QWidgetAction,
)

from .. import __version__
from .. import annotations as ann_io
from ..annotations import CLASS_COLORS, CLASS_NAMES, AnnotationStore
from ..analysis.segmentation import is_segmentation_file, save_segmentation, segmentation_from_labels
from ..io import CUBE_SAVE_FILTER, Cube, LoadError, cube_file_filter, load_cube, resolve_image_id, save_cube
from ..ui_spec import BIAS_STEP, CONTRAST_STEP, TRUE_COLOUR, menu_layout
from .actions import ActionRegistry
from .analysis import ANALYSES
from .display_panel import REF_ROUTES, ROUTES, DisplayPanel
from .app_icons import APP_ICONS, DEFAULT_APP_ICON, app_icon
from .icons import swatch_icon
from .image_view import ImageView
from .overlay_panel import OVERLAY_FILTER, OverlayPanel
from .segmentation import run_supervised, run_unsupervised
from .spectrum_panel import ClassPlotDialog, SpectrumPanel
from .toolbar import ToolbarManager

ANNOTATION_FILTER = "Annotations (*.pkl *.json);;Pickle (*.pkl);;JSON (*.json);;All files (*)"
ANNOTATION_LOAD_FILTER = ("Annotations and segmentations (*.pkl *.json *.npz);;Pickle (*.pkl);;JSON (*.json);;"
                          "Segmentation (*.npz *.pkl);;All files (*)")
SEGMENTATION_FILTER = "Pickle (*.pkl);;NumPy archive (*.npz)"


class BandSpecDialog(QDialog):
    """Ask for one band selection per row (e.g. '10-20, 35' or '650-700nm') and sum/mean."""

    def __init__(self, title, labels, values, mode="sum", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        form = QFormLayout(self)
        self.edits = []
        for label, value in zip(labels, values):
            edit = QLineEdit(value)
            edit.setPlaceholderText("10-20, 35  or  650-700nm")
            form.addRow(label, edit)
            self.edits.append(edit)
        self.mode = QComboBox()
        self.mode.addItems(["sum", "mean"])
        self.mode.setCurrentText(mode)
        form.addRow("Combine by", self.mode)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return [e.text().strip() for e in self.edits]


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
        self.cube_history: list[Cube] = []  # cubes before each analysis, oldest first
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
        self.overlays.changed.connect(self._update_actions)
        self.display.reference.changed.connect(self._update_actions)
        self.store.subscribe(self._update_actions)

        state = self.settings.value("window/state")
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        sep = self.registry["plot_separate"]
        sep.setChecked(self.settings.value("plot/separate_windows", False, type=bool))
        sep.toggled.connect(lambda on: self.settings.setValue("plot/separate_windows", on))
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
            "save_cube": self.save_cube_dialog,
            "finish_polygon": self.view.finish_polygon,
            "cancel_drawing": self.view.cancel_drawing,
            "show_rgb": self.show_rgb_dialog,
            "contrast_up": lambda: self.display.step_contrast(CONTRAST_STEP),
            "contrast_down": lambda: self.display.step_contrast(-CONTRAST_STEP),
            "bias_up": lambda: self.display.step_bias(BIAS_STEP),
            "bias_down": lambda: self.display.step_bias(-BIAS_STEP),
            "ref_opacity": self.reference_opacity_dialog,
            "remove_overlay": self.overlays.remove_current,
            "clear_overlays": self.clear_overlays,
            "show_overlays": lambda: self.overlays.set_all_visible(True),
            "hide_overlays": lambda: self.overlays.set_all_visible(False),
            "overlay_opacity": lambda: self.overlay_opacity_dialog(),
            "analysis_undo": self.undo_analysis,
            "segment_unsupervised": lambda: self.run_segmentation(run_unsupervised),
            "segment_supervised": lambda: self.run_segmentation(run_supervised),
            "save_segmentation": self.save_segmentation_dialog,
            "open_segmentation": self.open_segmentation_dialog,
            "analysis_restore": self.restore_original_cube,
            "plot_points": lambda: self.plot_by_class("point"),
            "plot_regions": lambda: self.plot_by_class("region"),
            "close_plots": self.close_plot_windows,
            "about": self.about,
        }
        for c in range(len(self.display.channels)):
            slots[f"ch{c}_band"] = lambda c=c: self.channel_band_dialog(c)
            slots[f"ch{c}_width"] = lambda c=c: self.channel_width_dialog(c)
            slots[f"ch{c}_sum"] = lambda c=c: self.channel_sum_dialog(c)
            slots[f"ch{c}_filter"] = lambda c=c: self.display.channels[c].load_filter_dialog()
        for cmd in ANALYSES:
            slots[cmd.id] = lambda cmd=cmd: self.run_analysis(cmd)
        for action_id, slot in slots.items():
            r.connect(action_id, lambda *_, f=slot: f())
        # Menu items that mirror combo boxes and check boxes in the panels stay in sync both ways.
        for c, ch in enumerate(self.display.channels):
            self._bind_combo(ch.route, [f"ch{c}_route_{x.lower()}" for x in ROUTES])
        self._bind_combo(self.display.reference.route, [f"ref_route_{x.lower()}" for x in REF_ROUTES])
        self._bind_combo(self.display.scaling, [f"scaling_{self.display.scaling.itemData(i)}"
                                                for i in range(self.display.scaling.count())])
        for action_id, box in (("spectra_cursor", self.spectra.follow_cursor), ("spectra_std", self.spectra.show_std),
                               ("spectra_by_class", self.spectra.by_class),
                               ("spectra_labels", self.spectra.show_labels)):
            act = r[action_id]
            act.setChecked(box.isChecked())
            act.toggled.connect(box.setChecked)
            box.toggled.connect(act.setChecked)
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

    def _bind_combo(self, combo, action_ids):
        """Radio menu items that choose a combo box entry and follow it when it changes."""
        acts = [self.registry[a] for a in action_ids]
        for i, act in enumerate(acts):
            act.triggered.connect(lambda _, i=i: combo.setCurrentIndex(i))
        sync = lambda i: 0 <= i < len(acts) and acts[i].setChecked(True)  # noqa: E731
        combo.currentIndexChanged.connect(sync)
        sync(combo.currentIndex())

    def _menu_layout(self):
        return menu_layout(ANALYSES, len(self.display.channels))

    def _add_menu_items(self, menu, ids):
        for entry in ids:
            if entry is None:
                menu.addSeparator()
            elif isinstance(entry, tuple):
                self._add_menu_items(menu.addMenu(entry[0]), entry[1])
            else:
                menu.addAction(self.registry[entry])

    def _build_menus(self):
        mb = self.menuBar()
        self.menus = {}
        for title, ids in self._menu_layout().items():
            menu = mb.addMenu(title)
            menu.setToolTipsVisible(True)
            self.menus[title.replace("&", "")] = menu
            self._add_menu_items(menu, ids)
        view = self.menus["View"]
        view.addSeparator()
        self._build_app_icon_menu(view.addMenu("Application icon"))
        self.overlay_list_menu = None
        self.menus["Overlays"].aboutToShow.connect(self._fill_overlay_menu)
        self.toolbar_menu = self.menus["Toolbars"]
        self.toolbar_menu.aboutToShow.connect(self._fill_toolbar_menu)

    def _build_app_icon_menu(self, menu):
        group = QActionGroup(self)
        current = self.settings.value("app/icon", DEFAULT_APP_ICON)
        for key, (label, _) in APP_ICONS.items():
            act = menu.addAction(app_icon(key), label)
            act.setCheckable(True)
            act.setChecked(key == current)
            act.triggered.connect(lambda _, k=key: self.set_app_icon(k))
            group.addAction(act)
        self.set_app_icon(current)

    def set_app_icon(self, key: str):
        """Use one of the candidate icons for this window and the application, and remember it."""
        if key not in APP_ICONS:
            key = DEFAULT_APP_ICON
        icon = app_icon(key)
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)
        self.settings.setValue("app/icon", key)

    def _fill_toolbar_menu(self):
        self.toolbars.fill_menu(self.toolbar_menu)

    def _fill_overlay_menu(self):
        """One submenu per overlay (visibility, opacity, selection, removal) after the fixed items."""
        menu = self.menus["Overlays"]
        for act in getattr(self, "_overlay_entries", []):
            menu.removeAction(act)
        self._overlay_entries = [menu.addSeparator()]
        current = self.overlays.current()
        for key, ov in self.overlays.overlays.items():
            sub = menu.addMenu(ov.name + ("  (selected)" if ov is current else ""))
            vis = sub.addAction("Visible")
            vis.setCheckable(True)
            vis.setChecked(ov.visible)
            vis.toggled.connect(lambda on, k=key: self.overlays.set_visible(k, on))
            sub.addAction("Select", lambda k=key: self.overlays.select(k))
            sub.addAction("Opacity…", lambda k=key: self.overlay_opacity_dialog(k))
            sub.addSeparator()
            sub.addAction("Remove", lambda k=key: self.overlays.remove(k))
            self._overlay_entries.append(sub.menuAction())
        has = bool(self.overlays.overlays)
        for action_id in ("clear_overlays", "show_overlays", "hide_overlays"):
            self.registry[action_id].setEnabled(has)
        self.registry["remove_overlay"].setEnabled(current is not None)
        self.registry["overlay_opacity"].setEnabled(current is not None)

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
        for spec in r.specs.values():
            if spec.category.startswith("Channel ") or spec.category == "Analysis" or spec.id in (
                    "save_cube", "show_rgb", "contrast_up", "contrast_down", "bias_up", "bias_down"):
                r[spec.id].setEnabled(has_cube)
        r["segment_unsupervised"].setEnabled(has_cube)
        r["segment_supervised"].setEnabled(has_cube and len(self.store) > 0)
        r["open_segmentation"].setEnabled(has_cube)
        r["save_segmentation"].setEnabled(self.segmentation_overlay() is not None)
        r["analysis_undo"].setEnabled(bool(self.cube_history))
        r["analysis_restore"].setEnabled(bool(self.cube_history))
        r["true_colour"].setEnabled(has_cube and self.cube.wavelengths is not None)
        has_ref = self.display.reference.img is not None
        r["reverse_rgb"].setEnabled(has_ref and self.display.reference.img.ndim == 3)
        for action_id in ("ref_route_show", "ref_route_overlay", "ref_opacity"):
            r[action_id].setEnabled(has_ref)
        cur = self.overlays.current()
        r["remove_overlay"].setEnabled(cur is not None)
        r["overlay_opacity"].setEnabled(cur is not None)
        has_ov = bool(self.overlays.overlays)
        for action_id in ("clear_overlays", "show_overlays", "hide_overlays"):
            r[action_id].setEnabled(has_ov)
        r["finish_polygon"].setEnabled(self.view.mode == "polygon")
        r["cancel_drawing"].setEnabled(self.view.mode in ("polygon", "rect"))
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
        self._update_actions()

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
        self.cube_history.clear()
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

    def save_cube_dialog(self):
        title = "Save processed cube" if self.cube_history or self.cube.metadata.get("analysis") else "Save cube"
        name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in self.cube.name).strip("_") or "cube"
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(self._last_dir()) / f"{name}.pkl"),
                                              CUBE_SAVE_FILTER)
        if path:
            try:
                save_cube(path, self.cube)
            except OSError as exc:
                QMessageBox.warning(self, "Could not save cube", str(exc))
                return
            self._remember(path)
            self.statusBar().showMessage(f"Saved cube to {path}", 5000)

    # ----------------------------------------------------------- analysis
    def replace_cube(self, cube: Cube):
        """Show a processed cube of the same size, keeping annotations, overlays and display settings."""
        if cube.shape[:2] != self.cube.shape[:2] or cube.n_bands != self.cube.n_bands:
            raise ValueError("A replacement cube must have the same shape.")
        self.cube = cube
        self.display.replace_cube(cube)
        self.spectra.set_cube(cube)
        self._update_actions()

    def run_analysis(self, cmd):
        if self.cube is None:
            return None
        try:
            result = cmd.run(self)
        except Exception as exc:  # show numerical failures instead of crashing
            QMessageBox.warning(self, cmd.text.replace("…", ""), str(exc))
            return None
        if result is None:
            return None
        self.cube_history.append(self.cube)
        self.replace_cube(result)
        self.statusBar().showMessage(f"{cmd.text.replace('…', '')}: showing {result.name}", 8000)
        return result

    # ------------------------------------------------------- segmentation
    def segmentation_overlay(self):
        """The selected overlay if it is a label image, else the most recent visible (or any) label overlay."""
        cur = self.overlays.current()
        if cur is not None and cur.as_labels:
            return cur
        labels = [ov for ov in self.overlays.overlays.values() if ov.as_labels]
        visible = [ov for ov in labels if ov.visible]
        return (visible or labels or [None])[-1]

    def run_segmentation(self, run):
        """Run a segmentation dialog on the current (possibly processed) cube and show the result as an overlay."""
        if self.cube is None:
            return None
        try:
            seg = run(self)
        except Exception as exc:  # show numerical failures instead of crashing
            QMessageBox.warning(self, "Segmentation", str(exc))
            return None
        if seg is None:
            return None
        for ov in self.overlays.overlays.values():  # only the newest segmentation is shown at first
            if ov.segmentation is not None and ov.visible:
                self.overlays.set_visible(ov.key, False)
        ov = self.overlays.add_segmentation(seg)
        if ov is not None:
            self.statusBar().showMessage(f"Segmentation: {seg.name}. Save it with Segment → Save segmentation…", 8000)
        self._update_actions()
        return seg

    def save_segmentation_dialog(self):
        ov = self.segmentation_overlay()
        if ov is None:
            return
        seg = ov.segmentation or segmentation_from_labels(ov.data, self.cube, ov.name, ov.colors)
        base = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in self.cube.name).strip("_") or "cube"
        path, _ = QFileDialog.getSaveFileName(self, f"Save segmentation '{ov.name}'",
                                              str(Path(self._last_dir()) / f"{base}_segmentation.pkl"),
                                              SEGMENTATION_FILTER)
        if not path:
            return
        try:
            save_segmentation(path, seg)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save segmentation", str(exc))
            return
        self._remember(path)
        self.statusBar().showMessage(f"Saved segmentation to {path}", 5000)

    def open_segmentation_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open segmentation", self._last_dir(),
                                              "Segmentations (*.pkl *.npz);;" + OVERLAY_FILTER)
        if path:
            self.overlays.add_file(path)

    def undo_analysis(self):
        if self.cube_history:
            self.replace_cube(self.cube_history.pop())

    def restore_original_cube(self):
        if self.cube_history:
            original = self.cube_history[0]
            self.cube_history.clear()
            self.replace_cube(original)

    # ------------------------------------------------------------ display
    def channel_band_dialog(self, c: int):
        ch = self.display.channels[c]
        n = self.cube.n_bands
        b, ok = QInputDialog.getInt(self, f"Band channel {c + 1}", f"Band (0–{n - 1}):", ch.spin.value(), 0, n - 1)
        if ok:
            ch.set_band(b)
            if not ch.route.currentIndex():
                ch.route.setCurrentText("Gray")

    def channel_width_dialog(self, c: int):
        ch = self.display.channels[c]
        n = self.cube.n_bands
        w, ok = QInputDialog.getInt(self, f"Band channel {c + 1}", "Average this many bands:",
                                    ch.width_spin.value(), 1, n)
        if ok:
            ch.set_width(w)

    def channel_sum_dialog(self, c: int):
        ch = self.display.channels[c]
        dlg = BandSpecDialog(f"Band channel {c + 1}", ["Bands"], [ch.spec.text()], ch.sum_mode.currentText(), self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.values()[0]:
            ch.set_spec(dlg.values()[0], dlg.mode.currentText())
            if not ch.route.currentIndex():
                ch.route.setCurrentText("Gray")

    def show_rgb_dialog(self):
        chans = self.display.channels
        defaults = [ch.spec.text() for ch in chans]
        if not any(defaults) and self.cube.wavelengths is not None:
            defaults = list(TRUE_COLOUR)
        dlg = BandSpecDialog("RGB from bands", ["Red", "Green", "Blue"], defaults, chans[0].sum_mode.currentText(), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.display.show_rgb(dlg.values(), dlg.mode.currentText())

    def reference_opacity_dialog(self):
        ref = self.display.reference
        v, ok = QInputDialog.getInt(self, "Reference image", "Opacity (%):", ref.opacity.value(), 0, 100)
        if ok:
            ref.opacity.setValue(v)
            if ref.route.currentText() == "None":
                ref.route.setCurrentText("Overlay")

    def overlay_opacity_dialog(self, key: str | None = None):
        ov = self.overlays.overlays.get(key) if key else self.overlays.current()
        if ov is None:
            return
        v, ok = QInputDialog.getInt(self, ov.name, "Opacity (%):", int(round(ov.opacity * 100)), 0, 100)
        if ok:
            self.overlays.set_opacity(ov.key, v / 100)

    def clear_overlays(self):
        if self.overlays.overlays and QMessageBox.question(
                self, "Remove overlays", "Remove all overlays?") == QMessageBox.StandardButton.Yes:
            self.overlays.clear()

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
        path, _ = QFileDialog.getOpenFileName(self, "Load annotations", self._last_dir(), ANNOTATION_LOAD_FILTER)
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
        # Annotations taken from a segmentation file must not be saved over that file.
        self.annotation_path = None if is_segmentation_file(path) else str(path)
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
        title = "Point spectra" if which == "point" else "Region mean spectra"
        if self.registry["plot_separate"].isChecked():
            # One window per class, cascaded so they don't hide each other.
            dialogs = []
            origin = self.geometry().topLeft() + QPoint(60, 60)
            for i, cls in enumerate(sorted(groups)):
                dlg = ClassPlotDialog(self.cube, {cls: groups[cls]},
                                      f"{title} — {CLASS_NAMES[cls]} — {self.cube.name}", self)
                dlg.move(origin + QPoint(30 * i, 30 * i))
                dialogs.append(dlg)
        else:
            dialogs = [ClassPlotDialog(self.cube, groups, f"{title} by class — {self.cube.name}", self)]
        for dlg in dialogs:
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dlg.show()
        return dialogs

    def close_plot_windows(self):
        for dlg in self.findChildren(ClassPlotDialog):
            dlg.close()

    # --------------------------------------------------------------- misc
    def about(self):
        QMessageBox.about(self, "About hsview",
                          f"<b>hsview {__version__}</b><br>View and annotate hyperspectral cubes.<br><br>"
                          "Every command is in the menus. Use the Toolbars menu (or right-click a toolbar) to "
                          "change the toolbars.")

    def closeEvent(self, event):
        if not self.confirm_discard_annotations():
            event.ignore()
            return
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)

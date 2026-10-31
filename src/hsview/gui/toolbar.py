"""User-configurable toolbars and the dialog that edits them.

The configuration is a list of toolbars, each a name, a visibility flag and an
ordered list of item ids (actions, widget items, separators and spacers),
plus a shared icon size and button style. It is stored as JSON in QSettings;
toolbar positions are kept by QMainWindow.saveState.
"""
from __future__ import annotations

import copy
import json
import re

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .actions import SEPARATOR, SPACER, ActionRegistry

DEFAULT_CONFIG = {
    "icon_size": 22,
    "style": "icon",
    "toolbars": [
        {"name": "File", "visible": True, "items": [
            "open_cube", "open_reference", "add_overlay", SEPARATOR,
            "load_annotations", "save_annotations", "save_spectra", "export_csv"]},
        {"name": "Annotate", "visible": True, "items": [
            "mode_navigate", "mode_select", "mode_point", "mode_rect", "mode_polygon", SEPARATOR,
            "class_selector", SEPARATOR, "undo", "redo", "delete_selected", "clear_annotations"]},
        {"name": "View", "visible": True, "items": [
            "zoom_in", "zoom_out", "zoom_fit", "zoom_actual", SEPARATOR,
            "prev_band", "next_band", "band_readout", SEPARATOR,
            "true_colour", "reset_contrast", SEPARATOR, "plot_points", "plot_regions", "plot_separate"]},
    ],
}

STYLES = {
    "icon": ("Icons only", Qt.ToolButtonStyle.ToolButtonIconOnly),
    "text": ("Text only", Qt.ToolButtonStyle.ToolButtonTextOnly),
    "beside": ("Text beside icons", Qt.ToolButtonStyle.ToolButtonTextBesideIcon),
    "under": ("Text under icons", Qt.ToolButtonStyle.ToolButtonTextUnderIcon),
}
ICON_SIZES = (16, 20, 22, 24, 28, 32, 40, 48)
SETTINGS_KEY = "toolbars/config"


def _object_name(name: str) -> str:
    return "toolbar_" + re.sub(r"\W+", "_", name.strip().lower())


def validate_config(cfg, known_ids) -> dict:
    """Return a clean config: unknown items dropped, names unique, defaults filled in."""
    out = copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(cfg, dict):
        return out
    if cfg.get("icon_size") in ICON_SIZES:
        out["icon_size"] = cfg["icon_size"]
    if cfg.get("style") in STYLES:
        out["style"] = cfg["style"]
    bars, names = [], set()
    for tb in cfg.get("toolbars", []):
        if not isinstance(tb, dict):
            continue
        name = str(tb.get("name", "")).strip() or "Toolbar"
        base, i = name, 2
        while _object_name(name) in names:
            name, i = f"{base} {i}", i + 1
        names.add(_object_name(name))
        items = [x for x in tb.get("items", []) if x in (SEPARATOR, SPACER) or x in known_ids]
        bars.append({"name": name, "visible": bool(tb.get("visible", True)), "items": items})
    if "toolbars" in cfg:
        out["toolbars"] = bars
    return out


class ToolbarManager:
    """Builds the main window's toolbars from the configuration."""

    def __init__(self, window, registry: ActionRegistry, settings=None):
        self.window = window
        self.registry = registry
        self.settings = settings
        self.toolbars: list[QToolBar] = []
        self.config = self.load()

    def known_ids(self):
        return {i for i, _, _ in self.registry.all_items()}

    def load(self) -> dict:
        raw = self.settings.value(SETTINGS_KEY, "") if self.settings else ""
        try:
            cfg = json.loads(raw) if raw else DEFAULT_CONFIG
        except (TypeError, ValueError):
            cfg = DEFAULT_CONFIG
        return validate_config(cfg, self.known_ids())

    def save(self):
        if self.settings is not None:
            self.settings.setValue(SETTINGS_KEY, json.dumps(self.config))

    def set_config(self, cfg: dict):
        self.config = validate_config(cfg, self.known_ids())
        self.save()
        self.apply()

    def apply(self):
        """(Re)create the toolbars, keeping each existing toolbar's dock area."""
        areas = {tb.objectName(): self.window.toolBarArea(tb) for tb in self.toolbars}
        for tb in self.toolbars:
            self.window.removeToolBar(tb)
            tb.deleteLater()
        self.toolbars = []
        size = self.config["icon_size"]
        style = STYLES[self.config["style"]][1]
        for spec in self.config["toolbars"]:
            tb = QToolBar(spec["name"], self.window)
            tb.setObjectName(_object_name(spec["name"]))
            tb.setIconSize(QSize(size, size))
            tb.setToolButtonStyle(style)
            tb.setMovable(True)
            tb.setFloatable(True)
            for item in spec["items"]:
                if item == SEPARATOR:
                    tb.addSeparator()
                elif item == SPACER:
                    spacer = QWidget()
                    spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                    tb.addWidget(spacer)
                else:
                    act = self.registry.item(item)
                    if act is not None:
                        tb.addAction(act)
            self.window.addToolBar(areas.get(tb.objectName(), Qt.ToolBarArea.TopToolBarArea), tb)
            tb.setVisible(spec["visible"])
            tb.visibilityChanged.connect(lambda vis, name=spec["name"]: self._visibility(name, vis))
            self.toolbars.append(tb)

    def _visibility(self, name, visible):
        # Hiding a toolbar from its context menu is remembered too.
        if not self.window.isVisible():
            return
        for spec in self.config["toolbars"]:
            if spec["name"] == name and spec["visible"] != visible:
                spec["visible"] = visible
                self.save()

    def customize(self):
        dlg = CustomizeDialog(self.registry, self.config, self.window)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.set_config(dlg.config)
            dlg.apply_shortcuts()


class CustomizeDialog(QDialog):
    """Edit toolbar contents, order, names, icon size, button style and keyboard shortcuts."""

    def __init__(self, registry: ActionRegistry, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize toolbars and shortcuts")
        self.resize(820, 560)
        self.registry = registry
        self.config = copy.deepcopy(config)
        self._current = 0

        tabs = QTabWidget()
        tabs.addTab(self._build_toolbar_tab(), "Toolbars")
        tabs.addTab(self._build_shortcut_tab(), "Keyboard shortcuts")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel |
                                   QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs, 1)
        lay.addWidget(buttons)
        self._load_toolbar(0)

    # ------------------------------------------------------------ toolbars
    def _build_toolbar_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Toolbar"))
        self.tb_combo = QComboBox()
        self.tb_combo.currentIndexChanged.connect(self._switch_toolbar)
        row.addWidget(self.tb_combo, 1)
        self.tb_visible = QCheckBox("Visible")
        self.tb_visible.toggled.connect(self._set_visible)
        row.addWidget(self.tb_visible)
        for text, fn in (("New…", self.new_toolbar), ("Rename…", self.rename_toolbar), ("Delete", self.delete_toolbar)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

        grid = QGridLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter commands")
        self.filter_edit.textChanged.connect(self._fill_available)
        grid.addWidget(QLabel("Available commands"), 0, 0)
        grid.addWidget(self.filter_edit, 1, 0)
        self.available = QTreeWidget()
        self.available.setHeaderHidden(True)
        self.available.itemDoubleClicked.connect(lambda *_: self.add_item())
        grid.addWidget(self.available, 2, 0)

        mid = QVBoxLayout()
        mid.addStretch(1)
        self._btn_add = QPushButton("Add →")
        self._btn_add.clicked.connect(self.add_item)
        self._btn_remove = QPushButton("← Remove")
        self._btn_remove.clicked.connect(self.remove_item)
        self._btn_up = QPushButton("Move up")
        self._btn_up.clicked.connect(lambda: self.move_item(-1))
        self._btn_down = QPushButton("Move down")
        self._btn_down.clicked.connect(lambda: self.move_item(1))
        sep = QPushButton("Add separator")
        sep.clicked.connect(lambda: self._insert(SEPARATOR))
        spacer = QPushButton("Add spacer")
        spacer.setToolTip("Stretchable gap; pushes the following items to the far end")
        spacer.clicked.connect(lambda: self._insert(SPACER))
        for b in (self._btn_add, self._btn_remove, self._btn_up, self._btn_down, sep, spacer):
            mid.addWidget(b)
        mid.addStretch(1)
        grid.addLayout(mid, 2, 1)

        grid.addWidget(QLabel("On this toolbar (drag to reorder)"), 0, 2, 2, 1)
        self.current = QListWidget()
        self.current.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.current.model().rowsMoved.connect(lambda *_: self._store_items())
        self.current.itemDoubleClicked.connect(lambda *_: self.remove_item())
        grid.addWidget(self.current, 2, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid, 1)

        form = QFormLayout()
        self.size_combo = QComboBox()
        for s in ICON_SIZES:
            self.size_combo.addItem(f"{s} px", s)
        self.size_combo.setCurrentIndex(self.size_combo.findData(self.config["icon_size"]))
        self.size_combo.currentIndexChanged.connect(lambda _: self.config.update(icon_size=self.size_combo.currentData()))
        self.style_combo = QComboBox()
        for key, (label, _) in STYLES.items():
            self.style_combo.addItem(label, key)
        self.style_combo.setCurrentIndex(self.style_combo.findData(self.config["style"]))
        self.style_combo.currentIndexChanged.connect(lambda _: self.config.update(style=self.style_combo.currentData()))
        form.addRow("Icon size (all toolbars)", self.size_combo)
        form.addRow("Buttons show", self.style_combo)
        lay.addLayout(form)
        self._fill_available()
        self._fill_toolbar_combo()
        return w

    def _fill_available(self):
        needle = self.filter_edit.text().strip().lower()
        self.available.clear()
        cats = {}
        for item_id, text, cat in self.registry.all_items():
            if needle and needle not in text.lower() and needle not in cat.lower():
                continue
            parent = cats.get(cat)
            if parent is None:
                parent = cats[cat] = QTreeWidgetItem(self.available, [cat])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            child = QTreeWidgetItem(parent, [text.replace("…", "")])
            child.setData(0, Qt.ItemDataRole.UserRole, item_id)
            act = self.registry.item(item_id)
            if act is not None and not act.icon().isNull():
                child.setIcon(0, act.icon())
        self.available.expandAll()

    def _fill_toolbar_combo(self, select: int | None = None):
        self.tb_combo.blockSignals(True)
        self.tb_combo.clear()
        for tb in self.config["toolbars"]:
            self.tb_combo.addItem(tb["name"])
        self.tb_combo.blockSignals(False)
        has = bool(self.config["toolbars"])
        for w in (self.current, self.tb_visible, self._btn_add, self._btn_remove, self._btn_up, self._btn_down):
            w.setEnabled(has)
        if has:
            idx = min(select if select is not None else self._current, len(self.config["toolbars"]) - 1)
            self.tb_combo.setCurrentIndex(idx)
            self._load_toolbar(idx)
        else:
            self.current.clear()

    def _toolbar(self):
        tbs = self.config["toolbars"]
        return tbs[self._current] if 0 <= self._current < len(tbs) else None

    def _switch_toolbar(self, idx):
        self._store_items()  # keep any drag-reordering of the toolbar being left
        self._load_toolbar(idx)

    def _load_toolbar(self, idx):
        self._current = idx
        tb = self._toolbar()
        self.current.clear()
        if tb is None:
            return
        self.tb_visible.blockSignals(True)
        self.tb_visible.setChecked(tb["visible"])
        self.tb_visible.blockSignals(False)
        for item_id in tb["items"]:
            self.current.addItem(self._list_item(item_id))

    def _list_item(self, item_id) -> QListWidgetItem:
        if item_id == SEPARATOR:
            it = QListWidgetItem("──── separator ────")
        elif item_id == SPACER:
            it = QListWidgetItem("⟷  spacer")
        else:
            text = next((t for i, t, _ in self.registry.all_items() if i == item_id), item_id)
            it = QListWidgetItem(text.replace("…", ""))
            act = self.registry.item(item_id)
            if act is not None:
                it.setIcon(act.icon())
        it.setData(Qt.ItemDataRole.UserRole, item_id)
        return it

    def _store_items(self):
        tb = self._toolbar()
        if tb is not None:
            tb["items"] = [self.current.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.current.count())]

    def _insert(self, item_id):
        if self._toolbar() is None:
            return
        row = self.current.currentRow() + 1 if self.current.currentRow() >= 0 else self.current.count()
        self.current.insertItem(row, self._list_item(item_id))
        self.current.setCurrentRow(row)
        self._store_items()

    def add_item(self):
        sel = self.available.currentItem()
        item_id = sel.data(0, Qt.ItemDataRole.UserRole) if sel else None
        if item_id is None:
            return
        tb = self._toolbar()
        if tb is not None and item_id in tb["items"]:
            self.current.setCurrentRow(tb["items"].index(item_id))
            return
        self._insert(item_id)

    def remove_item(self):
        row = self.current.currentRow()
        if row >= 0:
            self.current.takeItem(row)
            self._store_items()

    def move_item(self, delta):
        row = self.current.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < self.current.count()):
            return
        it = self.current.takeItem(row)
        self.current.insertItem(new, it)
        self.current.setCurrentRow(new)
        self._store_items()

    def _set_visible(self, on):
        tb = self._toolbar()
        if tb is not None:
            tb["visible"] = on

    def _unique_name(self, name, ignore=None):
        taken = {_object_name(t["name"]) for i, t in enumerate(self.config["toolbars"]) if i != ignore}
        return _object_name(name) not in taken

    def new_toolbar(self):
        name, ok = QInputDialog.getText(self, "New toolbar", "Name:")
        name = name.strip()
        if not ok or not name:
            return
        if not self._unique_name(name):
            QMessageBox.warning(self, "New toolbar", f"There is already a toolbar called '{name}'.")
            return
        self.config["toolbars"].append({"name": name, "visible": True, "items": []})
        self._fill_toolbar_combo(len(self.config["toolbars"]) - 1)

    def rename_toolbar(self):
        tb = self._toolbar()
        if tb is None:
            return
        name, ok = QInputDialog.getText(self, "Rename toolbar", "Name:", text=tb["name"])
        name = name.strip()
        if ok and name and self._unique_name(name, self._current):
            tb["name"] = name
            self._fill_toolbar_combo(self._current)

    def delete_toolbar(self):
        if self._toolbar() is not None:
            del self.config["toolbars"][self._current]
            self._fill_toolbar_combo(max(0, self._current - 1))

    # ------------------------------------------------------------ shortcuts
    def _build_shortcut_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Click a shortcut and press the new key combination. Clear removes it."))
        specs = list(self.registry.specs.values())
        self.table = QTableWidget(len(specs), 3)
        self.table.setHorizontalHeaderLabels(["Command", "Shortcut", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._editors: dict[str, QKeySequenceEdit] = {}
        for row, spec in enumerate(specs):
            item = QTableWidgetItem(f"{spec.category}: {spec.text.replace('…', '')}")
            item.setIcon(self.registry[spec.id].icon())
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)
            ed = QKeySequenceEdit(QKeySequence(self.registry.shortcut(spec.id)))
            ed.setClearButtonEnabled(True)
            ed.setMaximumSequenceLength(1)
            self.table.setCellWidget(row, 1, ed)
            reset = QPushButton("Default")
            reset.setToolTip(spec.shortcut or "No default shortcut")
            reset.clicked.connect(lambda _, e=ed, s=spec.shortcut: e.setKeySequence(QKeySequence(s)))
            self.table.setCellWidget(row, 2, reset)
            self._editors[spec.id] = ed
        lay.addWidget(self.table, 1)
        return w

    def shortcuts(self) -> dict[str, str]:
        return {i: ed.keySequence().toString(QKeySequence.SequenceFormat.PortableText) for i, ed in self._editors.items()}

    def _conflicts(self):
        seen, clashes = {}, []
        for action_id, seq in self.shortcuts().items():
            if not seq:
                continue
            if seq in seen:
                clashes.append((seq, self.registry.specs[seen[seq]].text, self.registry.specs[action_id].text))
            seen[seq] = action_id
        return clashes

    def apply_shortcuts(self):
        for action_id, seq in self.shortcuts().items():
            if seq != self.registry.shortcut(action_id):
                self.registry.set_shortcut(action_id, seq)

    # ------------------------------------------------------------ dialog
    def restore_defaults(self):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.size_combo.setCurrentIndex(self.size_combo.findData(self.config["icon_size"]))
        self.style_combo.setCurrentIndex(self.style_combo.findData(self.config["style"]))
        self._fill_toolbar_combo(0)
        for action_id, ed in self._editors.items():
            ed.setKeySequence(QKeySequence(self.registry.default_shortcut(action_id)))

    def _accept(self):
        self._store_items()
        clashes = self._conflicts()
        if clashes:
            text = "\n".join(f"{seq}: {a} / {b}" for seq, a, b in clashes)
            QMessageBox.warning(self, "Shortcut conflict", f"These shortcuts are used twice:\n{text}")
            return
        self.accept()

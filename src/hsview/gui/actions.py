"""Every command of the application as a named action.

Menus and toolbars are built from these ids, which is what lets users put
any command on any toolbar and give it any shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import QWidgetAction

from ..annotations import CLASS_COLORS, CLASS_NAMES
from .icons import icon, swatch_icon


@dataclass(frozen=True)
class ActionSpec:
    id: str
    text: str
    icon: str = ""
    shortcut: str = ""
    tip: str = ""
    category: str = "General"
    checkable: bool = False
    group: str = ""  # actions sharing a group are mutually exclusive


SPECS = [
    # File
    ActionSpec("open_cube", "Open cube…", "open", "Ctrl+O", "Open a hyperspectral cube or image", "File"),
    ActionSpec("open_by_id", "Open by database id…", "", "", "Open a file from the camera image database", "File"),
    ActionSpec("open_reference", "Open reference image…", "image", "", "Load a photo of the scene to show or blend", "File"),
    ActionSpec("add_overlay", "Add overlay…", "layers", "", "Add a segmentation or processed image on top", "File"),
    ActionSpec("load_filter", "Load filter into channel 1…", "filter", "", "Show channel 1 through a spectral filter", "File"),
    ActionSpec("load_annotations", "Load annotations…", "load-annotations", "Ctrl+K", "Load points and regions (.pkl or .json)", "File"),
    ActionSpec("save_annotations", "Save annotations", "save", "Ctrl+S", "Save points and regions", "File"),
    ActionSpec("save_annotations_as", "Save annotations as…", "", "Ctrl+Shift+S", "", "File"),
    ActionSpec("save_spectra", "Save annotation spectra…", "spectra-file", "Ctrl+D", "Pickle of all spectra under the annotations", "File"),
    ActionSpec("export_csv", "Export spectra to CSV…", "table", "", "One column per annotation, one row per band", "File"),
    ActionSpec("quit", "Quit", "quit", "Ctrl+Q", "", "File"),
    # Edit
    ActionSpec("undo", "Undo", "undo", "Ctrl+Z", "Undo the last annotation change", "Edit"),
    ActionSpec("redo", "Redo", "redo", "Ctrl+Shift+Z", "", "Edit"),
    ActionSpec("delete_selected", "Delete selected", "delete", "Del", "Delete the selected annotations", "Edit"),
    ActionSpec("select_all", "Select all annotations", "", "Ctrl+A", "", "Edit"),
    ActionSpec("clear_annotations", "Clear annotations…", "trash", "", "Remove every annotation", "Edit"),
    ActionSpec("customize", "Customize toolbars and shortcuts…", "settings", "", "", "Edit"),
    # Modes
    ActionSpec("mode_navigate", "Navigate", "hand", "N", "Pan by dragging, zoom with the wheel", "Mode", True, "mode"),
    ActionSpec("mode_select", "Select / move", "pointer", "S", "Select, move and delete annotations", "Mode", True, "mode"),
    ActionSpec("mode_point", "Points", "point", "P", "Click to add points; right-click removes", "Mode", True, "mode"),
    ActionSpec("mode_rect", "Rectangles", "rect", "R", "Drag to add rectangles; right-click removes", "Mode", True, "mode"),
    ActionSpec("mode_polygon", "Polygons", "polygon", "G",
               "Click vertices, double-click or Enter to close, Esc cancels", "Mode", True, "mode"),
    # View
    ActionSpec("zoom_in", "Zoom in", "zoom-in", "Ctrl+=", "", "View"),
    ActionSpec("zoom_out", "Zoom out", "zoom-out", "Ctrl+-", "", "View"),
    ActionSpec("zoom_fit", "Fit to window", "zoom-fit", "Ctrl+0", "", "View"),
    ActionSpec("zoom_actual", "Actual pixels", "zoom-actual", "Ctrl+1", "One image pixel per screen pixel", "View"),
    ActionSpec("prev_band", "Previous band", "prev", "[", "", "View"),
    ActionSpec("next_band", "Next band", "next", "]", "", "View"),
    ActionSpec("true_colour", "True colour", "palette", "", "RGB from 620–670, 530–570 and 450–490 nm", "View"),
    ActionSpec("reverse_rgb", "Reverse reference RGB", "swap", "", "Swap red and blue of the reference image", "View"),
    ActionSpec("reset_contrast", "Reset contrast", "contrast", "", "", "View"),
    ActionSpec("toggle_display", "Display panel", "sliders", "", "", "View", True),
    ActionSpec("toggle_spectra", "Spectra panel", "spectrum", "", "", "View", True),
    ActionSpec("toggle_overlays", "Overlays panel", "eye", "", "", "View", True),
    # Plot
    ActionSpec("plot_points", "Plot point spectra by class", "plot-points", "", "", "Plot"),
    ActionSpec("plot_regions", "Plot region spectra by class", "plot-regions", "", "Mean spectrum of each rectangle and polygon", "Plot"),
    # Help
    ActionSpec("about", "About hsview", "info", "", "", "Help"),
] + [
    ActionSpec(f"class_{i}", f"Class {i}: {name}", "", str(i + 1), f"Draw new annotations as {name}", "Class", True, "class")
    for i, name in enumerate(CLASS_NAMES)
]

SEPARATOR = "|"
SPACER = "<spacer>"

# Items that are widgets rather than buttons.
WIDGET_ITEMS = {
    "class_selector": ("Class selector", "Class"),
    "band_readout": ("Band / wavelength readout", "View"),
}


class ActionRegistry:
    """Creates the QActions and remembers user shortcuts."""

    def __init__(self, parent, settings=None):
        self.parent = parent
        self.settings = settings
        self.specs = {s.id: s for s in SPECS}
        self.actions: dict[str, QAction] = {}
        self.widgets: dict[str, QWidgetAction] = {}
        self.groups: dict[str, QActionGroup] = {}
        for spec in SPECS:
            act = QAction(spec.text, parent)
            if spec.id.startswith("class_"):
                act.setIcon(swatch_icon(CLASS_COLORS[int(spec.id.split("_")[1])]))
            elif spec.icon:
                act.setIcon(icon(spec.icon))
            act.setToolTip(spec.text + (f"\n{spec.tip}" if spec.tip else ""))
            act.setStatusTip(spec.tip or spec.text)
            act.setCheckable(spec.checkable)
            if spec.group:
                group = self.groups.setdefault(spec.group, QActionGroup(parent))
                group.addAction(act)
            act.setObjectName(spec.id)
            self.actions[spec.id] = act
        self.load_shortcuts()

    def __getitem__(self, action_id: str) -> QAction:
        return self.actions[action_id]

    def connect(self, action_id: str, slot):
        act = self.actions[action_id]
        (act.toggled if act.isCheckable() and not self.specs[action_id].group else act.triggered).connect(slot)

    def add_widget_item(self, item_id: str, action: QWidgetAction):
        action.setObjectName(item_id)
        action.setText(WIDGET_ITEMS[item_id][0])
        self.widgets[item_id] = action

    def item(self, item_id: str):
        return self.actions.get(item_id) or self.widgets.get(item_id)

    def all_items(self) -> list[tuple[str, str, str]]:
        """(id, text, category) of everything that can go on a toolbar."""
        out = [(s.id, s.text, s.category) for s in SPECS]
        out += [(k, text, cat) for k, (text, cat) in WIDGET_ITEMS.items()]
        return out

    # ---- shortcuts
    def default_shortcut(self, action_id: str) -> str:
        return self.specs[action_id].shortcut

    def shortcut(self, action_id: str) -> str:
        return self.actions[action_id].shortcut().toString(QKeySequence.SequenceFormat.PortableText)

    def set_shortcut(self, action_id: str, seq: str):
        self.actions[action_id].setShortcut(QKeySequence(seq))
        if self.settings is not None:
            key = f"shortcuts/{action_id}"
            if seq == self.default_shortcut(action_id):
                self.settings.remove(key)
            else:
                self.settings.setValue(key, seq)

    def load_shortcuts(self):
        for action_id, spec in self.specs.items():
            seq = spec.shortcut
            if self.settings is not None:
                seq = self.settings.value(f"shortcuts/{action_id}", seq)
            self.actions[action_id].setShortcut(QKeySequence(seq))

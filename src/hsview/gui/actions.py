"""Every command of the application as a named action.

Menus and toolbars are built from these ids, which is what lets users put
any command on any toolbar and give it any shortcut. The command table itself
is in :mod:`hsview.ui_spec`, shared with the browser version.
"""
from __future__ import annotations

from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import QWidgetAction

from ..annotations import CLASS_COLORS
from ..ui_spec import SEPARATOR, SPACER, WIDGET_ITEMS, ActionSpec, build_specs  # noqa: F401 (re-exported)
from .analysis import ANALYSES
from .icons import icon, swatch_icon


SPECS = build_specs(ANALYSES)


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
            if spec.category.startswith("Channel "):
                # Toolbar buttons show the icon text; say which channel they belong to.
                act.setIconText(f"Ch{spec.category.split()[1]} {spec.text.replace('…', '')}")
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

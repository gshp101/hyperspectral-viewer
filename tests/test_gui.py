import json
import pickle

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QSettings, Qt

from hsview.demo import make_demo_cube
from hsview.io import Cube


@pytest.fixture
def win(qtbot, tmp_path):
    from hsview.gui.main_window import MainWindow

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings)
    # No "save annotations?" prompt when the test closes the window.
    qtbot.addWidget(w, before_close_func=lambda w: setattr(w.store, "modified", False))
    w.resize(1200, 800)
    w.show()
    ref, wl, lbl, rgb = make_demo_cube(rows=60, cols=80, n_bands=40)
    w.set_cube(Cube(ref, wl, "nm", name="demo", reference=rgb))
    w.display.update_image()
    w.labels = lbl
    return w


def click(qtbot, view, x, y, button=Qt.MouseButton.LeftButton):
    pos = view.mapFromScene(QPointF(x, y))
    qtbot.mouseClick(view.viewport(), button, pos=pos)


def test_no_opengl_loaded(win):
    import sys

    assert not any(m.startswith(("OpenGL", "PyQt6.QtOpenGL")) for m in sys.modules)


def test_default_display_is_gray_band(win):
    ch = win.display.channels[0]
    assert ch.route.currentText() == "Gray"
    img = win.display.composite()
    assert img.shape == (60, 80) and img.dtype == np.uint8
    assert win.view.has_image


def test_routing_rules(win):
    d = win.display
    c1, c2, c3 = d.channels
    c2.route.setCurrentText("Red")
    assert c1.route.currentText() == "None"  # Gray is exclusive
    c3.route.setCurrentText("Red")
    assert c2.route.currentText() == "None"  # one source per colour
    c3.route.setCurrentText("None")
    assert c1.route.currentText() == "Gray"  # never nothing
    d.reference.route.setCurrentText("Show")
    assert all(c.route.currentText() == "None" for c in d.channels)
    np.testing.assert_array_equal(d.composite(), d.reference.img)


def test_rgb_and_true_colour(win):
    win.display.true_colour()
    img = win.display.composite()
    assert img.shape == (60, 80, 3)
    assert [c.route.currentText() for c in win.display.channels] == ["Red", "Green", "Blue"]


def test_filter_channel(win, tmp_path, monkeypatch):
    n = win.cube.n_bands
    win.display.channels[1].set_filter(np.eye(n)[:, :5], "f.pkl")
    win.display.channels[1].route.setCurrentText("Gray")
    assert win.display.composite().shape == (60, 80)


def test_contrast_changes_display(win):
    win.display.update_image()
    before = win.display._last_composite.copy()
    shown = []
    win.display.imageChanged.connect(shown.append)
    win.display.contrast.setValue(70)
    assert shown and not np.array_equal(shown[-1], before)


def test_draw_point_rect_polygon_and_delete(win, qtbot):
    v = win.view
    v.zoom_fit()
    win.set_class(2)
    v.set_mode("point")
    click(qtbot, v, 10.5, 20.5)
    a = list(win.store)[0]
    assert a.kind == "point" and a.cls == 2 and tuple(a.coords[0]) == (10, 20)

    v.set_mode("rect")
    p0, p1 = v.mapFromScene(QPointF(30, 5)), v.mapFromScene(QPointF(50, 25))
    qtbot.mousePress(v.viewport(), Qt.MouseButton.LeftButton, pos=p0)
    qtbot.mouseMove(v.viewport(), p1)
    qtbot.mouseRelease(v.viewport(), Qt.MouseButton.LeftButton, pos=p1)
    assert [x.kind for x in win.store] == ["point", "rect"]

    v.set_mode("polygon")
    for x, y in ((5, 40), (25, 40), (15, 55)):
        click(qtbot, v, x, y)
    qtbot.keyClick(v, Qt.Key.Key_Return)
    assert [x.kind for x in win.store] == ["point", "rect", "polygon"]

    # Right-click on the rectangle's edge removes it.
    v.set_mode("rect")
    click(qtbot, v, 30.2, 15, Qt.MouseButton.RightButton)
    assert [x.kind for x in win.store] == ["point", "polygon"]
    win.store.undo()
    assert len(win.store) == 3

    # Select mode: select all, recolour, delete.
    win.registry["select_all"].trigger()
    assert len(v.selected_ids()) == 3
    win.set_class(5)
    assert {x.cls for x in win.store} == {5}
    win.registry["delete_selected"].trigger()
    assert len(win.store) == 0


def test_spectrum_panel_lines(win):
    win.store.add("point", 0, [[5, 5]])
    win.store.add("rect", 1, [[30, 5], [50, 25]])
    win.overlays.add_array(win.labels, "seg")
    win.spectra.show_labels.setChecked(True)
    win.spectra.redraw()
    labels = [t.get_text() for t in win.spectra.ax.get_legend().get_texts()]
    assert any(l.startswith("Pt") for l in labels) and any(l.startswith("Rect") for l in labels)
    assert sum(l.startswith("seg") for l in labels) == len(np.unique(win.labels)) - 1
    win.spectra.by_class.setChecked(True)
    win.spectra.redraw()
    labels = [t.get_text() for t in win.spectra.ax.get_legend().get_texts()]
    assert any(l.startswith("Red") for l in labels) and any(l.startswith("Green") for l in labels)


def test_plot_by_class(win):
    win.store.add("point", 0, [[5, 5]])
    win.store.add("point", 3, [[6, 5]])
    win.store.add("point", 3, [[7, 5]])
    (dlg,) = win.plot_by_class("point")
    assert len([ax for ax in dlg.figure.axes if ax.axison]) == 2
    dlg.close()


def test_plot_each_class_in_its_own_window(win):
    win.store.add("point", 0, [[5, 5]])
    win.store.add("point", 3, [[6, 5]])
    win.store.add("point", 3, [[7, 5]])
    win.store.add("rect", 1, [[30, 5], [50, 25]])
    win.registry["plot_separate"].setChecked(True)
    assert win.settings.value("plot/separate_windows", type=bool)
    dialogs = win.plot_by_class("point")
    assert [d.windowTitle().split(" — ")[1] for d in dialogs] == ["Red", "Yellow"]
    yellow = dialogs[1].figure.axes[0]
    assert len(yellow.lines) == 2  # both yellow points, nothing else
    win.close_plot_windows()
    assert all(not d.isVisible() for d in dialogs)


def test_save_and_load_annotations(win, tmp_path):
    win.store.add("point", 1, [[3, 4]])
    path = tmp_path / "a.pkl"
    assert win._write_annotations(str(path))
    assert pickle.loads(path.read_bytes())["points"] == [(3.0, 4.0, 1)]
    win.store.clear()
    assert win.load_annotations(str(path)) and len(win.store) == 1


def test_overlays(win):
    ov = win.overlays.add_array(win.labels, "seg")
    assert ov.as_labels and win.overlays.label_overlay() is ov
    img = win.overlays.add_array(np.random.rand(60, 80), "score")
    assert not img.as_labels
    win.overlays.remove_current()
    assert list(win.overlays.overlays.values()) == [ov]


def test_toolbar_config_persists_and_validates(win):
    from hsview.gui.toolbar import DEFAULT_CONFIG, SEPARATOR, SPACER

    tm = win.toolbars
    cfg = {"icon_size": 32, "style": "under", "toolbars": [
        {"name": "Mine", "visible": True, "items": ["zoom_in", SEPARATOR, "bogus", SPACER, "class_selector"]},
        {"name": "Mine", "visible": False, "items": ["undo"]},
    ]}
    tm.set_config(cfg)
    assert [tb.windowTitle() for tb in tm.toolbars] == ["Mine", "Mine 2"]
    assert tm.config["toolbars"][0]["items"] == ["zoom_in", SEPARATOR, SPACER, "class_selector"]
    assert tm.toolbars[0].iconSize().width() == 32
    assert tm.toolbars[0].toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextUnderIcon
    saved = json.loads(win.settings.value("toolbars/config"))
    assert saved["toolbars"][1]["name"] == "Mine 2" and saved["toolbars"][1]["visible"] is False
    tm.set_config(DEFAULT_CONFIG)
    assert len(tm.toolbars) == 3


def test_customize_dialog_edits(win, qtbot):
    from hsview.gui.toolbar import CustomizeDialog

    dlg = CustomizeDialog(win.registry, win.toolbars.config, win)
    qtbot.addWidget(dlg)
    n = dlg.current.count()
    dlg.current.setCurrentRow(0)
    dlg.remove_item()
    assert dlg.current.count() == n - 1
    # find "About" in the available tree and add it
    tree = dlg.available
    target = None
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            if top.child(j).data(0, Qt.ItemDataRole.UserRole) == "about":
                target = top.child(j)
    tree.setCurrentItem(target)
    dlg.add_item()
    dlg.move_item(-1)
    assert "about" in dlg.config["toolbars"][0]["items"]
    dlg.config["toolbars"].append({"name": "Extra", "visible": True, "items": []})
    dlg._fill_toolbar_combo(len(dlg.config["toolbars"]) - 1)
    dlg._insert("zoom_fit")
    dlg._editors["zoom_fit"].setKeySequence("Ctrl+F")
    dlg._accept()
    assert dlg.result() == dlg.DialogCode.Accepted
    win.toolbars.set_config(dlg.config)
    dlg.apply_shortcuts()
    assert [tb.windowTitle() for tb in win.toolbars.toolbars][-1] == "Extra"
    assert win.registry.shortcut("zoom_fit") == "Ctrl+F"
    assert win.settings.value("shortcuts/zoom_fit") == "Ctrl+F"


def test_shortcut_conflict_blocks_accept(win, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from hsview.gui.toolbar import CustomizeDialog

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dlg = CustomizeDialog(win.registry, win.toolbars.config, win)
    qtbot.addWidget(dlg)
    dlg._editors["zoom_fit"].setKeySequence("Ctrl+O")  # already Open cube
    dlg._accept()
    assert warned and dlg.result() != dlg.DialogCode.Accepted


def test_icons_render(qapp):
    from hsview.gui.icons import ICON_NAMES, icon

    for name in ICON_NAMES:
        assert not icon(name).pixmap(24, 24).isNull(), name


def test_app_icon_choice_persists(win):
    from hsview.gui.app_icons import APP_ICONS, render_app_icon

    for key in APP_ICONS:
        assert not render_app_icon(key, 32).isNull()
    win.set_app_icon("prism")
    assert win.settings.value("app/icon") == "prism"
    assert not win.windowIcon().isNull()
    win.set_app_icon("nonsense")
    assert win.settings.value("app/icon") == "scene-cube"

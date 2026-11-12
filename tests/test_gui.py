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
    assert len(tm.toolbars) == len(DEFAULT_CONFIG["toolbars"])


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
    assert win.settings.value("app/icon") == "scene-cube-light"


def test_band_window_averages_centred_bands(win):
    from hsview.gui.display_panel import band_window

    assert list(band_window(10, 5, 40)) == [8, 9, 10, 11, 12]
    assert list(band_window(10, 4, 40)) == [9, 10, 11, 12]
    assert list(band_window(0, 5, 40)) == [0, 1, 2, 3, 4]  # shifted to stay inside the cube
    assert list(band_window(39, 5, 40)) == [35, 36, 37, 38, 39]
    ch = win.display.channels[0]
    ch.set_band(20)
    assert np.array_equal(ch.image(), win.cube.data[:, :, 20])
    ch.set_width(5)
    assert np.allclose(ch.image(), win.cube.data[:, :, 18:23].mean(axis=2))
    assert "mean of 5 bands 18–22" in ch.info.text()
    ch.spin.setValue(30)  # moving the centre keeps the width
    assert np.allclose(ch.image(), win.cube.data[:, :, 28:33].mean(axis=2))


def test_reference_overlay_blends_over_bands(win):
    d = win.display
    d.update_image()
    bands = d.composite()
    ref = d.reference
    ref.route.setCurrentText("Overlay")
    assert d.channels[0].route.currentText() == "Gray"  # overlay keeps the band display
    ref.opacity.setValue(0)
    assert np.array_equal(d.composite()[:, :, 0], bands)
    ref.opacity.setValue(100)
    assert np.array_equal(d.composite(), ref.img[:, :, :3])
    ref.opacity.setValue(50)
    mid = d.composite()
    assert mid.shape == ref.img.shape[:2] + (3,)
    d.channels[1].route.setCurrentText("Red")  # choosing colours keeps the overlay
    assert ref.route.currentText() == "Overlay"
    ref.route.setCurrentText("Show")
    assert all(c.route.currentText() == "None" for c in d.channels)


def _menu_action_ids(menu, out):
    for act in menu.actions():
        if act.menu() is not None:
            _menu_action_ids(act.menu(), out)
        elif act.objectName():
            out.add(act.objectName())
    return out


def test_every_command_is_in_a_menu(win):
    found = set()
    for act in win.menuBar().actions():
        _menu_action_ids(act.menu(), found)
    missing = set(win.registry.actions) - found
    assert not missing, missing


def test_channel_route_and_panel_checkboxes_follow_menus(win):
    r = win.registry
    ch = win.display.channels[1]
    r["ch1_route_red"].trigger()
    assert ch.route.currentText() == "Red"
    ch.route.setCurrentText("Blue")
    assert r["ch1_route_blue"].isChecked()
    r["scaling_percentile"].trigger()
    assert win.display.scaling.currentData() == "percentile"
    r["ref_route_show"].trigger()
    assert win.display.reference.route.currentText() == "Show"
    assert r["spectra_cursor"].isChecked()
    r["spectra_cursor"].setChecked(False)
    assert not win.spectra.follow_cursor.isChecked()
    win.spectra.by_class.setChecked(True)
    assert r["spectra_by_class"].isChecked()
    before = win.display.contrast.value()
    r["contrast_up"].trigger()
    assert win.display.contrast.value() > before


def test_overlay_menu(win):
    win.overlays.add_array(win.labels, "seg")
    menu = win.menus["Overlays"]
    menu.aboutToShow.emit()
    sub = next(a.menu() for a in menu.actions() if a.menu() is not None and a.text().startswith("seg"))
    vis = next(a for a in sub.actions() if a.text() == "Visible")
    vis.setChecked(False)
    ov = next(iter(win.overlays.overlays.values()))
    assert not ov.visible
    win.overlays.set_opacity(ov.key, 0.2)
    assert ov.opacity == pytest.approx(0.2)
    win.registry["show_overlays"].trigger()
    assert ov.visible


def test_toolbars_editable_from_menu(win):
    tm = win.toolbars
    win.toolbar_menu.aboutToShow.emit()  # builds without errors
    assert win.toolbar_menu.actions()
    tm.new_toolbar("Mine")
    bar = len(tm.config["toolbars"]) - 1
    tm.set_item(bar, "zoom_in", True)
    tm.set_item(bar, "analysis_svd_background", True)
    tm.append_item(bar, "|")
    tm.set_item(bar, "undo", True)
    tm.move_at(bar, 3, -3)
    assert tm.config["toolbars"][bar]["items"] == ["undo", "zoom_in", "analysis_svd_background", "|"]
    tm.remove_at(bar, 3)
    tm.set_item(bar, "zoom_in", False)
    tm.rename_toolbar(bar, "Renamed")
    tm.set_visible(bar, False)
    tm.set_icon_size(32)
    tm.set_style("beside")
    saved = json.loads(win.settings.value("toolbars/config"))
    assert saved["toolbars"][bar] == {"name": "Renamed", "visible": False,
                                      "items": ["undo", "analysis_svd_background"]}
    assert saved["icon_size"] == 32 and saved["style"] == "beside"
    assert not tm.toolbars[bar].isVisible()
    # Checking a command in the menu puts it on the toolbar.
    win.toolbar_menu.aboutToShow.emit()
    sub = next(a.menu() for a in win.toolbar_menu.actions() if a.text() == "Renamed")
    commands = next(a.menu() for a in sub.actions() if a.text() == "Commands")
    view = next(a.menu() for a in commands.actions() if a.text() == "View")
    next(a for a in view.actions() if a.text() == "Zoom in").setChecked(True)
    assert tm.config["toolbars"][bar]["items"][-1] == "zoom_in"
    tm.delete_toolbar(bar)
    assert "Renamed" not in [tb.windowTitle() for tb in tm.toolbars]


def test_svd_background_analysis(win, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from hsview.gui import analysis

    win.store.add("point", 0, [[10, 10]])
    original = win.cube
    seen = {}

    def fake_exec(dlg):
        seen["max_k"] = dlg.k.maximum()
        dlg.k.setValue(3)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(analysis.SVDBackgroundDialog, "exec", fake_exec)
    win.registry["analysis_svd_background"].trigger()
    assert win.cube is not original and win.cube.shape == original.shape
    assert seen["max_k"] == original.n_bands - 1
    assert win.cube.metadata["analysis"][-1]["k"] == 3
    np.testing.assert_allclose(np.asarray(win.cube.data).reshape(-1, original.n_bands).mean(axis=0), 0, atol=1e-3)
    assert len(win.store) == 1  # annotations survive
    assert win.registry["analysis_undo"].isEnabled()
    assert win.settings.value("analysis/svd_k", type=int) == 3
    win.registry["analysis_svd_background"].trigger()
    assert len(win.cube_history) == 2
    win.registry["analysis_undo"].trigger()
    assert len(win.cube_history) == 1
    win.registry["analysis_restore"].trigger()
    assert win.cube is original and not win.registry["analysis_undo"].isEnabled()


def test_svd_background_cancel_keeps_cube(win, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from hsview.gui import analysis

    monkeypatch.setattr(analysis.SVDBackgroundDialog, "exec", lambda dlg: QDialog.DialogCode.Rejected)
    original = win.cube
    win.registry["analysis_svd_background"].trigger()
    assert win.cube is original and not win.cube_history


def test_svd_dialog_summary(qapp):
    from hsview.analysis import SVDBackground
    from hsview.gui.analysis import SVDBackgroundDialog

    model = SVDBackground(np.zeros(4), np.array([3.0, 2.0, 1.0, 0.0]), np.eye(4))
    dlg = SVDBackgroundDialog(model, k=1)
    assert "64.29 %" in dlg.summary.text()
    dlg.k.setValue(2)
    assert "92.86 %" in dlg.summary.text()
    dlg.deleteLater()


def test_segment_menu_and_toolbar(win):
    titles = [m.title().replace("&", "") for m in [a.menu() for a in win.menuBar().actions()] if m]
    assert titles.index("Segment") == titles.index("Analysis") + 1
    ids = [a.objectName() for a in win.menus["Segment"].actions() if a.objectName()]
    assert ids == ["segment_unsupervised", "segment_supervised", "save_segmentation", "open_segmentation"]
    assert {"save_cube", "save_segmentation"} <= _menu_action_ids(win.menus["File"], set())
    from hsview.gui.toolbar import DEFAULT_CONFIG

    items = [i for tb in DEFAULT_CONFIG["toolbars"] for i in tb["items"]]
    assert {"save_cube", "segment_unsupervised", "segment_supervised", "save_segmentation"} <= set(items)
    r = win.registry
    assert r["segment_unsupervised"].isEnabled()
    assert not r["segment_supervised"].isEnabled()  # no annotations yet
    assert not r["save_segmentation"].isEnabled()  # nothing to save yet
    win.store.add("point", 0, [[5, 5]])
    assert r["segment_supervised"].isEnabled()


def test_unsupervised_segmentation_overlay_and_save(win, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QDialog, QFileDialog

    from hsview.analysis import read_segmentation
    from hsview.gui import segmentation

    def fake_exec(dlg):
        dlg.clusters.setValue(4)
        dlg.components.setValue(5)
        dlg.intensity.setCurrentIndex(dlg.intensity.findData("ignore"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(segmentation.UnsupervisedDialog, "exec", fake_exec)
    cube_before = win.cube
    win.registry["segment_unsupervised"].trigger()
    assert win.cube is cube_before  # segmentation does not replace the cube
    ov = win.overlays.current()
    assert ov.as_labels and ov.segmentation is not None and ov.name == "k-means (4 clusters)"
    assert set(np.unique(ov.data)) == {1, 2, 3, 4}
    assert win.settings.value("segment_unsupervised/clusters", type=int) == 4
    assert win.settings.value("segment_unsupervised/intensity") == "ignore"
    assert ov.segmentation.params["intensity"] == "ignore"
    assert win.registry["save_segmentation"].isEnabled()
    # A second run hides the first.
    win.registry["segment_unsupervised"].trigger()
    assert not ov.visible and win.overlays.current().visible

    path = tmp_path / "seg.npz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    win.registry["save_segmentation"].trigger()
    d = read_segmentation(path)
    np.testing.assert_array_equal(d["labels"], win.overlays.current().data)
    assert d["cube"]["name"] == "demo" and not d["supervised"]


def test_supervised_segmentation_uses_class_colours(win, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QDialog, QFileDialog

    from hsview.analysis import read_segmentation
    from hsview.gui import segmentation

    lbl = win.labels  # 60 x 80 demo: 0 background, 1 a disc
    for cls, value in ((0, 0), (1, 1)):
        rr, cc = np.nonzero(lbl == value)
        for r, c in list(zip(rr, cc))[:: max(1, len(rr) // 6)][:6]:
            win.store.add("point", cls, [[c + 0.5, r + 0.5]])
    win.store.add("rect", 0, [[0, 0], [4, 4]])
    seen = {}

    def fake_exec(dlg):
        seen["classes"] = dlg.n_classes
        seen["ok"] = dlg.ok_button.isEnabled()
        dlg.method.setCurrentIndex(dlg.method.findData("lda"))
        dlg.intensity.setCurrentIndex(dlg.intensity.findData("ignore_offset"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(segmentation.SupervisedDialog, "exec", fake_exec)
    win.registry["segment_supervised"].trigger()
    assert seen == {"classes": 2, "ok": True}
    ov = win.overlays.current()
    assert ov.segmentation.supervised and ov.segmentation.method == "lda"
    assert ov.segmentation.params["intensity"] == "ignore_offset"
    assert (ov.data == lbl + 1).mean() > 0.95
    assert ov.label_color(1) == (1.0, 0.0, 0.0) and ov.label_color(2) == (0.0, 1.0, 0.0)
    # The overlay is drawn in the annotation class colours.
    from hsview.gui.overlay_panel import colorize

    rgba = colorize(ov.data, True, ov.colors)
    r, c = np.argwhere(ov.data == 2)[0]
    assert tuple(rgba[r, c, :3]) == (0, 255, 0)

    path = tmp_path / "seg.pkl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    win.registry["save_segmentation"].trigger()
    d = read_segmentation(path)
    assert len(d["annotations"]) == len(win.store) and d["supervised"]

    # Open it again: same colours; load its annotations: the segmentation file is not overwritten later.
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    win.registry["open_segmentation"].trigger()
    again = win.overlays.current()
    assert again is not ov and again.colors == ov.colors
    assert win.load_annotations(path) and win.annotation_path is None


def test_supervised_dialog_needs_two_classes(win, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from hsview.gui import segmentation

    win.store.add("point", 2, [[5, 5]])
    seen = {}

    def fake_exec(dlg):
        seen["ok"] = dlg.ok_button.isEnabled()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(segmentation.SupervisedDialog, "exec", fake_exec)
    win.registry["segment_supervised"].trigger()
    assert seen == {"ok": False} and not win.overlays.overlays


def test_save_processed_cube_from_menu(win, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QDialog, QFileDialog

    from hsview.gui import analysis
    from hsview.io import load_cube

    monkeypatch.setattr(analysis.SVDBackgroundDialog, "exec", lambda dlg: QDialog.DialogCode.Accepted)
    win.registry["analysis_svd_background"].trigger()
    seen = {}

    def fake_save(parent, title, default, filters):
        seen.update(title=title, default=default, filters=filters)
        return str(tmp_path / "processed.h5"), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)
    win.registry["save_cube"].trigger()
    assert seen["title"] == "Save processed cube" and "*.npz" in seen["filters"]
    assert "[" not in seen["default"]  # the processed name is made file-safe
    back = load_cube(tmp_path / "processed.h5")
    np.testing.assert_allclose(back.data, win.cube.data)
    assert back.metadata["analysis"][-1]["name"] == "svd_background"


def test_open_examples(win, tmp_path, monkeypatch):
    import scipy.io

    from hsview import datasets

    win.open_demo()
    assert win.cube.shape == (120, 160, 100) and len(win.overlays.overlays) == 1
    # A cached download opens without asking; the ground truth comes up as a named label overlay.
    folder = tmp_path / "pavia_university"
    folder.mkdir()
    cube = np.ones((6, 5, 103), np.uint16)
    gt = np.zeros((6, 5), np.uint8)
    gt[0, 0] = 4
    scipy.io.savemat(folder / "PaviaU.mat", {"paviaU": cube})
    scipy.io.savemat(folder / "PaviaU_gt.mat", {"paviaU_gt": gt})
    ds = datasets.PAVIA_UNIVERSITY
    monkeypatch.setattr("hsview.gui.main_window.cached_files",
                        lambda d: {r: folder / f.name for r, f in d.files.items()})
    monkeypatch.setitem(datasets.OPENERS, ds.key, lambda a, b: datasets.open_pavia_university(a, b, check=False))
    assert win.open_dataset(ds)
    assert win.cube.shape == (6, 5, 103)
    (ov,) = win.overlays.overlays.values()
    assert ov.name == "PaviaU ground truth" and ov.label_text(4) == "PaviaU ground truth: Trees"


def _recent_titles(win):
    win._fill_recent_menu()
    return [a.text() for a in win.recent_menu.actions() if not a.isSeparator()]


def test_open_recent(win, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    assert _recent_titles(win) == ["No recent files", "Clear recent files"]
    paths = []
    for name in ("a.pkl", "b.pkl"):
        p = tmp_path / name
        with open(p, "wb") as f:
            pickle.dump({"ref": np.random.rand(6, 5, 4).astype(np.float32)}, f)
        assert win.open_cube(str(p))
        paths.append(str(p.resolve()))
    assert win.recent_files() == paths[::-1]
    assert _recent_titles(win) == ["&1. b.pkl", "&2. a.pkl", "Clear recent files"]
    assert win.registry["clear_recent"].isEnabled()

    # Reopening moves a file to the top; the list survives a restart (same QSettings).
    win.recent_menu.actions()[1].trigger()
    assert win.cube.name == "a" and win.recent_files() == paths
    from hsview.gui.main_window import MainWindow
    assert MainWindow(win.settings).recent_files() == paths

    # A file that has gone is dropped from the list.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    (tmp_path / "b.pkl").unlink()
    assert not win.open_recent(paths[1])
    assert win.recent_files() == [paths[0]]

    win.registry["clear_recent"].trigger()
    assert win.recent_files() == []
    assert _recent_titles(win) == ["No recent files", "Clear recent files"]

"""The browser version: hsview.webapp (the Python side, run here in CPython) and the files the page loads."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hsview.webapp import WebApp, ui_tables

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "hsview"
WEB = ROOT / "web"


@pytest.fixture
def app():
    a = WebApp()
    a.load_demo()
    return a


def test_page_loads_every_qt_free_module():
    listed = json.loads((WEB / "python-files.json").read_text())
    expected = sorted(str(p.relative_to(SRC)).replace("\\", "/") for p in SRC.rglob("*.py")
                      if "gui" not in p.parts and p.name not in ("app.py", "__main__.py"))
    assert sorted(listed) == expected


def test_webapp_does_not_need_qt():
    code = ("import sys, hsview.webapp; "
            "bad = [m for m in sys.modules if m.startswith(('PyQt', 'hsview.gui', 'matplotlib.backends'))]; "
            "print(bad); sys.exit(1 if bad else 0)")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_ui_tables_match_the_qt_tables():
    t = ui_tables()
    json.dumps(t)  # everything must convert to JavaScript
    ids = {s["id"] for s in t["specs"]}
    from hsview.ui_spec import ICON_SVG, build_specs

    def walk(items):
        for e in items:
            if isinstance(e, dict):
                yield from walk(e["items"])
            elif e is not None:
                yield e

    for menu in t["menus"]:
        assert set(walk(menu["items"])) <= ids, menu["title"]
    for bar in t["toolbars"]["toolbars"]:
        assert set(bar["items"]) - {t["separator"], t["spacer"]} <= ids | set(t["widget_items"])
    assert all(s["icon"] in ICON_SVG for s in t["specs"] if s["icon"])
    from hsview.webapp import ANALYSES
    assert [s["id"] for s in t["specs"]] == [s.id for s in build_specs(ANALYSES)]


def test_qt_analyses_match_the_web_list():
    pytest.importorskip("PyQt6")
    from hsview.gui.analysis import ANALYSES as QT
    from hsview.webapp import ANALYSES as WEB_ANALYSES

    assert [(a.id, a.text, a.icon, a.tip) for a in QT] == [(a.id, a.text, a.icon, a.tip) for a in WEB_ANALYSES]


def test_demo_renders(app):
    img = app.render()
    assert (img["width"], img["height"]) == (160, 120)
    assert len(img["rgba"]) == 160 * 120 * 4
    s = app.display_state()
    assert [c["route"] for c in s["channels"]] == ["Gray", "None", "None"]
    assert s["readout"].startswith("Band 50")
    assert app.overlay_list()[0]["labels"] and not app.overlay_list()[0]["visible"]


def test_routing_follows_the_qt_rules(app):
    app.set_channel(1, {"route": "Red"})  # a colour replaces Gray
    assert [c.route for c in app.channels] == ["None", "Red", "None"]
    app.set_channel(2, {"route": "Red"})  # one source per colour
    assert [c.route for c in app.channels] == ["None", "None", "Red"]
    app.set_channel(2, {"route": "None"})  # never nothing
    assert [c.route for c in app.channels] == ["Gray", "None", "None"]
    app.set_reference({"route": "Overlay"})  # overlay blends over the bands
    assert app.channels[0].route == "Gray" and app.reference.route == "Overlay"
    app.set_reference({"route": "Show"})  # show is exclusive
    assert all(c.route == "None" for c in app.channels)
    app.true_colour()
    assert [c.route for c in app.channels] == ["Red", "Green", "Blue"] and app.reference.route == "None"
    assert app.display_state()["channels"][0]["info"] == "sum of 8 bands"


def test_band_width_and_contrast(app):
    s = app.set_channel(0, {"band": 10, "width": 5})
    assert s["channels"][0]["info"].endswith("mean of 5 bands 8–12")
    before = app.render()["rgba"]
    app.step_contrast(10)
    assert app.render()["rgba"] != before
    app.reset_contrast()
    assert app.render()["rgba"] == before


def test_annotations_spectra_and_undo(app):
    app.add_annotation("point", 0, [[45.7, 40.2]])
    rid = app.add_annotation("rect", 1, [[80, 70], [140, 110]])
    app.add_annotation("polygon", 2, [[100, 20], [150, 20], [140, 50]])
    anns = app.annotations()
    assert anns[0]["coords"] == [[45.0, 40.0]]
    series = app.spectra_series(show_std=True)
    assert len(series) == 3 and "lo" in series[1] and series[0]["label"] == "Pt 0 (45, 40)"
    assert len(app.spectra_series(by_class=True)) == 3
    app.move_annotations([rid], 2.4, -1)
    assert app.store.get(rid).coords[0].tolist() == [82.4, 69.0]
    app.undo()
    assert app.store.get(rid).coords[0].tolist() == [80.0, 70.0]
    groups = app.class_groups("region")["groups"]
    assert [g["name"] for g in groups] == ["Green", "Blue"]
    assert app.cursor_spectrum(10.5, 3.2) is not None and app.cursor_spectrum(-1, 3) is None


def test_analysis_history(app):
    fractions = app.svd_fit()
    assert abs(sum(fractions) - 1) < 1e-6
    original = app.cube
    name = app.svd_apply(1, False)
    assert "bg" in name and app.state()["history"] == 1
    app.svd_apply(1, True)
    app.undo_analysis()
    assert app.state()["history"] == 1
    app.restore_original_cube()
    assert app.cube is original and app.state()["history"] == 0


def test_segmentations_and_files(app, tmp_path):
    from hsview import annotations as ann_io
    from hsview.analysis import read_segmentation
    from hsview.io import load_cube

    app.add_annotation("rect", 0, [[30, 25], [55, 50]])
    app.add_annotation("rect", 1, [[85, 75], [130, 105]])
    assert app.training_summary()["n_classes"] == 2
    seg = app.segment_supervised({"method": "nearest_mean", "n_components": 5, "intensity": "use", "smooth": 0})
    assert seg["segmentation"] and seg["visible"]
    km = app.segment_unsupervised({"n_clusters": 4, "n_components": 5, "intensity": "ignore", "smooth": 3})
    overlays = {o["key"]: o for o in app.overlay_list()}
    assert not overlays[seg["key"]]["visible"] and overlays[km["key"]]["visible"]  # only the newest shows
    assert app.segmentation_name() == km["name"]
    app.select_overlay(seg["key"])
    assert app.spectra_series(show_labels=True)[0]["dash"] == "dashed"
    app.save_segmentation(str(tmp_path / "seg.npz"))
    assert read_segmentation(tmp_path / "seg.npz")["supervised"]
    app.save_cube(str(tmp_path / "cube.npz"))
    assert load_cube(tmp_path / "cube.npz").shape == app.cube.shape
    app.save_annotations(str(tmp_path / "a.json"), "a.json")
    assert len(ann_io.load_annotations(tmp_path / "a.json")) == 2 and not app.state()["modified"]
    app.save_spectra(str(tmp_path / "s.pkl"))
    assert app.export_csv(str(tmp_path / "s.csv")) == 2
    # Annotations from a segmentation file are not saved back over it.
    app.clear_annotations()
    assert app.load_annotations(str(tmp_path / "seg.npz"), "seg.npz") == 2
    assert app.state()["annotation_name"] is None
    info = app.add_overlay_file(str(tmp_path / "seg.npz"), "seg.npz")
    assert info["segmentation"] and info["name"] == "seg"
    rgba = app.overlay_rgba(info["key"])
    assert len(rgba["rgba"]) == 160 * 120 * 4
    app.remove_overlay(info["key"])
    assert info["key"] not in {o["key"] for o in app.overlay_list()}


def test_reference_filter_and_errors(app, tmp_path):
    import pickle

    from PIL import Image

    from hsview.io import LoadError

    Image.fromarray(np.zeros((120, 160, 3), np.uint8)).save(tmp_path / "ref.png")
    s = app.load_reference(str(tmp_path / "ref.png"), "ref.png")
    assert s["reference"]["route"] == "Show" and s["reference"]["name"] == "ref.png"
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(tmp_path / "small.png")
    with pytest.raises(LoadError, match="10×10"):
        app.load_reference(str(tmp_path / "small.png"), "small.png")
    with open(tmp_path / "f.pkl", "wb") as f:
        pickle.dump(np.ones((100, 2)), f)
    s = app.load_filter(1, str(tmp_path / "f.pkl"), "f.pkl")
    assert s["channels"][1]["source"] == "Filter" and s["channels"][1]["info"] == "filter 2 component(s)"
    assert app.render() is not None
    with pytest.raises(LoadError):
        app.add_overlay_array(np.zeros((5, 5), int), "wrong size")
    with pytest.raises(LoadError):
        app.set_channel(2, {"source": "Filter"})


def test_database_file(tmp_path):
    import sqlite3

    with sqlite3.connect(tmp_path / "image_database.db") as conn:
        conn.execute("create table images (id integer, name text)")
        conn.execute("insert into images values (7, 'sub/cube7.pkl')")
    assert WebApp.database_file(7, str(tmp_path)) == "sub/cube7.pkl"

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from hsview.demo import make_demo_cube
from hsview.io import Cube


@pytest.fixture
def widget(make_napari_viewer):
    from hsview.widget import HyperspectralWidget

    ref, wl, lbl = make_demo_cube(rows=40, cols=50, n_bands=30)
    cube = Cube(ref, wl, "nm", name="demo", overlays={"lbl": lbl})
    viewer = make_napari_viewer()
    w = HyperspectralWidget(viewer, cube)
    return viewer, w


def test_loads_with_layers(widget):
    viewer, w = widget
    names = [l.name for l in viewer.layers]
    assert names[0] == "band" and names[-2:] == ["regions", "points"] and "lbl" in names


def test_band_views(widget):
    viewer, w = widget
    w.show_single_band(3)
    np.testing.assert_allclose(viewer.layers["band"].data, w.cube.data[:, :, 3])
    w.show_summed_bands("0-2")
    np.testing.assert_allclose(viewer.layers["band"].data, w.cube.data[:, :, :3].sum(2), rtol=1e-5)
    w.show_rgb(["620-670nm", "530-570nm", ""])
    assert viewer.layers["RGB"].data.shape == (40, 50, 3)
    assert not viewer.layers["band"].visible


def test_annotations_get_ids_colors_and_spectra(widget, tmp_path):
    viewer, w = widget
    w.points_layer.add([[5, 5]])
    w.points_layer.add([[10, 20]])
    w.shapes_layer.add_rectangles(np.array([[2, 2], [8, 8]]))
    w.shapes_layer.add_polygons([np.array([[20, 20], [20, 40], [35, 30]])])
    assert list(w.points_layer.features["id"]) == [0, 1]
    assert list(w.shapes_layer.features["id"]) == [2, 3]
    w.show_labels.setChecked(True)
    w.update_spectrum_plot()
    labels = [t.get_text() for t in w.ax.get_legend().get_texts()]
    assert any(l.startswith("Pt 0") for l in labels) and any(l.startswith("Poly 3") for l in labels)
    n_classes = len(np.unique(viewer.layers["lbl"].data)) - 1
    assert n_classes >= 1 and sum(l.startswith("class") for l in labels) == n_classes

    d = w.current_annotations()
    w.load_annotation_dict({"points": [], "shapes": []} | {"format": "hsview-annotations"})
    assert len(w.points_layer.data) == 0
    w.load_annotation_dict(d)
    assert w.current_annotations() == d


def test_overlay_continuous_and_labels(widget):
    viewer, w = widget
    from napari.layers import Image, Labels

    assert isinstance(w.add_overlay(np.random.rand(40, 50), name="score"), Image)
    assert isinstance(w.add_overlay(np.zeros((40, 50), np.uint8), name="seg"), Labels)
    assert [l.name for l in viewer.layers][-2:] == ["regions", "points"]

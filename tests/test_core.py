import pickle
import sqlite3

import numpy as np
import pytest

from hsview import annotations as ann_io
from hsview.annotations import Annotation, AnnotationStore
from hsview.demo import make_demo_cube
from hsview.io import Cube, LoadError, is_label_image, load_cube, load_filter, resolve_image_id, unpack_raw_frame
from hsview.processing import (
    apply_filter, band_image, compose, contrast_bias_lut, parse_band_spec, pixel_spectrum, polygon_pixels, rect_pixels,
)


@pytest.fixture
def demo():
    ref, wl, lbl, rgb = make_demo_cube(rows=40, cols=52, n_bands=30)
    return ref, wl, lbl, rgb


def _pickle(tmp_path, obj, name="c.pkl"):
    p = tmp_path / name
    p.write_bytes(pickle.dumps(obj))
    return p


# ------------------------------------------------------------------ loading
def test_pickle_ref_wavelengths_lbl(tmp_path, demo):
    ref, wl, _, rgb = demo
    cube = load_cube(_pickle(tmp_path, {"ref": ref, "wavelengths": wl, "lbl": rgb}))
    assert cube.shape == ref.shape
    np.testing.assert_allclose(cube.band_axis, wl)
    np.testing.assert_array_equal(cube.reference, rgb)


@pytest.mark.parametrize("extra", [{}, {"wavelengths": None}])
def test_pickle_without_wavelengths_uses_index(tmp_path, demo, extra):
    ref = demo[0]
    cube = load_cube(_pickle(tmp_path, {"ref": ref, **extra}))
    assert cube.wavelengths is None and cube.reference is None
    np.testing.assert_array_equal(cube.band_axis, np.arange(ref.shape[2]))


def test_pickle_wrong_length_wavelengths_fall_back(tmp_path, demo):
    cube = load_cube(_pickle(tmp_path, {"ref": demo[0], "wavelengths": [1, 2, 3]}))
    assert cube.wavelengths is None and "wavelength_warning" in cube.metadata


def test_pickle_hypercube(tmp_path, demo):
    ref, wl, *_ = demo
    cube = load_cube(_pickle(tmp_path, {"hypercube": ref, "wavelengths": wl}))
    np.testing.assert_array_equal(cube.data, ref)


def test_pickle_raw_frame(tmp_path):
    rng = np.random.default_rng(1)
    sensor = rng.integers(0, 4096, size=(8, 12), dtype=np.uint16)
    frame = sensor.view(np.uint8).reshape(8, 24)  # little-endian byte pairs, as the camera stores them
    webcam = np.zeros((4, 6, 3), np.uint8)
    cube = load_cube(_pickle(tmp_path, {"frame": frame, "webcam": webcam}))
    assert cube.shape == (4, 6, 4)
    np.testing.assert_array_equal(cube.data[:, :, 0], sensor[::2, ::2])
    np.testing.assert_array_equal(cube.data[:, :, 1], sensor[1::2, ::2])
    np.testing.assert_array_equal(cube.data[:, :, 2], sensor[::2, 1::2])
    np.testing.assert_array_equal(cube.data[:, :, 3], sensor[1::2, 1::2])
    assert cube.band_names == ("red", "green1", "green2", "blue")
    assert cube.reference is not None
    with pytest.raises(LoadError):
        unpack_raw_frame(np.zeros((7, 24), np.uint8))


def test_pickle_errors(tmp_path):
    with pytest.raises(LoadError, match="hypercube"):
        load_cube(_pickle(tmp_path, {"cube": np.zeros((2, 2, 2))}))
    with pytest.raises(LoadError, match="DataLook"):
        load_cube(_pickle(tmp_path, {"angle": 3}))
    (tmp_path / "bad.pkl").write_bytes(b"not a pickle")
    with pytest.raises(LoadError):
        load_cube(tmp_path / "bad.pkl")


def test_mat_v73_and_v5(tmp_path, demo):
    import hdf5storage
    import scipy.io

    ref, wl, _, rgb = demo
    hdf5storage.savemat(str(tmp_path / "a.mat"), {"ref": ref, "wavelengths": wl, "lbl": rgb},
                        format="7.3", matlab_compatible=True)
    cube = load_cube(tmp_path / "a.mat")
    np.testing.assert_allclose(cube.data, ref)
    np.testing.assert_allclose(cube.wavelengths, wl)
    scipy.io.savemat(tmp_path / "b.mat", {"hypercube": ref})
    np.testing.assert_allclose(load_cube(tmp_path / "b.mat").data, ref)


def test_plain_image_is_its_own_reference(tmp_path, demo):
    from PIL import Image

    rgb = demo[3]
    Image.fromarray(rgb).save(tmp_path / "photo.png")
    cube = load_cube(tmp_path / "photo.png")
    assert cube.shape == rgb.shape and cube.band_names == ("red", "green", "blue")
    np.testing.assert_array_equal(cube.reference, rgb)


def test_npy_band_first_is_transposed(tmp_path, demo):
    ref = demo[0]
    np.save(tmp_path / "c.npy", np.moveaxis(ref, 2, 0))
    np.testing.assert_array_equal(load_cube(tmp_path / "c.npy").data, ref)


def test_envi_roundtrip(tmp_path, demo):
    import spectral.io.envi as envi

    ref, wl, *_ = demo
    envi.save_image(str(tmp_path / "c.hdr"), ref, metadata={"wavelength": list(wl), "wavelength units": "nm"})
    cube = load_cube(tmp_path / "c.hdr")
    np.testing.assert_allclose(cube.data, ref)
    np.testing.assert_allclose(cube.wavelengths, wl)
    assert cube.wavelength_units == "nm"


def test_hdf5(tmp_path, demo):
    import h5py

    ref, wl, *_ = demo
    with h5py.File(tmp_path / "c.h5", "w") as f:
        f["data/cube"] = ref
        f["wavelengths"] = wl
    cube = load_cube(tmp_path / "c.h5")
    np.testing.assert_allclose(cube.data, ref)
    np.testing.assert_allclose(cube.wavelengths, wl)


def test_filter_loading(tmp_path, demo):
    n = demo[0].shape[2]
    assert load_filter(_pickle(tmp_path, np.ones(n)), n).shape == (n,)
    assert load_filter(_pickle(tmp_path, np.eye(n)[:, :3]), n).shape == (n, 3)
    with pytest.raises(LoadError):
        load_filter(_pickle(tmp_path, np.ones(n + 1)), n)


def test_resolve_image_id(tmp_path):
    with sqlite3.connect(tmp_path / "image_database.db") as conn:
        conn.execute("create table images (id int, name text, timestamp text, comment text)")
        conn.execute("insert into images values (7, 'IMG_00007.pkl', '', '')")
    assert resolve_image_id(7, str(tmp_path)) == tmp_path / "IMG_00007.pkl"
    with pytest.raises(LoadError):
        resolve_image_id(8, str(tmp_path))


def test_is_label_image():
    assert is_label_image(np.array([[0, 1], [2, 1]]))
    assert not is_label_image(np.random.rand(5, 5))


# --------------------------------------------------------------- processing
def test_band_spec_indices_and_wavelengths(demo):
    ref, wl, *_ = demo
    cube = Cube(ref, wl, "nm")
    np.testing.assert_array_equal(parse_band_spec("3-5, 9", cube), [3, 4, 5, 9])
    sel = parse_band_spec("500-600nm", cube)
    assert wl[sel].min() >= 500 and wl[sel].max() <= 600
    assert len(parse_band_spec("551nm", cube)) == 1
    for bad in ("99", "x"):
        with pytest.raises(ValueError):
            parse_band_spec(bad, cube)
    with pytest.raises(ValueError):
        parse_band_spec("500nm", Cube(ref))


def test_band_image_and_compose(demo):
    ref = demo[0]
    cube = Cube(ref)
    np.testing.assert_allclose(band_image(cube, [1, 2]), ref[:, :, 1] + ref[:, :, 2], rtol=1e-6)
    gray = compose(gray=band_image(cube, [0]))
    assert gray.dtype == np.uint8 and gray.min() == 0 and gray.max() == 255
    rgb = compose(rgb={0: band_image(cube, [0]), 2: band_image(cube, [5])}, scaling="percentile")
    assert rgb.shape == ref.shape[:2] + (3,) and rgb[..., 1].max() == 0
    photo = demo[3]
    np.testing.assert_array_equal(compose(gray=photo), photo)


def test_contrast_bias_lut():
    np.testing.assert_array_equal(contrast_bias_lut(45, 0), np.arange(256))
    assert contrast_bias_lut(45, 100)[0] == 100 and contrast_bias_lut(60, 0)[200] == 255


def test_filter_projection(demo):
    ref = demo[0]
    cube = Cube(ref)
    n = ref.shape[2]
    w = np.linspace(0, 1, n)
    np.testing.assert_allclose(apply_filter(cube, w), ref.astype(float) @ w, rtol=1e-6)
    full = apply_filter(cube, np.eye(n))  # the whole space holds all the energy
    np.testing.assert_allclose(full, 1, rtol=1e-6)


def test_spectra_legacy_semantics(demo):
    ref = demo[0]
    cube = Cube(ref)
    np.testing.assert_allclose(pixel_spectrum(cube, 5.9, 3.2), ref[3, 5])
    assert pixel_spectrum(cube, -1, 0) is None
    # Rectangles include every pixel index between the corners, inclusive.
    assert rect_pixels(cube, (2.0, 3.0), (5.0, 7.0)).shape == (5 * 4, ref.shape[2])
    assert rect_pixels(cube, (5.0, 7.0), (2.0, 3.0)).shape[0] == 20
    assert rect_pixels(cube, (-5, -5), (1.5, 1.5)).shape[0] == 4
    square = [(10, 10), (20, 10), (20, 20), (10, 20)]
    px = polygon_pixels(cube, square)
    assert len(px) == 100
    np.testing.assert_allclose(px.mean(0), ref[10:20, 10:20].reshape(-1, ref.shape[2]).mean(0), rtol=1e-5)


# -------------------------------------------------------------- annotations
def test_store_undo_redo():
    s = AnnotationStore()
    events = []
    s.subscribe(lambda: events.append(1))
    a = s.add("point", 2, [[1, 2]])
    s.add("rect", 0, [[0, 0], [3, 3]])
    s.update({a.id: {"cls": 4, "coords": [[5, 5]]}})
    assert s.get(a.id).cls == 4
    s.undo()
    assert s.get(a.id).cls == 2
    s.undo()
    assert len(s) == 1
    s.redo()
    s.redo()
    assert len(s) == 2 and s.get(a.id).cls == 4
    s.remove([a.id])
    assert len(s) == 1 and len(events) == 8


def _items():
    return [Annotation("point", 1, [[5, 6]]), Annotation("rect", 2, [[2, 2], [6, 7]]),
            Annotation("polygon", 3, [[10, 10], [20, 10], [15, 18]])]


def test_legacy_pickle_roundtrip(tmp_path):
    ann_io.save_annotations(tmp_path / "a.pkl", _items())
    d = pickle.loads((tmp_path / "a.pkl").read_bytes())
    # The original viewer's layout: (x, y, cls) points, (pt1, pt2, cls) rects, version tag.
    assert d["annotation_version"] == "1.0" and d["points"] == [(5.0, 6.0, 1)]
    assert len(d["rects"]) == 1 and d["rects"][0][2] == 2
    back = ann_io.load_annotations(tmp_path / "a.pkl")
    assert [(a.kind, a.cls) for a in back] == [("point", 1), ("rect", 2), ("polygon", 3)]
    np.testing.assert_allclose(back[2].coords, _items()[2].coords)


def test_reads_original_viewer_file(tmp_path):
    legacy = {"points": [(3.0, 4.0, 0)], "rects": [(np.array([1.0, 1.0]), np.array([4.0, 5.0]), 6)],
              "annotation_version": "1.0"}
    items = ann_io.load_annotations(_pickle(tmp_path, legacy, "old.pkl"))
    assert [(a.kind, a.cls) for a in items] == [("point", 0), ("rect", 6)]
    with pytest.raises(LoadError):
        ann_io.load_annotations(_pickle(tmp_path, {"points": []}, "bad.pkl"))


def test_json_roundtrip(tmp_path):
    ann_io.save_annotations(tmp_path / "a.json", _items())
    back = ann_io.load_annotations(tmp_path / "a.json")
    assert [(a.kind, a.cls) for a in back] == [("point", 1), ("rect", 2), ("polygon", 3)]


def test_spectra_outputs(tmp_path, demo):
    ref, wl, *_ = demo
    cube = Cube(ref, wl, "nm", name="demo")
    d = ann_io.annotation_spectra_dict(cube, _items())
    assert d["point_spectra"].shape == (1, ref.shape[2])
    np.testing.assert_allclose(d["point_spectra"][0], ref[6, 5])
    assert d["rect_spectra"][0].shape == (5 * 6, ref.shape[2])
    assert len(d["polygon_spectra"]) == 1 and d["annotation_version"] == "1.0"
    assert ann_io.export_spectra_csv(tmp_path / "s.csv", cube, _items()) == 3
    lines = (tmp_path / "s.csv").read_text().splitlines()
    assert len(lines) == ref.shape[2] + 1 and "Rect" in lines[0]

import pickle

import numpy as np
import pytest

from hsview import annotations as ann_io
from hsview.demo import make_demo_cube
from hsview.io import Cube, is_label_image, load_cube
from hsview.processing import band_image, parse_band_spec, pixel_spectrum, region_spectrum, rgb_composite


@pytest.fixture
def demo():
    ref, wl, lbl = make_demo_cube(rows=40, cols=50, n_bands=30)
    return ref, wl, lbl


def _pickle(tmp_path, d):
    p = tmp_path / "c.pkl"
    p.write_bytes(pickle.dumps(d))
    return p


def test_pickle_with_wavelengths_and_labels(tmp_path, demo):
    ref, wl, lbl = demo
    cube = load_cube(_pickle(tmp_path, {"ref": ref, "wavelengths": wl, "lbl": lbl}))
    assert cube.shape == ref.shape
    np.testing.assert_allclose(cube.band_axis, wl)
    np.testing.assert_array_equal(cube.overlays["lbl"], lbl)


@pytest.mark.parametrize("d", [{}, {"wavelengths": None}])
def test_pickle_without_wavelengths_uses_index(tmp_path, demo, d):
    ref = demo[0]
    cube = load_cube(_pickle(tmp_path, {"ref": ref, **d}))
    assert cube.wavelengths is None
    np.testing.assert_array_equal(cube.band_axis, np.arange(ref.shape[2]))
    assert cube.overlays == {}


def test_pickle_requires_ref(tmp_path):
    with pytest.raises(ValueError, match="ref"):
        load_cube(_pickle(tmp_path, {"cube": np.zeros((2, 2, 2))}))


def test_npy_band_first_is_transposed(tmp_path, demo):
    ref = demo[0]
    np.save(tmp_path / "c.npy", np.moveaxis(ref, 2, 0))
    np.testing.assert_array_equal(load_cube(tmp_path / "c.npy").data, ref)


def test_envi_roundtrip(tmp_path, demo):
    import spectral.io.envi as envi

    ref, wl, _ = demo
    envi.save_image(str(tmp_path / "c.hdr"), ref, metadata={"wavelength": list(wl), "wavelength units": "nm"})
    cube = load_cube(tmp_path / "c.hdr")
    np.testing.assert_allclose(cube.data, ref)
    np.testing.assert_allclose(cube.wavelengths, wl)
    assert cube.wavelength_units == "nm"


def test_hdf5(tmp_path, demo):
    import h5py

    ref, wl, _ = demo
    with h5py.File(tmp_path / "c.h5", "w") as f:
        f["data/cube"] = ref
        f["wavelengths"] = wl
    cube = load_cube(tmp_path / "c.h5")
    np.testing.assert_allclose(cube.data, ref)
    np.testing.assert_allclose(cube.wavelengths, wl)


def test_band_spec_indices_and_wavelengths(demo):
    ref, wl, _ = demo
    cube = Cube(ref, wl, "nm")
    np.testing.assert_array_equal(parse_band_spec("3-5, 9", cube), [3, 4, 5, 9])
    sel = parse_band_spec("500-600nm", cube)
    assert wl[sel].min() >= 500 and wl[sel].max() <= 600
    assert len(parse_band_spec("551nm", cube)) == 1
    with pytest.raises(ValueError):
        parse_band_spec("99", cube)
    with pytest.raises(ValueError):
        parse_band_spec("500nm", Cube(ref))


def test_band_image_and_rgb(demo):
    cube = Cube(demo[0])
    np.testing.assert_allclose(band_image(cube, [1, 2]), demo[0][:, :, 1] + demo[0][:, :, 2], rtol=1e-6)
    np.testing.assert_allclose(band_image(cube, [1, 2], "mean"), demo[0][:, :, 1:3].mean(2), rtol=1e-5)
    rgb = rgb_composite([band_image(cube, [0]), None, band_image(cube, [5])])
    assert rgb.shape == demo[0].shape[:2] + (3,)
    assert rgb[..., 1].max() == 0 and 0 <= rgb.min() and rgb.max() <= 1


def test_spectra(demo):
    ref, _, lbl = demo
    cube = Cube(ref)
    np.testing.assert_allclose(pixel_spectrum(cube, 3.2, 4.7), ref[3, 5])
    assert pixel_spectrum(cube, -1, 0) is None
    rect = np.array([[10, 10], [10, 19], [19, 19], [19, 10]])
    mean, std, n = region_spectrum(cube, rect)
    assert n == 100
    np.testing.assert_allclose(mean, ref[10:20, 10:20].reshape(-1, ref.shape[2]).mean(0), rtol=1e-5)
    tri = np.array([[0, 0], [0, 20], [20, 0]])
    assert 150 < region_spectrum(cube, tri)[2] < 260


def test_annotation_roundtrip_and_csv(tmp_path, demo):
    ref, wl, _ = demo
    cube = Cube(ref, wl, "nm", name="demo")
    rect = np.array([[2, 2], [2, 6], [6, 6], [6, 2]], float)
    d = ann_io.annotations_to_dict(np.array([[5.0, 6.0]]), [0], [rect], ["rectangle"], [1], cube)
    ann_io.save_annotations(tmp_path / "a.json", d)
    loaded = ann_io.load_annotations(tmp_path / "a.json")
    assert loaded == d
    assert ann_io.export_spectra_csv(tmp_path / "s.csv", cube, loaded) == 2
    lines = (tmp_path / "s.csv").read_text().splitlines()
    assert len(lines) == ref.shape[2] + 1 and "rectangle_1 (n=25)" in lines[0]


def test_is_label_image():
    assert is_label_image(np.array([[0, 1], [2, 1]]))
    assert not is_label_image(np.random.rand(5, 5))

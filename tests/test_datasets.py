"""Example scenes (hsview.datasets): downloading, checking and opening, without the network."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from hsview import datasets  # noqa: E402
from hsview.datasets import DataFile, Dataset, cached_files, download, open_pavia_university  # noqa: E402
from hsview.io import LoadError, load_overlay  # noqa: E402


def _spec(path: Path, urls) -> DataFile:
    data = path.read_bytes()
    return DataFile(path.name, len(data), hashlib.sha256(data).hexdigest(), tuple(urls))


def _dataset(files) -> Dataset:
    return Dataset("test", "Test", "", "", "", files)


def test_download_tries_the_next_url_and_checks_the_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"right bytes")
    (src / "wrong.bin").write_bytes(b"wrong bytes")
    spec = _spec(src / "a.bin", [(src / "missing.bin").as_uri(), (src / "wrong.bin").as_uri(), (src / "a.bin").as_uri()])
    ds = _dataset({"cube": spec})
    seen = []
    paths = download(ds, tmp_path / "cache", progress=lambda d, t: seen.append((d, t)))
    assert paths["cube"].read_bytes() == b"right bytes"
    assert seen[-1] == (len(b"right bytes"), ds.size)
    assert cached_files(ds, tmp_path / "cache") == paths
    assert not list((tmp_path / "cache").glob("*.part"))


def test_download_reports_every_failure_and_can_be_cancelled(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    spec = DataFile("a.bin", 100, "0" * 64, ((tmp_path / "a.bin").as_uri(),))
    with pytest.raises(LoadError, match="Could not download a.bin"):
        download(_dataset({"cube": spec}), tmp_path / "c1")
    good = _spec(tmp_path / "a.bin", [(tmp_path / "a.bin").as_uri()])
    with pytest.raises(LoadError, match="cancelled"):
        download(_dataset({"cube": good}), tmp_path / "c2", progress=lambda d, t: False)
    assert cached_files(_dataset({"cube": good}), tmp_path / "c2") is None


def test_cache_dir_follows_hsview_data(monkeypatch, tmp_path):
    monkeypatch.setenv("HSVIEW_DATA", str(tmp_path))
    assert datasets.cache_dir() == tmp_path


def test_pavia_entry_points_at_the_publisher_first():
    ds = datasets.DATASETS["pavia_university"]
    assert all(f.urls[0].startswith("https://www.ehu.eus/") for f in ds.files.values())
    assert ds.size == 34806917 + 11005


def test_open_pavia_university_builds_cube_and_named_ground_truth(tmp_path):
    cube = np.arange(6 * 5 * 103, dtype=np.uint16).reshape(6, 5, 103)
    gt = np.zeros((6, 5), np.uint8)
    gt[0, :] = 2
    gt[3:, 1] = 9
    scipy_io.savemat(tmp_path / "PaviaU.mat", {"paviaU": cube})
    scipy_io.savemat(tmp_path / "PaviaU_gt.mat", {"paviaU_gt": gt})
    assert (load_overlay(tmp_path / "PaviaU_gt.mat") == gt).all()
    with pytest.raises(LoadError, match="not the expected file"):
        open_pavia_university(tmp_path / "PaviaU.mat", tmp_path / "PaviaU_gt.mat")
    c, seg = open_pavia_university(tmp_path / "PaviaU.mat", tmp_path / "PaviaU_gt.mat", check=False)
    assert c.shape == (6, 5, 103) and c.name == "PaviaU"
    assert c.wavelengths[0] == 430 and c.wavelengths[-1] == 860 and c.wavelength_units == "nm"
    assert seg.supervised and seg.label_names[2] == "Meadows" and seg.label_names[9] == "Shadows"
    assert seg.pixel_counts == {2: 5, 9: 3}

    from hsview.webapp import WebApp

    app = WebApp()
    datasets.OPENERS["pavia_university"] = lambda a, b: open_pavia_university(a, b, check=False)
    try:
        info = app.open_dataset("pavia_university", {"cube": str(tmp_path / "PaviaU.mat"),
                                                     "ground_truth": str(tmp_path / "PaviaU_gt.mat")})
    finally:
        datasets.OPENERS["pavia_university"] = open_pavia_university
    assert info["bands"] == 103 and "Pavia University" in info["messages"][0]
    labels = [s["label"] for s in app.spectra_series(show_labels=True)]
    assert labels == ["PaviaU ground truth: Meadows", "PaviaU ground truth: Shadows"]

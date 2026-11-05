"""Public example scenes that hsview downloads on first use. No GUI code here.

The files are not part of hsview: they are fetched from their publishers (or a
mirror when that fails), checked against a SHA-256 and cached, so they are only
downloaded once. ``python -m hsview.datasets [name]`` downloads one and prints
where it is.

Pavia University
    ROSIS-03 airborne scene over the University of Pavia, Italy: 610 × 340
    pixels, 103 bands (430–860 nm after removing 12 noisy bands), 1.3 m pixels,
    with a 9-class ground truth. Courtesy of Prof. Paolo Gamba, Telecommunications
    and Remote Sensing Laboratory, University of Pavia, distributed by the
    Computational Intelligence Group of the University of the Basque Country
    (EHU): https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .io import LoadError, load_mat, load_overlay

EHU = "https://www.ehu.eus/ccwintco/uploads"
# A copy of the EHU files (same bytes) in a public GitHub repository, pinned to one commit. Browsers
# can download it (GitHub allows other web sites to fetch its files, which the EHU server may not), and the desktop
# app falls back to it when the EHU server cannot be reached.
PAVIA_MIRROR = ("https://media.githubusercontent.com/media/hxlnwpu/Hyperspectral_image_classification/"
                "bd5beccbca612633a597f9af2f921421f51f6f68/Data/matData")


@dataclass(frozen=True)
class DataFile:
    name: str
    size: int
    sha256: str
    urls: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    description: str
    credit: str
    page: str
    files: dict[str, DataFile]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files.values())


PAVIA_CLASSES = ("Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare soil", "Bitumen",
                 "Self-blocking bricks", "Shadows")
# The colours usually used for this ground truth in papers, so the overlay looks familiar.
PAVIA_COLORS = ((0.75, 0.75, 0.75), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.5, 0.0), (1.0, 0.0, 1.0),
                (0.65, 0.32, 0.17), (0.5, 0.0, 0.5), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))

PAVIA_UNIVERSITY = Dataset(
    key="pavia_university",
    title="Pavia University",
    description="ROSIS airborne scene, 610 × 340 pixels, 103 bands (430–860 nm), with a 9-class ground truth",
    credit="Courtesy of Prof. Paolo Gamba, Telecommunications and Remote Sensing Laboratory, University of Pavia, "
           "via the Computational Intelligence Group, University of the Basque Country (EHU).",
    page="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    files={
        "cube": DataFile("PaviaU.mat", 34806917, "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb",
                         (f"{EHU}/e/ee/PaviaU.mat", f"{PAVIA_MIRROR}/PaviaU.mat")),
        "ground_truth": DataFile("PaviaU_gt.mat", 11005,
                                 "23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829",
                                 (f"{EHU}/5/50/PaviaU_gt.mat", f"{PAVIA_MIRROR}/PaviaU_gt.mat")),
    },
)

DATASETS = {d.key: d for d in (PAVIA_UNIVERSITY,)}


def cache_dir() -> Path:
    """Where downloads are kept: $HSVIEW_DATA, else the user's cache folder."""
    if os.environ.get("HSVIEW_DATA"):
        return Path(os.environ["HSVIEW_DATA"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "hsview" / "datasets"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: str | Path, spec: DataFile) -> None:
    """Raise LoadError unless the file is the one the dataset expects."""
    path = Path(path)
    if path.stat().st_size != spec.size or sha256(path) != spec.sha256:
        raise LoadError(f"{spec.name} is not the expected file (size or checksum differs)")


def cached_files(dataset: Dataset, directory: str | Path | None = None) -> dict[str, Path] | None:
    """The dataset's files if they are all in the cache already, else None."""
    directory = Path(directory) if directory else cache_dir() / dataset.key
    paths = {role: directory / spec.name for role, spec in dataset.files.items()}
    if all(p.exists() and p.stat().st_size == dataset.files[r].size for r, p in paths.items()):
        return paths
    return None


def download(dataset: Dataset, directory: str | Path | None = None,
             progress: Callable[[int, int], bool | None] | None = None) -> dict[str, Path]:
    """Download (or find in the cache) every file of the dataset and return their paths by role.

    ``progress(done_bytes, total_bytes)`` is called as data arrives; returning False cancels
    (LoadError "Download cancelled"). Each file is tried from its URLs in order.
    """
    import urllib.request

    directory = Path(directory) if directory else cache_dir() / dataset.key
    directory.mkdir(parents=True, exist_ok=True)
    total, done, paths = dataset.size, 0, {}
    for role, spec in dataset.files.items():
        path = directory / spec.name
        paths[role] = path
        if path.exists() and path.stat().st_size == spec.size:
            done += spec.size
            continue
        errors = []
        for url in spec.urls:
            part = path.with_name(path.name + ".part")
            got = 0
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "hsview"})
                with urllib.request.urlopen(req, timeout=30) as res, open(part, "wb") as out:
                    while chunk := res.read(1 << 16):
                        out.write(chunk)
                        got += len(chunk)
                        if progress is not None and progress(done + got, total) is False:
                            raise InterruptedError
                check_file(part, spec)
                part.replace(path)
                break
            except InterruptedError:
                part.unlink(missing_ok=True)
                raise LoadError("Download cancelled") from None
            except Exception as exc:  # network errors, HTTP errors, a wrong file: try the next URL
                part.unlink(missing_ok=True)
                errors.append(f"{url}: {exc}")
        else:
            raise LoadError(f"Could not download {spec.name}:\n" + "\n".join(errors))
        done += spec.size
    return paths


def open_pavia_university(cube_path: str | Path, gt_path: str | Path, check: bool = True):
    """Build the Pavia University cube and its ground truth as a Segmentation.

    Returns (cube, segmentation). The bands are given evenly spaced wavelengths over 430–860 nm:
    the published files carry no wavelengths, and this is the range of the retained bands.
    """
    from .analysis.segmentation import segmentation_from_labels

    ds = PAVIA_UNIVERSITY
    if check:
        check_file(cube_path, ds.files["cube"])
        check_file(gt_path, ds.files["ground_truth"])
    cube = load_mat(cube_path)
    cube.name = "PaviaU"
    cube.path = str(cube_path)
    cube.wavelengths = np.linspace(430, 860, cube.n_bands)
    cube.wavelength_units = "nm"
    cube.metadata["dataset"] = ds.title
    cube.metadata["credit"] = ds.credit
    cube.metadata["wavelength_note"] = "Approximate wavelengths: 103 bands spread evenly over 430–860 nm."
    labels = load_overlay(gt_path).astype(np.int32)
    if labels.shape != cube.shape[:2]:
        raise LoadError(f"Ground truth is {labels.shape[1]}×{labels.shape[0]} but the cube is "
                        f"{cube.shape[1]}×{cube.shape[0]}")
    colors = {i + 1: c for i, c in enumerate(PAVIA_COLORS)}
    seg = segmentation_from_labels(labels, cube, "ground truth", colors)
    seg.supervised = True
    seg.label_names = {i + 1: n for i, n in enumerate(PAVIA_CLASSES)}
    return cube, seg


OPENERS = {PAVIA_UNIVERSITY.key: open_pavia_university}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    key = argv[0] if argv else PAVIA_UNIVERSITY.key
    if key not in DATASETS:
        sys.exit(f"Unknown dataset {key!r}; choose from {', '.join(DATASETS)}")
    ds = DATASETS[key]

    def show(done, total):
        print(f"\r{ds.title}: {done / 1e6:5.1f} of {total / 1e6:.1f} MB", end="", file=sys.stderr)

    paths = download(ds, progress=show)
    print(file=sys.stderr)
    for p in paths.values():
        print(p)


if __name__ == "__main__":
    main()

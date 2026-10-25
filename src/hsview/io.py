"""Loading hyperspectral cubes and overlay images.

Every loader returns a :class:`Cube` whose ``data`` is laid out as
``(rows, cols, bands)``. Large files are memory-mapped where the format
allows it, so opening a multi-GB cube is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Cube:
    data: np.ndarray  # (rows, cols, bands), may be a memmap
    wavelengths: np.ndarray | None = None
    wavelength_units: str = ""
    name: str = "cube"
    metadata: dict = field(default_factory=dict)
    # 2-D images that ship with the cube (e.g. a label map); shown as overlays on load.
    overlays: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[2]

    @property
    def band_axis(self) -> np.ndarray:
        """X values for spectra: wavelengths if known, else band indices."""
        if self.wavelengths is not None and len(self.wavelengths) == self.n_bands:
            return self.wavelengths
        return np.arange(self.n_bands, dtype=float)

    @property
    def band_axis_label(self) -> str:
        if self.wavelengths is not None:
            return f"Wavelength ({self.wavelength_units})" if self.wavelength_units else "Wavelength"
        return "Band"


def _to_rows_cols_bands(arr: np.ndarray, band_axis: int | None) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D cube, got shape {arr.shape}")
    if band_axis is None:
        # Heuristic: band axis is the first axis if it is clearly the smallest,
        # otherwise the last (the most common convention).
        band_axis = 0 if arr.shape[0] < min(arr.shape[1], arr.shape[2]) and arr.shape[2] > arr.shape[0] else 2
    return np.moveaxis(arr, band_axis, 2)


def load_envi(path: str | Path, **_) -> Cube:
    import spectral.io.envi as envi

    path = Path(path)
    hdr = path if path.suffix.lower() == ".hdr" else path.with_suffix(".hdr")
    img = envi.open(str(hdr), None if path.suffix.lower() == ".hdr" else str(path))
    data = img.open_memmap(interleave="bip", writable=False)  # (rows, cols, bands)
    meta = dict(img.metadata)
    wl = meta.get("wavelength")
    wavelengths = np.asarray(wl, dtype=float) if wl else None
    units = meta.get("wavelength units", "")
    if units.lower() in ("nanometers", "nanometer"):
        units = "nm"
    return Cube(data, wavelengths, units, path.stem, meta)


def load_npy(path: str | Path, band_axis: int | None = None, wavelengths=None) -> Cube:
    path = Path(path)
    arr = np.load(path, mmap_mode="r")
    if wavelengths is None:
        wl_file = path.with_name(path.stem + "_wavelengths.txt")
        if wl_file.exists():
            wavelengths = np.loadtxt(wl_file)
    return Cube(_to_rows_cols_bands(arr, band_axis), _as_array(wavelengths), "nm" if wavelengths is not None else "", path.stem)


def load_hdf5(path: str | Path, dataset: str | None = None, band_axis: int | None = None) -> Cube:
    import h5py

    path = Path(path)
    f = h5py.File(path, "r")
    if dataset is None:
        candidates = []
        f.visititems(lambda n, o: candidates.append(n) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
        if not candidates:
            raise ValueError(f"No 3-D dataset found in {path}")
        dataset = max(candidates, key=lambda n: f[n].size)
    dset = f[dataset]
    arr = dset[()]  # HDF5 datasets are read into memory; chunked reads would need a dask wrapper
    wavelengths = None
    for key in ("wavelengths", "wavelength", "wl", "bands"):
        if key in dset.attrs:
            wavelengths = np.asarray(dset.attrs[key], dtype=float)
            break
        if key in f and isinstance(f[key], h5py.Dataset) and f[key].ndim == 1:
            wavelengths = f[key][()].astype(float)
            break
    units = dset.attrs.get("wavelength_units", "nm" if wavelengths is not None else "")
    if isinstance(units, bytes):
        units = units.decode()
    f.close()
    return Cube(_to_rows_cols_bands(arr, band_axis), wavelengths, units, f"{path.stem}:{dataset}")


def load_tiff(path: str | Path, band_axis: int | None = None) -> Cube:
    import tifffile

    path = Path(path)
    arr = tifffile.memmap(path) if _tiff_is_memmappable(path) else tifffile.imread(path)
    return Cube(_to_rows_cols_bands(arr, band_axis), None, "", path.stem)


def _tiff_is_memmappable(path: Path) -> bool:
    import tifffile

    try:
        with tifffile.TiffFile(path) as t:
            return t.series[0].dataoffset is not None
    except Exception:
        return False


def load_pickle(path: str | Path, band_axis: int | None = None) -> Cube:
    """Pickled dict: ``ref`` holds the cube, optional ``wavelengths`` and ``lbl`` (label image).

    Only open pickles you trust: unpickling can run arbitrary code.
    """
    import pickle

    path = Path(path)
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "ref" not in d:
        raise ValueError(f"{path.name}: expected a pickled dict with a 'ref' entry")
    wl = d.get("wavelengths")
    wavelengths = None if wl is None else np.asarray(wl, dtype=float).ravel()
    cube = Cube(_to_rows_cols_bands(np.asarray(d["ref"]), band_axis), wavelengths, "", path.stem)
    if wavelengths is not None and len(wavelengths) != cube.n_bands:
        cube.wavelengths = None  # mismatched list: fall back to band index
    lbl = d.get("lbl")
    if lbl is not None:
        cube.overlays["lbl"] = np.squeeze(np.asarray(lbl))
    return cube


def _as_array(wl):
    return None if wl is None else np.asarray(wl, dtype=float)


# Extension -> reader. To support a new format, write a function returning a Cube
# and register its extensions here.
READERS = {
    **dict.fromkeys((".hdr", ".img", ".dat", ".raw", ".bil", ".bip", ".bsq"), load_envi),
    ".npy": load_npy,
    **dict.fromkeys((".h5", ".hdf5", ".he5", ".nc", ".mat"), load_hdf5),
    **dict.fromkeys((".tif", ".tiff"), load_tiff),
    **dict.fromkeys((".pkl", ".pickle", ".p"), load_pickle),
}


def load_cube(path: str | Path, **kwargs) -> Cube:
    """Load a cube, picking the reader from the file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == "" and path.with_suffix(".hdr").exists():
        return load_envi(path)
    if ext not in READERS:
        raise ValueError(f"Unsupported cube format: {path.name}")
    return READERS[ext](path, **kwargs)


def load_overlay(path: str | Path) -> np.ndarray:
    """Load a 2-D processed image (segmentation, classification, score map...)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    elif ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            names = []
            f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) and o.ndim == 2 else None)
            if not names:
                raise ValueError(f"No 2-D dataset found in {path}")
            arr = f[names[0]][()]
    else:
        from skimage.io import imread

        arr = imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr  # RGB(A) image, shown as-is
    return np.squeeze(arr)


def is_label_image(arr: np.ndarray, max_labels: int = 1000) -> bool:
    """Integer images with a modest number of distinct values are treated as segmentations."""
    if arr.ndim != 2 or not (np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool):
        return False
    return len(np.unique(arr[:: max(1, arr.shape[0] // 256), :: max(1, arr.shape[1] // 256)])) <= max_labels

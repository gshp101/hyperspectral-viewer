"""Loading hyperspectral cubes, reference images, overlays and filters.

Every cube loader returns a :class:`Cube` whose ``data`` is laid out as
``(rows, cols, bands)``. Large files are memory-mapped where the format
allows it, so opening a multi-GB cube is cheap.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class LoadError(ValueError):
    """A file could not be interpreted; the message is meant for the user."""


@dataclass
class Cube:
    data: np.ndarray  # (rows, cols, bands), may be a memmap
    wavelengths: np.ndarray | None = None
    wavelength_units: str = ""
    name: str = "cube"
    metadata: dict = field(default_factory=dict)
    # Optional image of the same scene (RGB or gray), e.g. the 'lbl' entry of a pickle.
    reference: np.ndarray | None = None
    # Optional names per band, e.g. ("red", "green1", "green2", "blue") for raw Bayer frames.
    band_names: tuple[str, ...] | None = None
    path: str | None = None

    def __post_init__(self):
        if self.wavelengths is not None:
            self.wavelengths = np.asarray(self.wavelengths, dtype=float).ravel()
            if len(self.wavelengths) != self.n_bands:
                self.metadata["wavelength_warning"] = (
                    f"File has {len(self.wavelengths)} wavelengths for {self.n_bands} bands; using band index instead."
                )
                self.wavelengths = None
        if self.reference is not None and np.asarray(self.reference).shape[:2] != self.shape[:2]:
            self.metadata["reference_warning"] = "Reference image size does not match the cube; ignored."
            self.reference = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[2]

    @property
    def band_axis(self) -> np.ndarray:
        """X values for spectra: wavelengths if known, else band indices."""
        if self.wavelengths is not None:
            return self.wavelengths
        return np.arange(self.n_bands, dtype=float)

    @property
    def band_axis_label(self) -> str:
        if self.wavelengths is not None:
            return f"Wavelength ({self.wavelength_units})" if self.wavelength_units else "Wavelength"
        return "Band"

    def band_label(self, b: int) -> str:
        """Human readable description of one band, e.g. 'Band 12 · 550 nm'."""
        text = f"Band {b}"
        if self.band_names is not None:
            text += f" ({self.band_names[b]})"
        if self.wavelengths is not None:
            text += f" · {self.wavelengths[b]:.4g} {self.wavelength_units}".rstrip()
        return text


def _to_rows_cols_bands(arr: np.ndarray, band_axis: int | None) -> np.ndarray:
    if arr.ndim != 3:
        raise LoadError(f"Expected a 3-D cube, got shape {arr.shape}")
    if band_axis is None:
        # Heuristic: band axis is the first axis if it is clearly the smallest,
        # otherwise the last (the most common convention).
        band_axis = 0 if arr.shape[0] < min(arr.shape[1], arr.shape[2]) and arr.shape[2] > arr.shape[0] else 2
    return np.moveaxis(arr, band_axis, 2)


# ---------------------------------------------------------------- readers
def load_envi(path: str | Path, **_) -> Cube:
    import spectral.io.envi as envi

    path = Path(path)
    hdr = path if path.suffix.lower() == ".hdr" else path.with_suffix(".hdr")
    img = envi.open(str(hdr), None if path.suffix.lower() == ".hdr" else str(path))
    data = img.open_memmap(interleave="bip", writable=False)  # (rows, cols, bands)
    meta = dict(img.metadata)
    wl = meta.get("wavelength")
    units = meta.get("wavelength units", "")
    if units.lower() in ("nanometers", "nanometer"):
        units = "nm"
    names = meta.get("band names")
    return Cube(data, wl if wl else None, units, path.stem, meta,
                band_names=tuple(names) if names and len(names) == data.shape[2] else None)


def load_npy(path: str | Path, band_axis: int | None = None, **_) -> Cube:
    path = Path(path)
    arr = np.load(path, mmap_mode="r")
    wavelengths = None
    wl_file = path.with_name(path.stem + "_wavelengths.txt")
    if wl_file.exists():
        wavelengths = np.loadtxt(wl_file)
    return Cube(_to_rows_cols_bands(arr, band_axis), wavelengths, "nm" if wavelengths is not None else "", path.stem)


def _find_wavelengths(group, dset):
    for key in ("wavelengths", "wavelength", "wl"):
        if key in dset.attrs:
            return np.asarray(dset.attrs[key], dtype=float)
        if key in group and getattr(group[key], "ndim", 0) == 1:
            return group[key][()].astype(float)
    return None


def load_hdf5(path: str | Path, band_axis: int | None = None, dataset: str | None = None, **_) -> Cube:
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        if dataset is None:
            for key in ("hypercube", "ref", "cube"):
                if key in f and getattr(f[key], "ndim", 0) == 3:
                    dataset = key
                    break
        if dataset is None:
            candidates = []
            f.visititems(lambda n, o: candidates.append(n) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
            if not candidates:
                raise LoadError(f"No 3-D dataset found in {path.name}")
            dataset = max(candidates, key=lambda n: f[n].size)
        dset = f[dataset]
        arr = dset[()]
        wavelengths = _find_wavelengths(f, dset)
        units = dset.attrs.get("wavelength_units", "nm" if wavelengths is not None else "")
        reference = f["lbl"][()] if "lbl" in f and f["lbl"].ndim in (2, 3) else None
    if isinstance(units, bytes):
        units = units.decode()
    return Cube(_to_rows_cols_bands(arr, band_axis), wavelengths, units, f"{path.stem}:{dataset}",
                reference=reference)


def load_tiff(path: str | Path, band_axis: int | None = None, **_) -> Cube:
    import tifffile

    path = Path(path)
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    return Cube(_to_rows_cols_bands(arr, band_axis), None, "", path.stem)


def load_image_as_cube(path: str | Path, **_) -> Cube:
    """Plain images (jpg, png, ...) load as a 1- or 3-band cube that is also its own reference."""
    path = Path(path)
    img = read_image(path)
    data = img[..., None] if img.ndim == 2 else img
    names = ("gray",) if img.ndim == 2 else ("red", "green", "blue")
    return Cube(data, None, "", path.stem, reference=img, band_names=names)


def unpack_raw_frame(frame: np.ndarray) -> np.ndarray:
    """Turn a raw camera 'frame' into a 4-band cube of its 2x2 Bayer mosaic.

    The frame stores 16-bit samples as little-endian byte pairs along each row,
    so the 2-D uint8 array is twice as wide as the sensor. Bands are the
    (even row, even col), (odd, even), (even, odd), (odd, odd) sites, i.e.
    red, green, green, blue for an RGGB sensor.
    """
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.shape[0] % 2 or frame.shape[1] % 4:
        raise LoadError("Raw 'frame' must be 2-D with an even number of rows and a width divisible by 4")
    sensor = frame[:, ::2].astype(np.uint16) + frame[:, 1::2].astype(np.uint16) * 256
    cube = np.empty((sensor.shape[0] // 2, sensor.shape[1] // 2, 4), dtype=np.uint16)
    cube[:, :, 0] = sensor[::2, ::2]
    cube[:, :, 1] = sensor[1::2, ::2]
    cube[:, :, 2] = sensor[::2, 1::2]
    cube[:, :, 3] = sensor[1::2, 1::2]
    return cube


def cube_from_dict(d: dict, name: str, band_axis: int | None = None) -> Cube:
    """Interpret the dict layouts written by the camera and processing tools.

    * ``{'hypercube': cube, 'wavelengths': ...}``
    * ``{'ref': cube, 'wavelengths': ..., 'lbl': reference image}``
    * ``{'frame': raw Bayer bytes, 'webcam': reference image}`` (raw camera capture)
    """
    if "angle" in d and not ({"hypercube", "ref", "frame"} & d.keys()):
        raise LoadError("This is a DataLook scan collection. Convert it with DataLook.asHSI() first.")
    band_names = None
    if "hypercube" in d:
        arr, reference = d["hypercube"], d.get("lbl")
    elif "ref" in d:
        arr, reference = d["ref"], d.get("lbl")
    elif "frame" in d:
        arr, reference = unpack_raw_frame(d["frame"]), d.get("webcam")
        band_axis, band_names = 2, ("red", "green1", "green2", "blue")
    else:
        raise LoadError("Expected a 'hypercube', 'ref' or 'frame' entry in the file")
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise LoadError(f"The cube must be a 3-D array, got shape {arr.shape}")
    wl = d.get("wavelengths")
    return Cube(_to_rows_cols_bands(arr, band_axis), None if wl is None else np.asarray(wl, dtype=float), "", name,
                reference=None if reference is None else np.squeeze(np.asarray(reference)), band_names=band_names)


def load_pickle(path: str | Path, band_axis: int | None = None, **_) -> Cube:
    """Pickled dict (see :func:`cube_from_dict`). Only open pickles you trust: unpickling can run code."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as exc:
        raise LoadError(f"Could not read {path.name} as a pickle file: {exc}") from exc
    if isinstance(d, np.ndarray) and d.ndim == 3:
        return Cube(_to_rows_cols_bands(d, band_axis), name=path.stem)
    if not isinstance(d, dict):
        raise LoadError(f"{path.name}: expected a pickled dict")
    return cube_from_dict(d, path.stem, band_axis)


def load_mat(path: str | Path, band_axis: int | None = None, **_) -> Cube:
    """MATLAB files, both v7.3 (HDF5) and older, holding 'hypercube' or 'ref' (+ 'wavelengths', 'lbl')."""
    path = Path(path)
    d = None
    try:
        import hdf5storage

        d = hdf5storage.loadmat(str(path))
    except ImportError:
        pass
    except Exception:
        d = None
    if d is None:
        try:
            import scipy.io

            d = scipy.io.loadmat(str(path), squeeze_me=True)
        except Exception:
            return load_hdf5(path, band_axis)  # last resort: treat as a plain HDF5 file
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    if not ({"hypercube", "ref", "frame"} & d.keys()):
        cubes = [k for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim == 3]
        if not cubes:
            raise LoadError(f"{path.name}: no 'hypercube' or 'ref' variable and no 3-D array")
        d["hypercube"] = d[max(cubes, key=lambda k: d[k].size)]
    for key in ("wavelengths", "lbl"):
        if key in d and isinstance(d[key], np.ndarray) and d[key].size == 0:
            d[key] = None
    return cube_from_dict(d, path.stem, band_axis)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# Extension -> reader. To support a new format, write a function returning a Cube
# and register its extensions here.
READERS = {
    **dict.fromkeys((".hdr", ".img", ".dat", ".raw", ".bil", ".bip", ".bsq"), load_envi),
    ".npy": load_npy,
    **dict.fromkeys((".h5", ".hdf5", ".he5", ".nc"), load_hdf5),
    ".mat": load_mat,
    **dict.fromkeys((".tif", ".tiff"), load_tiff),
    **dict.fromkeys((".pkl", ".pickle", ".p"), load_pickle),
    **dict.fromkeys(IMAGE_EXTENSIONS, load_image_as_cube),
}


def cube_file_filter() -> str:
    exts = " ".join(f"*{e}" for e in sorted(READERS))
    return f"Hyperspectral cubes and images ({exts});;All files (*)"


def load_cube(path: str | Path, **kwargs) -> Cube:
    """Load a cube, picking the reader from the file extension."""
    path = Path(path)
    if not path.exists():
        raise LoadError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == "" and path.with_suffix(".hdr").exists():
        cube = load_envi(path)
    elif ext not in READERS:
        raise LoadError(f"Unsupported file type: {path.name}")
    else:
        cube = READERS[ext](path, **kwargs)
    cube.path = str(path)
    return cube


# ------------------------------------------------------- other inputs
def read_image(path: str | Path) -> np.ndarray:
    """Read an ordinary image file as uint8/uint16 (H, W) or (H, W, 3)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        img = tifffile.imread(path)
    else:
        from PIL import Image

        with Image.open(path) as im:
            if im.mode not in ("L", "RGB", "I;16", "I"):
                im = im.convert("RGB")
            img = np.array(im)
    img = np.squeeze(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim not in (2, 3):
        raise LoadError(f"{path.name} is not a 2-D image")
    return img


def load_overlay(path: str | Path) -> np.ndarray:
    """Load a 2-D processed image (segmentation, classification, score map...)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            names = []
            f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) and o.ndim == 2 else None)
            if not names:
                raise LoadError(f"No 2-D dataset found in {path.name}")
            arr = f[names[0]][()]
    elif ext in (".pkl", ".pickle", ".p"):
        with open(path, "rb") as f:
            arr = pickle.load(f)
        if isinstance(arr, dict):
            arr = next((v for k, v in arr.items() if isinstance(v, np.ndarray) and v.ndim == 2), None)
        if not isinstance(arr, np.ndarray):
            raise LoadError(f"{path.name} does not contain a 2-D array")
    else:
        arr = read_image(path)
    return np.squeeze(np.asarray(arr))


def is_label_image(arr: np.ndarray, max_labels: int = 256) -> bool:
    """Integer images with a modest number of distinct values are treated as segmentations."""
    if arr.ndim != 2 or not (np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool):
        return False
    step0, step1 = max(1, arr.shape[0] // 256), max(1, arr.shape[1] // 256)
    return len(np.unique(arr[::step0, ::step1])) <= max_labels


def load_filter(path: str | Path, n_bands: int) -> np.ndarray:
    """A spectral filter / projection pickle: a (bands,) vector or a (bands, k) matrix."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        raise LoadError(f"Could not read {path.name} as a pickle file: {exc}") from exc
    data = np.asarray(data, dtype=float) if isinstance(data, (np.ndarray, list, tuple)) else None
    if data is None or data.ndim not in (1, 2) or data.shape[0] != n_bands:
        raise LoadError(f"{path.name} is not a filter for this cube: expected a ({n_bands},) or ({n_bands}, k) array")
    return data


# ---------------------------------------------------- image database
DEFAULT_IMAGE_DIR = "/Volumes/HyperspectralData/Images"
DATABASE_NAME = "image_database.db"


def resolve_image_id(image_id: int, image_dir: str | None = None) -> Path:
    """Map a camera database id to a file, like ``databaseUtilities.getFilesById``.

    The directory comes from the argument, then ``$HSVIEW_IMAGE_DIR``, then the
    camera tools' default.
    """
    image_dir = Path(image_dir or os.environ.get("HSVIEW_IMAGE_DIR", DEFAULT_IMAGE_DIR))
    db = image_dir / DATABASE_NAME
    if not db.exists():
        raise LoadError(f"No image database at {db}")
    with sqlite3.connect(db) as conn:
        rows = conn.execute("select name from images where id = ?", (int(image_id),)).fetchall()
    if len(rows) != 1:
        raise LoadError(f"Image id {image_id} {'not found' if not rows else 'is ambiguous'} in {db}")
    return image_dir / rows[0][0]

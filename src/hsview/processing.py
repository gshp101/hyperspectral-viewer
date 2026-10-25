"""Band arithmetic, composites, and spectrum extraction. No GUI code here."""
from __future__ import annotations

import re

import numpy as np

from .io import Cube


def parse_band_spec(spec: str, cube: Cube) -> np.ndarray:
    """Turn a band selection string into sorted unique band indices.

    Accepts comma-separated items, each either a single value or a range:
    ``"12"``, ``"10-20"``, ``"10-20, 35, 40-42"``. Values are band indices,
    unless suffixed with ``nm`` (or the item contains a decimal point and the
    cube has wavelengths), in which case they are wavelengths and map to the
    nearest bands: ``"450-500nm"``, ``"550nm"``.
    """
    indices: list[int] = []
    wl = cube.wavelengths
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        use_wl = item.endswith("nm") or ("." in item and wl is not None)
        item = item.removesuffix("nm").strip()
        m = re.fullmatch(r"(-?[\d.]+)\s*(?:-|:|to)\s*(-?[\d.]+)", item)
        lo, hi = (float(m.group(1)), float(m.group(2))) if m else (float(item), float(item))
        if lo > hi:
            lo, hi = hi, lo
        if use_wl:
            if wl is None:
                raise ValueError("This cube has no wavelength information; use band indices")
            sel = np.nonzero((wl >= lo) & (wl <= hi))[0]
            if sel.size == 0:  # range narrower than band spacing: take nearest band(s)
                sel = np.unique([np.abs(wl - lo).argmin(), np.abs(wl - hi).argmin()])
            indices.extend(sel.tolist())
        else:
            lo_i, hi_i = int(round(lo)), int(round(hi))
            if lo_i < 0 or hi_i >= cube.n_bands:
                raise ValueError(f"Band {lo_i if lo_i < 0 else hi_i} is out of range 0-{cube.n_bands - 1}")
            indices.extend(range(lo_i, hi_i + 1))
    if not indices:
        raise ValueError("No bands selected")
    return np.unique(indices)


def band_image(cube: Cube, bands, mode: str = "sum") -> np.ndarray:
    """Combine the selected bands into one 2-D float image (sum or mean)."""
    bands = np.atleast_1d(bands)
    out = np.zeros(cube.shape[:2], dtype=np.float64)
    for b in bands:  # band by band keeps memory flat for memmapped cubes
        out += cube.data[:, :, int(b)]
    if mode == "mean":
        out /= len(bands)
    return out


def stretch(img: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch to 0..1."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1e-12
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def rgb_composite(channels, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Stack up to three 2-D images (None for an empty channel) into an RGB float image."""
    ref = next(c for c in channels if c is not None)
    rgb = np.zeros(ref.shape + (3,), dtype=np.float32)
    for i, c in enumerate(channels[:3]):
        if c is not None:
            rgb[..., i] = stretch(c, low_pct, high_pct)
    return rgb


def pixel_spectrum(cube: Cube, row: float, col: float) -> np.ndarray | None:
    r, c = int(round(row)), int(round(col))
    if not (0 <= r < cube.shape[0] and 0 <= c < cube.shape[1]):
        return None
    return np.asarray(cube.data[r, c, :], dtype=np.float64)


def shape_mask(vertices: np.ndarray, image_shape) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Rasterise a polygon (N x 2 array of row, col vertices).

    Returns the mask cropped to the polygon's bounding box together with the
    slices locating that box in the image, so callers only read the pixels
    they need from a large cube.
    """
    from skimage.draw import polygon

    v = np.asarray(vertices, dtype=float)[:, -2:]
    r0 = max(int(np.floor(v[:, 0].min())), 0)
    r1 = min(int(np.ceil(v[:, 0].max())) + 1, image_shape[0])
    c0 = max(int(np.floor(v[:, 1].min())), 0)
    c1 = min(int(np.ceil(v[:, 1].max())) + 1, image_shape[1])
    if r1 <= r0 or c1 <= c0:
        return np.zeros((0, 0), bool), (slice(0, 0), slice(0, 0))
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    rr, cc = polygon(v[:, 0] - r0, v[:, 1] - c0, mask.shape)
    mask[rr, cc] = True
    return mask, (slice(r0, r1), slice(c0, c1))


def region_spectrum(cube: Cube, vertices: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Mean and standard deviation spectrum of pixels inside a polygon/rectangle."""
    mask, (rs, cs) = shape_mask(vertices, cube.shape[:2])
    n = int(mask.sum())
    if n == 0:
        return None, None, 0
    pixels = np.asarray(cube.data[rs, cs, :], dtype=np.float64)[mask]
    return pixels.mean(axis=0), pixels.std(axis=0), n


def label_spectra(cube: Cube, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean spectrum for every non-zero label of a segmentation."""
    out = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        out[int(lab)] = np.asarray(cube.data[labels == lab], dtype=np.float64).mean(axis=0)
    return out

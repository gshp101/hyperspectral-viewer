"""Band arithmetic, composites, filters and spectrum extraction. No GUI code here.

Image coordinates follow the viewer: ``x`` is the column and ``y`` the row,
and pixel ``(row, col)`` covers ``[col, col+1) x [row, row+1)``.
"""
from __future__ import annotations

import re

import numpy as np

from .io import Cube


# ------------------------------------------------------------- band selection
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
        try:
            lo, hi = (float(m.group(1)), float(m.group(2))) if m else (float(item), float(item))
        except ValueError:
            raise ValueError(f"Cannot read band selection '{raw.strip()}'") from None
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


def apply_filter(cube: Cube, filt: np.ndarray, pixel_norm: np.ndarray | None = None) -> np.ndarray:
    """Project every pixel spectrum onto a filter.

    A ``(bands,)`` vector gives the dot product. A ``(bands, k)`` matrix gives
    ``sqrt(|P^T s|^2 / |s|^2)``, the fraction of each spectrum's energy that
    lies in the filter's subspace (as in the original viewer).
    """
    flat = np.asarray(cube.data, dtype=np.float64).reshape(-1, cube.n_bands)
    proj = flat @ filt
    if filt.ndim == 2:
        if pixel_norm is None:
            pixel_norm = spectral_energy(cube)
        proj = np.sqrt((proj * proj).sum(axis=1) / np.where(pixel_norm.ravel() > 0, pixel_norm.ravel(), 1))
    return proj.reshape(cube.shape[:2])


def spectral_energy(cube: Cube) -> np.ndarray:
    """Per-pixel sum of squares over bands."""
    out = np.zeros(cube.shape[:2], dtype=np.float64)
    for b in range(cube.n_bands):
        band = np.asarray(cube.data[:, :, b], dtype=np.float64)
        out += band * band
    return out


# ----------------------------------------------------------------- display
SCALINGS = ("global", "channel", "percentile")


def to_uint8(channels: list[np.ndarray], scaling: str = "global", low_pct=2.0, high_pct=98.0) -> list[np.ndarray]:
    """Scale float images to uint8.

    ``global`` shares one min/max across all channels (keeps their relative
    brightness, like the original viewer), ``channel`` uses each channel's own
    min/max, ``percentile`` stretches each channel between two percentiles.
    """
    if not channels:
        return []
    if scaling == "global":
        finite = [c[np.isfinite(c)] for c in channels]
        lo = min((f.min() for f in finite if f.size), default=0.0)
        hi = max((f.max() for f in finite if f.size), default=1.0)
        limits = [(lo, hi)] * len(channels)
    else:
        limits = []
        for c in channels:
            f = c[np.isfinite(c)]
            if not f.size:
                limits.append((0.0, 1.0))
            elif scaling == "percentile":
                limits.append(tuple(np.percentile(f, [low_pct, high_pct])))
            else:
                limits.append((f.min(), f.max()))
    out = []
    for c, (lo, hi) in zip(channels, limits):
        span = hi - lo if hi > lo else 1.0
        scaled = np.clip((np.nan_to_num(c, nan=lo) - lo) * (255.0 / span), 0, 255)
        out.append(scaled.astype(np.uint8))
    return out


def compose(gray: np.ndarray | None = None, rgb: dict[int, np.ndarray] | None = None,
            scaling: str = "global") -> np.ndarray | None:
    """Build the uint8 display image.

    Either ``gray`` (a 2-D image, or a 3-channel image shown as-is) or ``rgb``,
    a mapping of channel index (0 red, 1 green, 2 blue) to 2-D images; missing
    channels are black.
    """
    if gray is not None:
        if gray.ndim == 3:
            if gray.dtype == np.uint8:
                return np.ascontiguousarray(gray[:, :, :3])
            chans = to_uint8([gray[:, :, i].astype(np.float64) for i in range(3)], scaling)
            return np.ascontiguousarray(np.stack(chans, axis=2))
        if gray.dtype == np.uint8:  # already a display image (e.g. a reference photo)
            return np.ascontiguousarray(gray)
        return np.ascontiguousarray(to_uint8([gray], scaling)[0])
    if not rgb:
        return None
    keys = sorted(rgb)
    chans = to_uint8([rgb[k] for k in keys], scaling)
    out = np.zeros(chans[0].shape + (3,), dtype=np.uint8)
    for k, c in zip(keys, chans):
        out[:, :, k] = c
    return out


def contrast_bias_lut(contrast_deg: float = 45.0, bias: float = 0.0) -> np.ndarray:
    """Look-up table for the original viewer's contrast (slope angle) and bias controls."""
    slope = np.tan(np.radians(contrast_deg))
    return np.clip(np.rint(bias + np.arange(256) * slope), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- spectra
def pixel_spectrum(cube: Cube, x: float, y: float) -> np.ndarray | None:
    """Spectrum of the pixel containing image position (x, y)."""
    c, r = int(np.floor(x)), int(np.floor(y))
    if not (0 <= r < cube.shape[0] and 0 <= c < cube.shape[1]):
        return None
    return np.asarray(cube.data[r, c, :], dtype=np.float64)


def rect_pixels(cube: Cube, p1, p2) -> np.ndarray:
    """All spectra in a rectangle, (n, bands).

    Pixel indices between the corner coordinates inclusive, as in the original
    viewer, so annotation files keep giving the same spectra.
    """
    (x0, y0), (x1, y1) = p1, p2
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    r0, r1 = max(int(np.ceil(ymin)), 0), min(int(np.floor(ymax)), cube.shape[0] - 1)
    c0, c1 = max(int(np.ceil(xmin)), 0), min(int(np.floor(xmax)), cube.shape[1] - 1)
    if r1 < r0 or c1 < c0:
        return np.empty((0, cube.n_bands))
    return np.asarray(cube.data[r0:r1 + 1, c0:c1 + 1, :], dtype=np.float64).reshape(-1, cube.n_bands)


def polygon_mask(vertices, image_shape) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Pixels whose centres lie inside a polygon of (x, y) vertices.

    Returns the mask cropped to the polygon's bounding box and the slices that
    place it in the image, so only the needed part of a large cube is read.
    """
    from matplotlib.path import Path

    v = np.asarray(vertices, dtype=float)
    r0 = max(int(np.floor(v[:, 1].min())), 0)
    r1 = min(int(np.ceil(v[:, 1].max())), image_shape[0])
    c0 = max(int(np.floor(v[:, 0].min())), 0)
    c1 = min(int(np.ceil(v[:, 0].max())), image_shape[1])
    if r1 <= r0 or c1 <= c0:
        return np.zeros((0, 0), bool), (slice(0, 0), slice(0, 0))
    yy, xx = np.mgrid[r0:r1, c0:c1]
    centres = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    mask = Path(v).contains_points(centres).reshape(yy.shape)
    return mask, (slice(r0, r1), slice(c0, c1))


def polygon_pixels(cube: Cube, vertices) -> np.ndarray:
    mask, (rs, cs) = polygon_mask(vertices, cube.shape[:2])
    if not mask.any():
        return np.empty((0, cube.n_bands))
    return np.asarray(cube.data[rs, cs, :], dtype=np.float64)[mask]


def label_spectra(cube: Cube, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean spectrum for every non-zero label of a segmentation."""
    out = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        out[int(lab)] = np.asarray(cube.data[labels == lab], dtype=np.float64).mean(axis=0)
    return out

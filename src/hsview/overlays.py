"""Overlay images (segmentations, score maps) and how they are coloured. No GUI code here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def label_colors(n: int) -> np.ndarray:
    """Distinct RGB colours for labels 1..n (index 0 is unused/transparent)."""
    from matplotlib import colormaps

    base = colormaps["tab20"].colors
    cols = np.array([base[(i * 2) % 20 if i < 10 else (i * 2 + 1) % 20] for i in range(max(n, 1))])
    return np.vstack([[0, 0, 0], cols[:n]])


def colorize(arr: np.ndarray, as_labels: bool, colors: dict | None = None) -> np.ndarray:
    """RGBA uint8 rendering of an overlay array.

    ``colors`` maps label values to RGB in 0..1 (e.g. annotation class colours
    for a supervised segmentation); other labels get the default palette.
    """
    if arr.ndim == 3:  # an RGB(A) image is shown as-is
        rgb = arr[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(255 * (rgb - rgb.min()) / max(np.ptp(rgb), 1e-12), 0, 255).astype(np.uint8)
        return np.dstack([rgb, np.full(arr.shape[:2], 255, np.uint8)])
    if as_labels:
        labels = arr.astype(np.int64)
        values = np.unique(labels)
        values = values[values != 0]
        lut_rgb = label_colors(len(values))
        if colors:
            for i, v in enumerate(values):
                if int(v) in colors:
                    lut_rgb[i + 1] = colors[int(v)]
        index = np.searchsorted(values, labels) + 1
        index[labels == 0] = 0
        rgba = np.zeros(arr.shape + (4,), np.uint8)
        rgba[..., :3] = (lut_rgb[index] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(labels == 0, 0, 255)
        return rgba
    from matplotlib import colormaps

    finite = arr[np.isfinite(arr)]
    lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0, 1))
    norm = np.clip((arr - lo) / (hi - lo if hi > lo else 1), 0, 1)
    rgba = (colormaps["magma"](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[~np.isfinite(arr), 3] = 0
    return rgba


@dataclass
class Overlay:
    key: str
    name: str
    data: np.ndarray
    as_labels: bool
    opacity: float = 0.5
    visible: bool = True
    colors: dict | None = None  # label value -> RGB in 0..1
    segmentation: object = None  # the hsview.analysis.Segmentation that made it, if any

    def label_color(self, value: int):
        if self.colors and int(value) in self.colors:
            return tuple(self.colors[int(value)])
        values = np.unique(self.data)
        values = values[values != 0]
        cols = label_colors(len(values))
        idx = int(np.searchsorted(values, value)) + 1
        return tuple(cols[idx]) if idx < len(cols) else (0.5, 0.5, 0.5)

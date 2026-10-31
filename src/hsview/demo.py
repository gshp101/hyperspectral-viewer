"""Synthetic cube for trying the viewer: ``python -m hsview.demo out.pkl``.

Writes the camera tools' layout: ``ref`` cube, ``wavelengths`` and an RGB
reference photo in ``lbl``. Also writes ``<out>_segmentation.npy``, a label
image to try as an overlay.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def make_demo_cube(rows=120, cols=160, n_bands=100, seed=0):
    """Three materials with distinct spectra on a background, plus noise.

    Returns (cube, wavelengths, labels, rgb reference).
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 1000, n_bands)

    def g(c, w):
        return np.exp(-0.5 * ((wl - c) / w) ** 2)

    spectra = np.stack([
        0.2 + 0.05 * g(700, 200),        # 0 background
        0.1 + 0.6 * g(550, 40),          # 1 green peak
        0.1 + 0.5 * (wl > 700) * (wl - 700) / 300 + 0.2 * g(480, 30),  # 2 blue + NIR rise
        0.6 - 0.4 * g(850, 60),          # 3 absorption feature
    ])
    lbl = np.zeros((rows, cols), np.int32)
    rr, cc = np.mgrid[:rows, :cols]
    lbl[(rr - 40) ** 2 + (cc - 45) ** 2 < 25 ** 2] = 1
    lbl[70:110, 80:140] = 2
    lbl[(rr - 35) ** 2 / 15 ** 2 + (cc - 120) ** 2 / 30 ** 2 < 1] = 3
    cube = spectra[lbl] * (1 + 0.1 * np.sin(cc / 15.0))[..., None]
    cube += rng.normal(0, 0.02, cube.shape)
    palette = np.array([[110, 100, 90], [60, 170, 70], [70, 90, 190], [220, 200, 160]], np.uint8)
    rgb = palette[lbl]
    return cube.astype(np.float32), wl, lbl, rgb


def main(path="demo_cube.pkl"):
    ref, wl, lbl, rgb = make_demo_cube()
    with open(path, "wb") as f:
        pickle.dump({"ref": ref, "wavelengths": wl, "lbl": rgb}, f)
    seg = Path(path).with_name(Path(path).stem + "_segmentation.npy")
    np.save(seg, lbl)
    print(f"wrote {path}: cube {ref.shape}, {len(wl)} wavelengths, reference image {rgb.shape}; labels in {seg}")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Saving/loading annotations and exporting spectra, independent of the GUI."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io import Cube
from .processing import pixel_spectrum, region_spectrum

FORMAT_VERSION = 1


def annotations_to_dict(points, point_ids, shapes, shape_types, shape_ids, cube: Cube | None = None) -> dict:
    """Plain-JSON representation. Coordinates are (row, col) in pixels."""
    return {
        "format": "hsview-annotations",
        "version": FORMAT_VERSION,
        "cube": cube.name if cube else None,
        "image_shape": list(cube.shape[:2]) if cube else None,
        "points": [{"id": int(i), "row": float(p[-2]), "col": float(p[-1])} for p, i in zip(points, point_ids)],
        "shapes": [
            {"id": int(i), "type": t, "vertices": np.asarray(v)[:, -2:].tolist()}
            for v, t, i in zip(shapes, shape_types, shape_ids)
        ],
    }


def save_annotations(path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_annotations(path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != "hsview-annotations":
        raise ValueError(f"{path} is not an hsview annotation file")
    return data


def annotation_spectra(cube: Cube, ann: dict) -> list[tuple[str, np.ndarray, int]]:
    """(label, spectrum, n_pixels) for every annotation; regions give mean spectra."""
    rows = []
    for p in ann["points"]:
        s = pixel_spectrum(cube, p["row"], p["col"])
        if s is not None:
            rows.append((f"point_{p['id']}", s, 1))
    for sh in ann["shapes"]:
        mean, _, n = region_spectrum(cube, np.asarray(sh["vertices"]))
        if mean is not None:
            rows.append((f"{sh['type']}_{sh['id']}", mean, n))
    return rows


def export_spectra_csv(path, cube: Cube, ann: dict) -> int:
    """One row per band, one column per annotation. Returns the number of spectra written."""
    spectra = annotation_spectra(cube, ann)
    x = cube.band_axis
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "wavelength" if cube.wavelengths is not None else "band_index"] + [f"{n} (n={k})" for n, _, k in spectra])
        for b in range(cube.n_bands):
            w.writerow([b, x[b]] + [f"{s[b]:.6g}" for _, s, _ in spectra])
    return len(spectra)

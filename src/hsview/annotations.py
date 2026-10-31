"""Annotation model, file formats and spectra export. No GUI code here.

Coordinates are image ``(x, y)`` = ``(column, row)``. Points are stored as the
index of the pixel they mark; rectangles as two corners; polygons as vertices.
"""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .io import Cube, LoadError
from .processing import pixel_spectrum, polygon_pixels, rect_pixels

# Classes of the original viewer, in order; files store the class index.
CLASS_NAMES = ["Red", "Green", "Blue", "Yellow", "Magenta", "Turquoise", "White"]
CLASS_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
]
# Darker versions for plotting on a white background.
CLASS_PLOT_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#bcbd22", "#c837ab", "#17becf", "#7f7f7f"]

KINDS = ("point", "rect", "polygon")
ANNOTATION_TAG = "annotation_version"
ANNOTATION_VERSION = "1.0"


@dataclass
class Annotation:
    kind: str  # "point" | "rect" | "polygon"
    cls: int
    coords: np.ndarray  # (n, 2) of (x, y)
    id: int = -1

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 2)

    def copy(self) -> "Annotation":
        return Annotation(self.kind, self.cls, self.coords.copy(), self.id)

    @property
    def label(self) -> str:
        return {"point": "Pt", "rect": "Rect", "polygon": "Poly"}[self.kind] + f" {self.id}"

    def pixels(self, cube: Cube) -> np.ndarray:
        """All spectra covered by the annotation, (n, bands)."""
        if self.kind == "point":
            s = pixel_spectrum(cube, *self.coords[0])
            return np.empty((0, cube.n_bands)) if s is None else s[None, :]
        if self.kind == "rect":
            return rect_pixels(cube, self.coords[0], self.coords[1])
        return polygon_pixels(cube, self.coords)


class AnnotationStore:
    """The list of annotations, with change notification and undo/redo."""

    def __init__(self):
        self._items: list[Annotation] = []
        self._next_id = 0
        self._undo: list[list[Annotation]] = []
        self._redo: list[list[Annotation]] = []
        self._listeners: list[Callable[[], None]] = []
        self.modified = False

    # --- observation
    def subscribe(self, fn: Callable[[], None]):
        self._listeners.append(fn)

    def _changed(self):
        self.modified = True
        for fn in self._listeners:
            fn()

    # --- access
    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def get(self, ann_id: int) -> Annotation | None:
        return next((a for a in self._items if a.id == ann_id), None)

    def of_kind(self, kind: str) -> list[Annotation]:
        return [a for a in self._items if a.kind == kind]

    # --- edits (each one is a single undo step)
    def _snapshot(self):
        self._undo.append([a.copy() for a in self._items])
        del self._undo[:-200]
        self._redo.clear()

    def add(self, kind: str, cls: int, coords) -> Annotation:
        self._snapshot()
        ann = Annotation(kind, int(cls), coords, self._next_id)
        self._next_id += 1
        self._items.append(ann)
        self._changed()
        return ann

    def remove(self, ids) -> int:
        ids = set(ids)
        if not any(a.id in ids for a in self._items):
            return 0
        self._snapshot()
        before = len(self._items)
        self._items = [a for a in self._items if a.id not in ids]
        self._changed()
        return before - len(self._items)

    def update(self, changes: dict[int, dict]):
        """Apply ``{id: {"coords": ..., "cls": ...}}`` as one undo step."""
        changes = {i: c for i, c in changes.items() if self.get(i) is not None}
        if not changes:
            return
        self._snapshot()
        for a in self._items:
            c = changes.get(a.id)
            if c:
                if "coords" in c:
                    a.coords = np.asarray(c["coords"], dtype=float).reshape(-1, 2)
                if "cls" in c:
                    a.cls = int(c["cls"])
        self._changed()

    def clear(self):
        if self._items:
            self._snapshot()
            self._items = []
            self._changed()

    def replace(self, items: list[Annotation], undoable: bool = True):
        if undoable:
            self._snapshot()
        self._items = []
        for a in items:
            a = a.copy()
            a.id = self._next_id
            self._next_id += 1
            self._items.append(a)
        self._changed()
        if not undoable:
            self.modified = False

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._items)
        self._items = self._undo.pop()
        self._changed()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._items)
        self._items = self._redo.pop()
        self._changed()
        return True

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)


# ------------------------------------------------------------ file formats
def to_legacy_dict(items) -> dict:
    """The original viewer's pickle layout, plus a 'polygons' entry it ignores."""
    out = {"points": [], "rects": [], "polygons": [], ANNOTATION_TAG: ANNOTATION_VERSION}
    for a in items:
        if a.kind == "point":
            x, y = a.coords[0]
            out["points"].append((float(x), float(y), a.cls))
        elif a.kind == "rect":
            out["rects"].append((a.coords[0].copy(), a.coords[1].copy(), a.cls))
        else:
            out["polygons"].append((a.coords.copy(), a.cls))
    return out


def from_legacy_dict(d: dict) -> list[Annotation]:
    if not isinstance(d, dict) or not {"points", "rects", ANNOTATION_TAG} <= d.keys():
        raise LoadError("Not an annotation file: expected 'points', 'rects' and 'annotation_version' entries")
    items = [Annotation("point", int(c), [[x, y]]) for x, y, c in d["points"]]
    items += [Annotation("rect", int(c), [p1, p2]) for p1, p2, c in d["rects"]]
    items += [Annotation("polygon", int(c), v) for v, c in d.get("polygons", [])]
    return items


def to_json_dict(items, cube: Cube | None = None) -> dict:
    return {
        "format": "hsview-annotations",
        ANNOTATION_TAG: ANNOTATION_VERSION,
        "cube": cube.name if cube else None,
        "image_shape": list(cube.shape[:2]) if cube else None,
        "classes": CLASS_NAMES,
        "annotations": [{"kind": a.kind, "class": a.cls, "xy": a.coords.tolist()} for a in items],
    }


def from_json_dict(d: dict) -> list[Annotation]:
    if d.get("format") != "hsview-annotations":
        raise LoadError("Not an hsview annotation file")
    return [Annotation(a["kind"], a["class"], a["xy"]) for a in d["annotations"]]


def save_annotations(path, items, cube: Cube | None = None):
    """Write ``.json`` as JSON, anything else as the original viewer's pickle."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(to_json_dict(items, cube), indent=1))
    else:
        with open(path, "wb") as f:
            pickle.dump(to_legacy_dict(items), f)


def load_annotations(path) -> list[Annotation]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return from_json_dict(json.loads(path.read_text()))
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as exc:
        raise LoadError(f"Could not load {path.name} as a pickle file") from exc
    return from_legacy_dict(d)


# ------------------------------------------------------------------ spectra
def annotation_spectra_dict(cube: Cube, items) -> dict:
    """The original viewer's 'Save Annotation Spectra' layout.

    ``point_spectra`` is (n_points, bands); ``rect_spectra`` and
    ``polygon_spectra`` hold every pixel spectrum of each region. The
    annotations themselves and the wavelengths are included.
    """
    items = list(items)
    points = [a.pixels(cube) for a in items if a.kind == "point"]
    out = {
        "point_spectra": np.array([p[0] for p in points if len(p)]).reshape(-1, cube.n_bands),
        "rect_spectra": [a.pixels(cube) for a in items if a.kind == "rect"],
        "polygon_spectra": [a.pixels(cube) for a in items if a.kind == "polygon"],
        "wavelengths": cube.wavelengths,
    }
    out.update(to_legacy_dict(items))
    return out


def save_annotation_spectra(path, cube: Cube, items):
    with open(path, "wb") as f:
        pickle.dump(annotation_spectra_dict(cube, items), f)


def mean_spectra(cube: Cube, items) -> list[tuple[Annotation, np.ndarray, np.ndarray, int]]:
    """(annotation, mean, std, n_pixels) for every annotation covering at least one pixel."""
    out = []
    for a in items:
        px = a.pixels(cube)
        if len(px):
            out.append((a, px.mean(axis=0), px.std(axis=0), len(px)))
    return out


def export_spectra_csv(path, cube: Cube, items) -> int:
    """One row per band, one column per annotation (mean for regions). Returns the number of spectra."""
    spectra = mean_spectra(cube, items)
    x = cube.band_axis
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "wavelength" if cube.wavelengths is not None else "band_index"]
                   + [f"{a.label} {CLASS_NAMES[a.cls]} (n={n})" for a, _, _, n in spectra])
        for b in range(cube.n_bands):
            w.writerow([b, x[b]] + [f"{m[b]:.6g}" for _, m, _, _ in spectra])
    return len(spectra)

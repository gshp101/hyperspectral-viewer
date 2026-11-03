"""Segment a cube into regions, without or with training annotations.

Every pixel spectrum is first turned into a feature vector. With
``intensity="ignore"`` each spectrum is scaled to unit length, so only its
shape matters: a pixel and the same pixel at any brightness land in the same
class, and Euclidean distance between features is a monotonic function of the
spectral angle. ``intensity="ignore_offset"`` also subtracts each spectrum's
own mean first, so a constant offset (stray light, dark level) is ignored too;
distances then follow the correlation between spectra. The features are then
optionally projected onto the first principal components of the cube.

* :func:`segment_unsupervised` clusters the features with k-means.
* :func:`segment_supervised` takes the pixels under the annotations as
  training data (a point is one pixel, a rectangle or polygon every pixel
  inside it) and classifies every pixel with a nearest class mean, linear
  discriminant or k-nearest-neighbour rule. Label values are the annotation
  class index + 1, so label 1 is always the Red class.

Both return a :class:`Segmentation`, which :func:`save_segmentation` writes as
a ``.pkl`` or ``.npz`` file that can be read back with plain pickle/numpy
(see ``docs/segmentation-format.md``). No Qt here.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..annotations import CLASS_COLORS, CLASS_NAMES, Annotation, to_legacy_dict
from ..io import Cube, LoadError
from ..processing import polygon_mask
from .background import _blocks

INTENSITY_MODES = {
    "use": "Use intensity",
    "ignore": "Ignore intensity (spectral shape, angle)",
    "ignore_offset": "Ignore intensity and offset (correlation)",
}
FORMAT = "hsview-segmentation"
FORMAT_VERSION = 1
SUPERVISED_METHODS = {
    "nearest_mean": "Nearest class mean",
    "lda": "Linear discriminant (LDA)",
    "knn": "k-nearest neighbours",
}
PCA_SAMPLE = 200_000  # pixels used to fit the principal components
DIST_CHUNK = 1 << 16  # pixels per block when computing distances


# ------------------------------------------------------------------ result
@dataclass
class Segmentation:
    labels: np.ndarray  # (rows, cols) int32; 0 = not assigned
    method: str
    supervised: bool
    label_names: dict[int, str]  # label value -> name
    label_colors: dict[int, tuple[float, float, float]]  # label value -> RGB in 0..1
    params: dict = field(default_factory=dict)
    mean_spectra: dict[int, np.ndarray] = field(default_factory=dict)  # label value -> (bands,) in cube units
    pixel_counts: dict[int, int] = field(default_factory=dict)
    annotations: list[Annotation] | None = None  # training annotations (supervised only)
    training_labels: np.ndarray | None = None  # (rows, cols) int32 class + 1 under the annotations, else 0
    wavelengths: np.ndarray | None = None
    cube_info: dict = field(default_factory=dict)  # name, path, shape and processing history
    created: str = ""

    @property
    def name(self) -> str:
        if self.supervised:
            return f"{SUPERVISED_METHODS.get(self.method, self.method)} ({len(self.label_names)} classes)"
        return f"k-means ({len(self.label_names)} clusters)"


# ---------------------------------------------------------------- features
@dataclass
class FeatureTransform:
    """Spectrum -> feature vector: optional removal of intensity (see ``INTENSITY_MODES``), then optional PCA."""

    intensity: str = "use"
    mean: np.ndarray | None = None  # (bands,), used with basis
    basis: np.ndarray | None = None  # (bands, d)

    def __post_init__(self):
        if self.intensity not in INTENSITY_MODES:
            raise ValueError(f"intensity must be one of {', '.join(INTENSITY_MODES)}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.intensity == "ignore_offset":
            x = x - x.mean(axis=-1, keepdims=True)
        if self.intensity != "use":
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            x = x / np.where(norm > 0, norm, 1)
        if self.basis is not None:
            x = (x - self.mean) @ self.basis
        return x.astype(np.float32)


def _sample_pixels(cube: Cube, n: int, rng) -> np.ndarray:
    rows, cols, bands = cube.shape
    total = rows * cols
    if total <= n:
        return np.asarray(cube.data, dtype=np.float64).reshape(-1, bands)
    idx = np.sort(rng.choice(total, n, replace=False))
    r, c = np.divmod(idx, cols)
    return np.asarray(cube.data[r, c, :], dtype=np.float64)


def fit_features(cube: Cube, intensity: str = "use", n_components: int = 0, seed: int = 0) -> FeatureTransform:
    """Fit the feature transform; ``n_components`` 0 keeps every band."""
    t = FeatureTransform(intensity)
    n_components = int(n_components)
    if 0 < n_components < cube.n_bands:
        x = t(_sample_pixels(cube, PCA_SAMPLE, np.random.default_rng(seed))).astype(np.float64)
        mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        basis = vt[:n_components].T
        signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(basis.shape[1])])
        signs[signs == 0] = 1
        t.mean, t.basis = mean, basis * signs
    return t


def cube_features(cube: Cube, transform: FeatureTransform) -> np.ndarray:
    """(rows * cols, d) float32 features of every pixel, in row-major order."""
    rows, cols, _ = cube.shape
    first = transform(np.asarray(cube.data[0, 0, :])[None])
    out = np.empty((rows * cols, first.shape[1]), np.float32)
    for rs, x in _blocks(cube):
        out[rs.start * cols:rs.stop * cols] = transform(x)
    return out


# ---------------------------------------------------------------- helpers
def _sq_dist(x: np.ndarray, centres: np.ndarray, centres_sq: np.ndarray | None = None) -> np.ndarray:
    if centres_sq is None:
        centres_sq = (centres.astype(np.float64) ** 2).sum(axis=1)
    d = (x.astype(np.float64) ** 2).sum(axis=1)[:, None] - 2.0 * (x @ centres.T) + centres_sq[None, :]
    return np.maximum(d, 0)


def _argmin_dist(x: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and squared distance to the nearest centre for every row of ``x``."""
    centres = centres.astype(np.float32)
    csq = (centres.astype(np.float64) ** 2).sum(axis=1)
    idx = np.empty(len(x), np.int32)
    dist = np.empty(len(x), np.float64)
    for s in range(0, len(x), DIST_CHUNK):
        d = _sq_dist(x[s:s + DIST_CHUNK], centres, csq)
        idx[s:s + DIST_CHUNK] = d.argmin(axis=1)
        dist[s:s + DIST_CHUNK] = d[np.arange(len(d)), idx[s:s + DIST_CHUNK]]
    return idx, dist


def _group_means(x: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    sums = np.stack([np.bincount(labels, weights=x[:, j], minlength=k) for j in range(x.shape[1])], axis=1)
    return sums / np.maximum(counts, 1)[:, None], counts


def label_mean_spectra(cube: Cube, labels: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Mean spectrum (cube units) and pixel count of every non-zero label, read block by block."""
    values = np.unique(labels)
    values = values[values != 0]
    if not len(values):
        return {}, {}
    index = np.searchsorted(values, labels)
    index[labels == 0] = len(values)  # a bin for unassigned pixels, dropped below
    k = len(values) + 1
    sums = np.zeros((k, cube.n_bands))
    counts = np.zeros(k)
    for rs, x in _blocks(cube):
        lab = index[rs].ravel()
        counts += np.bincount(lab, minlength=k)
        for j in range(cube.n_bands):
            sums[:, j] += np.bincount(lab, weights=x[:, j], minlength=k)
    means = sums[:-1] / np.maximum(counts[:-1], 1)[:, None]
    return ({int(v): means[i] for i, v in enumerate(values)},
            {int(v): int(counts[i]) for i, v in enumerate(values)})


def majority_filter(labels: np.ndarray, size: int) -> np.ndarray:
    """Replace each label by the most common label in a ``size`` x ``size`` window (removes speckle)."""
    size = int(size)
    if size <= 1:
        return labels
    r = size // 2
    values = np.unique(labels)
    best = labels.copy()
    best_count = np.full(labels.shape, -1, np.int32)
    for v in values:
        m = np.pad((labels == v).astype(np.int32), ((r + 1, r), (r + 1, r)))
        s = m.cumsum(0).cumsum(1)
        count = s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]
        better = count > best_count
        best[better] = v
        best_count[better] = count[better]
    return best


def annotation_mask(ann: Annotation, shape) -> np.ndarray:
    """Boolean (rows, cols) mask of the pixels an annotation covers (same pixels as its spectra)."""
    rows, cols = shape
    mask = np.zeros((rows, cols), bool)
    if ann.kind == "point":
        c, r = (int(np.floor(v)) for v in ann.coords[0])
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = True
    elif ann.kind == "rect":
        (x0, y0), (x1, y1) = ann.coords[:2]
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        r0, r1 = max(int(np.ceil(ymin)), 0), min(int(np.floor(ymax)), rows - 1)
        c0, c1 = max(int(np.ceil(xmin)), 0), min(int(np.floor(xmax)), cols - 1)
        if r1 >= r0 and c1 >= c0:
            mask[r0:r1 + 1, c0:c1 + 1] = True
    else:
        sub, (rs, cs) = polygon_mask(ann.coords, shape)
        if sub.size:
            mask[rs, cs] = sub
    return mask


def training_label_image(annotations, shape) -> np.ndarray:
    """(rows, cols) int32 with class index + 1 under each annotation (later annotations win), else 0."""
    out = np.zeros(shape, np.int32)
    for ann in annotations:
        out[annotation_mask(ann, shape)] = ann.cls + 1
    return out


def _cube_info(cube: Cube) -> dict:
    return {"name": cube.name, "path": cube.path, "shape": list(cube.shape),
            "processing": list(cube.metadata.get("analysis", []))}


def _finish(seg: Segmentation, cube: Cube, smooth: int) -> Segmentation:
    if smooth > 1:
        seg.labels = majority_filter(seg.labels, smooth).astype(np.int32)
    seg.mean_spectra, seg.pixel_counts = label_mean_spectra(cube, seg.labels)
    seg.wavelengths = None if cube.wavelengths is None else np.asarray(cube.wavelengths, float)
    seg.cube_info = _cube_info(cube)
    seg.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return seg


def cluster_colors(n: int) -> dict[int, tuple[float, float, float]]:
    from matplotlib import colormaps

    base = colormaps["tab20"].colors
    return {i + 1: tuple(float(c) for c in base[(i * 2) % 20 if i < 10 else (i * 2 + 1) % 20]) for i in range(n)}


# ------------------------------------------------------------ unsupervised
def kmeans(x: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-4, seed: int = 0,
           init_sample: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ seeding. Returns (centres (k, d), labels 0..k-1)."""
    n = len(x)
    if not 1 <= k <= n:
        raise ValueError(f"The number of clusters must be between 1 and {n}.")
    rng = np.random.default_rng(seed)
    sample = x[rng.choice(n, min(n, init_sample), replace=False)].astype(np.float64)
    centres = [sample[rng.integers(len(sample))]]
    d = _sq_dist(sample, np.array(centres)).ravel()
    for _ in range(1, k):
        p = d / d.sum() if d.sum() > 0 else None
        centres.append(sample[rng.choice(len(sample), p=p)])
        d = np.minimum(d, _sq_dist(sample, centres[-1][None]).ravel())
    centres = np.array(centres)
    scale = float(np.mean(np.var(sample, axis=0))) or 1.0
    labels = np.zeros(n, np.int32)
    for _ in range(max_iter):
        labels, dist = _argmin_dist(x, centres)
        new, counts = _group_means(x, labels, k)
        for j in np.flatnonzero(counts == 0):  # re-seed empty clusters at the worst-fitting pixel
            far = int(dist.argmax())
            new[j] = x[far]
            dist[far] = 0
        shift = float(((new - centres) ** 2).sum(axis=1).max())
        centres = new
        if shift <= tol * scale:
            break
    labels, _ = _argmin_dist(x, centres)
    return centres, labels


def segment_unsupervised(cube: Cube, n_clusters: int = 6, n_components: int = 10, intensity: str = "use",
                         max_iter: int = 50, smooth: int = 0, seed: int = 0) -> Segmentation:
    """k-means clustering of the pixel spectra. Labels are 1..n_clusters, largest cluster first.

    ``intensity`` is ``"use"``, ``"ignore"`` or ``"ignore_offset"`` (see ``INTENSITY_MODES``).
    """
    transform = fit_features(cube, intensity, n_components, seed)
    x = cube_features(cube, transform)
    _, idx = kmeans(x, int(n_clusters), max_iter=max_iter, seed=seed)
    k = int(n_clusters)
    order = np.argsort(-np.bincount(idx, minlength=k), kind="stable")
    relabel = np.empty(k, np.int32)
    relabel[order] = np.arange(1, k + 1)
    labels = relabel[idx].reshape(cube.shape[:2])
    seg = Segmentation(labels, "kmeans", False, {i: f"Cluster {i}" for i in range(1, k + 1)}, cluster_colors(k),
                       params={"n_clusters": k, "n_components": int(n_components), "intensity": intensity,
                               "max_iter": int(max_iter), "smooth": int(smooth), "seed": int(seed)})
    return _finish(seg, cube, smooth)


# -------------------------------------------------------------- supervised
def _lda(train_x, train_y, classes, shrinkage=0.1):
    """Linear discriminant with a shared, shrunk covariance and equal priors."""
    means = np.array([train_x[train_y == c].mean(axis=0) for c in classes], dtype=np.float64)
    resid = train_x - means[np.searchsorted(classes, train_y)]
    cov = resid.T @ resid / max(len(train_x) - len(classes), 1)
    d = cov.shape[0]
    target = np.trace(cov) / d if d else 1.0
    cov = (1 - shrinkage) * cov + shrinkage * (target if target > 0 else 1.0) * np.eye(d)
    w = np.linalg.solve(cov, means.T)  # (d, n_classes)
    b = -0.5 * np.einsum("ij,ji->i", means, w)
    return w, b


def segment_supervised(cube: Cube, annotations, method: str = "nearest_mean", n_components: int = 10,
                       intensity: str = "use", k_neighbors: int = 5, max_per_class: int = 500,
                       smooth: int = 0, seed: int = 0) -> Segmentation:
    """Classify every pixel from the pixels under the annotations.

    ``method`` is ``"nearest_mean"``, ``"lda"`` or ``"knn"``. Label values are
    annotation class + 1. Regions contribute every pixel inside them; at most
    ``max_per_class`` training pixels per class are used by k-NN. ``intensity``
    is ``"use"``, ``"ignore"`` or ``"ignore_offset"`` (see ``INTENSITY_MODES``):
    with either ignore mode a class is defined by spectral shape only, so a
    pixel is classified the same at any brightness.
    """
    if method not in SUPERVISED_METHODS:
        raise ValueError(f"Unknown method {method!r}")
    annotations = [a.copy() for a in annotations]
    shape = cube.shape[:2]
    train = training_label_image(annotations, shape)
    classes = np.unique(train[train > 0])
    if len(classes) < 2:
        raise ValueError("Supervised segmentation needs annotations of at least two classes "
                         "that cover pixels of the image.")
    transform = fit_features(cube, intensity, n_components, seed)
    x = cube_features(cube, transform)
    flat_train = train.ravel()
    train_idx = np.flatnonzero(flat_train)
    train_x, train_y = x[train_idx].astype(np.float64), flat_train[train_idx]

    if method == "nearest_mean":
        means = np.array([train_x[train_y == c].mean(axis=0) for c in classes])
        idx, _ = _argmin_dist(x, means)
        labels = classes[idx]
    elif method == "lda":
        w, b = _lda(train_x, train_y, classes)
        labels = np.empty(len(x), np.int32)
        for s in range(0, len(x), DIST_CHUNK):
            labels[s:s + DIST_CHUNK] = classes[(x[s:s + DIST_CHUNK] @ w + b).argmax(axis=1)]
    else:
        rng = np.random.default_rng(seed)
        keep = []
        for c in classes:
            members = np.flatnonzero(train_y == c)
            if len(members) > max_per_class:
                members = rng.choice(members, max_per_class, replace=False)
            keep.append(members)
        keep = np.concatenate(keep)
        ref_x, ref_y = train_x[keep].astype(np.float32), train_y[keep]
        k = max(1, min(int(k_neighbors), len(ref_x)))
        rsq = (ref_x.astype(np.float64) ** 2).sum(axis=1)
        chunk = max(256, (1 << 24) // max(len(ref_x), 1))
        labels = np.empty(len(x), np.int32)
        for s in range(0, len(x), chunk):
            d = _sq_dist(x[s:s + chunk], ref_x, rsq)
            nn = np.argpartition(d, k - 1, axis=1)[:, :k] if k < d.shape[1] else np.argsort(d, axis=1)
            votes = np.zeros((len(d), len(classes)))
            cls_idx = np.searchsorted(classes, ref_y[nn])
            np.add.at(votes, (np.arange(len(d))[:, None], cls_idx), 1.0)
            # Ties go to the class of the single nearest neighbour.
            nearest = np.searchsorted(classes, ref_y[d.argmin(axis=1)])
            votes[np.arange(len(d)), nearest] += 0.5
            labels[s:s + chunk] = classes[votes.argmax(axis=1)]
    labels = labels.reshape(shape).astype(np.int32)
    names = {int(c): CLASS_NAMES[c - 1] for c in classes}
    colors = {int(c): tuple(float(v) for v in CLASS_COLORS[c - 1]) for c in classes}
    seg = Segmentation(labels, method, True, names, colors,
                       params={"method": method, "n_components": int(n_components), "intensity": intensity,
                               "k_neighbors": int(k_neighbors), "max_per_class": int(max_per_class),
                               "smooth": int(smooth), "seed": int(seed),
                               "training_pixels": {CLASS_NAMES[c - 1]: int((train_y == c).sum()) for c in classes}},
                       annotations=annotations, training_labels=train)
    return _finish(seg, cube, smooth)


# ------------------------------------------------------------------- files
def _annotation_records(annotations) -> list[dict]:
    return [{"kind": a.kind, "class": int(a.cls), "class_name": CLASS_NAMES[a.cls], "xy": a.coords.tolist()}
            for a in annotations]


def segmentation_dict(seg: Segmentation) -> dict:
    """The plain-Python layout written to ``.pkl`` files (only numpy arrays, lists, dicts and strings)."""
    values = np.array(sorted(seg.label_names), dtype=np.int32)
    bands = len(next(iter(seg.mean_spectra.values()))) if seg.mean_spectra else 0
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "labels": np.asarray(seg.labels, np.int32),
        "label_values": values,
        "label_names": [seg.label_names[int(v)] for v in values],
        "label_colors": np.array([seg.label_colors[int(v)] for v in values], float).reshape(-1, 3),
        "mean_spectra": np.array([seg.mean_spectra.get(int(v), np.full(bands, np.nan)) for v in values],
                                 float).reshape(len(values), bands),
        "pixel_counts": np.array([seg.pixel_counts.get(int(v), 0) for v in values], np.int64),
        "wavelengths": seg.wavelengths,
        "method": seg.method,
        "supervised": seg.supervised,
        "params": dict(seg.params),
        "cube": dict(seg.cube_info),
        "created": seg.created,
        "class_names": list(CLASS_NAMES),
        "annotations": _annotation_records(seg.annotations) if seg.annotations is not None else None,
        "annotations_legacy": to_legacy_dict(seg.annotations) if seg.annotations is not None else None,
        "training_labels": None if seg.training_labels is None else np.asarray(seg.training_labels, np.int32),
    }


_NPZ_ARRAYS = ("labels", "label_values", "label_colors", "mean_spectra", "pixel_counts", "wavelengths",
               "training_labels")


def save_segmentation(path, seg: Segmentation) -> None:
    """Write ``.npz`` (numpy only, no pickle needed to read) or, for any other suffix, a pickled dict."""
    path = Path(path)
    d = segmentation_dict(seg)
    if path.suffix.lower() == ".npz":
        arrays = {k: d[k] for k in _NPZ_ARRAYS if d[k] is not None}
        meta = {k: v for k, v in d.items() if k not in _NPZ_ARRAYS and k != "annotations_legacy"}
        arrays["metadata_json"] = np.array(json.dumps(meta))
        if seg.annotations is not None:
            arrays["annotation_kind"] = np.array([a.kind for a in seg.annotations], dtype="<U7")
            arrays["annotation_class"] = np.array([a.cls for a in seg.annotations], np.int32)
            xy = [a.coords for a in seg.annotations]
            arrays["annotation_xy"] = np.concatenate(xy).astype(float) if xy else np.zeros((0, 2))
            arrays["annotation_offsets"] = np.cumsum([0] + [len(c) for c in xy]).astype(np.int64)
        np.savez_compressed(path, **arrays)
    else:
        with open(path, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_segmentation(path) -> dict:
    """Read either file layout back into the :func:`segmentation_dict` layout."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "metadata_json" not in z:
                raise LoadError(f"{path.name} is not an hsview segmentation")
            d = json.loads(str(z["metadata_json"]))
            for k in _NPZ_ARRAYS:
                d[k] = z[k] if k in z else None
            if "annotation_kind" in z:
                off = z["annotation_offsets"]
                xy = z["annotation_xy"]
                d["annotations"] = [{"kind": str(k), "class": int(c), "class_name": CLASS_NAMES[int(c)],
                                     "xy": xy[off[i]:off[i + 1]].tolist()}
                                    for i, (k, c) in enumerate(zip(z["annotation_kind"], z["annotation_class"]))]
    else:
        with open(path, "rb") as f:
            d = pickle.load(f)
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise LoadError(f"{path.name} is not an hsview segmentation")
    return d


def is_segmentation_file(path) -> bool:
    try:
        read_segmentation(path)
        return True
    except Exception:
        return False


def annotations_from_segmentation(d: dict) -> list[Annotation]:
    return [Annotation(a["kind"], int(a["class"]), a["xy"]) for a in (d.get("annotations") or [])]


def segmentation_from_dict(d: dict) -> Segmentation:
    values = [int(v) for v in d["label_values"]]
    means = d.get("mean_spectra")
    counts = d.get("pixel_counts")
    anns = annotations_from_segmentation(d) if d.get("annotations") is not None else None
    return Segmentation(
        np.asarray(d["labels"], np.int32), d.get("method", ""), bool(d.get("supervised")),
        dict(zip(values, d["label_names"])),
        {v: tuple(float(c) for c in col) for v, col in zip(values, np.asarray(d["label_colors"]))},
        params=dict(d.get("params") or {}),
        mean_spectra={v: np.asarray(means[i]) for i, v in enumerate(values)} if means is not None else {},
        pixel_counts={v: int(counts[i]) for i, v in enumerate(values)} if counts is not None else {},
        annotations=anns, training_labels=d.get("training_labels"),
        wavelengths=d.get("wavelengths"), cube_info=dict(d.get("cube") or {}), created=d.get("created", ""))


def segmentation_from_labels(labels: np.ndarray, cube: Cube | None = None, name: str = "labels",
                             colors: dict | None = None) -> Segmentation:
    """Wrap a plain label image (e.g. an overlay loaded from a file) so it can be saved in this format."""
    labels = np.asarray(labels).astype(np.int32)
    values = [int(v) for v in np.unique(labels) if v != 0]
    default = cluster_colors(len(values))
    seg = Segmentation(labels, name, False, {v: f"Label {v}" for v in values},
                       {v: tuple(colors[v]) if colors and v in colors else default[i + 1]
                        for i, v in enumerate(values)})
    if cube is not None:
        seg = _finish(seg, cube, 0)
    return seg

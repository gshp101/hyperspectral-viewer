"""The state and numerics behind the browser version (``web/``), with no GUI code.

The browser page runs this package in Pyodide. Its JavaScript draws the
window (menus, toolbars, panels, canvas and plots) and calls a single
:class:`WebApp`, which plays the role of the Qt app's ``MainWindow`` and its
panels: it keeps the cube, annotations, overlays and display settings, and
does all the work with the same functions as the desktop app
(:mod:`hsview.io`, :mod:`hsview.processing`, :mod:`hsview.annotations`,
:mod:`hsview.analysis`).

Files reach Python through Pyodide's in-memory file system: the page writes an
uploaded file to a path and passes the path here; saves are written to a path
the page then downloads. Everything returned is plain lists, dicts, numbers,
strings and ``bytes``, so it converts to JavaScript directly. Because nothing
here needs a browser, the same class is tested with pytest.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from . import annotations as ann_io
from .analysis import fit_svd_background, remove_svd_background
from .analysis.segmentation import (
    INTENSITY_MODES,
    SUPERVISED_METHODS,
    is_segmentation_file,
    read_segmentation,
    save_segmentation,
    segment_supervised,
    segment_unsupervised,
    segmentation_from_dict,
    segmentation_from_labels,
    training_label_image,
)
from .annotations import CLASS_COLORS, CLASS_NAMES, CLASS_PLOT_COLORS, AnnotationStore, mean_spectra
from .app_icon_svg import APP_ICONS, DEFAULT_APP_ICON
from .datasets import DATASETS, OPENERS
from .io import (
    Cube,
    LoadError,
    is_label_image,
    load_cube,
    load_filter,
    load_overlay,
    read_image,
    resolve_image_id,
    save_cube,
)
from .overlays import Overlay, colorize
from .processing import (
    SCALINGS,
    apply_filter,
    band_image,
    band_window,
    compose,
    contrast_bias_lut,
    label_spectra,
    parse_band_spec,
    pixel_spectrum,
    spectral_energy,
)
from .ui_spec import (
    DEFAULT_BIAS,
    DEFAULT_CONTRAST,
    DEFAULT_TOOLBAR_CONFIG,
    ICON_SIZES,
    ICON_SVG,
    REF_ROUTES,
    ROUTES,
    SCALING_LABELS,
    SEPARATOR,
    SMOOTHING,
    SOURCES,
    SPACER,
    STYLE_LABELS,
    TRUE_COLOUR,
    WIDGET_ITEMS,
    build_specs,
    menu_layout,
)


class WebAnalysis:
    """An Analysis menu entry; the browser page shows its parameter dialog."""

    def __init__(self, id, text, icon, tip):
        self.id, self.text, self.icon, self.tip = id, text, icon, tip


# The same entries as hsview.gui.analysis.ANALYSES (that module needs Qt).
ANALYSES = [
    WebAnalysis("analysis_svd_background", "Remove background (SVD)…", "background",
                "Subtract the mean and project out the largest SVD components"),
]


def _color_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(255 * float(c))) for c in rgb[:3])


def _floats(a) -> list:
    """A float array as a list, with NaN/inf as None (valid JSON, a gap in plots)."""
    return [float(v) if math.isfinite(v) else None for v in np.asarray(a, dtype=float).ravel()]


def _safe_name(name: str, default: str = "cube") -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name).strip("_") or default


def ui_tables() -> dict:
    """Everything the page needs to build menus, toolbars and dialogs, from the tables the Qt app uses."""
    specs = build_specs(ANALYSES)

    def menu_entries(ids):
        out = []
        for entry in ids:
            if entry is None:
                out.append(None)
            elif isinstance(entry, tuple):
                out.append({"title": entry[0], "items": menu_entries(entry[1])})
            else:
                out.append(entry)
        return out

    return {
        "version": __version__,
        "specs": [asdict(spec) for spec in specs],
        "widget_items": {k: {"text": t, "category": c} for k, (t, c) in WIDGET_ITEMS.items()},
        "menus": [{"title": title, "items": menu_entries(ids)} for title, ids in menu_layout(ANALYSES).items()],
        "toolbars": DEFAULT_TOOLBAR_CONFIG,
        "separator": SEPARATOR,
        "spacer": SPACER,
        "icon_sizes": list(ICON_SIZES),
        "styles": STYLE_LABELS,
        "icons": ICON_SVG,
        "datasets": {d.key: {"title": d.title, "description": d.description, "credit": d.credit, "page": d.page,
                             "size": d.size,
                             "files": {role: {"name": f.name, "size": f.size, "urls": list(f.urls)}
                                       for role, f in d.files.items()}}
                     for d in DATASETS.values()},
        "app_icons": {k: {"label": label, "svg": svg} for k, (label, svg) in APP_ICONS.items()},
        "default_app_icon": DEFAULT_APP_ICON,
        "class_names": CLASS_NAMES,
        "class_colors": [_color_hex(c) for c in CLASS_COLORS],
        "class_plot_colors": CLASS_PLOT_COLORS,
        "routes": list(ROUTES),
        "ref_routes": list(REF_ROUTES),
        "sources": list(SOURCES),
        "scalings": [{"key": k, "label": SCALING_LABELS[k]} for k in SCALINGS],
        "smoothing": [{"size": s, "label": t} for s, t in SMOOTHING],
        "intensity_modes": [{"key": k, "label": v} for k, v in INTENSITY_MODES.items()],
        "supervised_methods": [{"key": k, "label": v} for k, v in SUPERVISED_METHODS.items()],
        "default_contrast": DEFAULT_CONTRAST,
        "default_bias": DEFAULT_BIAS,
        "true_colour": list(TRUE_COLOUR),
    }


# ------------------------------------------------------------------ display
class Channel:
    """One band channel of the Display panel (source, routing and its cached image)."""

    def __init__(self):
        self.source = "Band"
        self.route = "None"
        self.band = 0
        self.width = 1
        self.spec = ""
        self.sum_mode = "sum"
        self.filter: np.ndarray | None = None
        self.filter_name = ""
        self.cache = None


class Reference:
    def __init__(self):
        self.img: np.ndarray | None = None
        self.name = ""
        self.route = "None"
        self.opacity = 50


class WebApp:
    """The browser version's application state; one method per thing the page can ask for."""

    def __init__(self):
        self.cube: Cube | None = None
        self.cube_history: list[Cube] = []
        self.store = AnnotationStore()
        self.annotation_name: str | None = None  # file name the annotations were loaded from or saved to
        self.channels = [Channel() for _ in range(3)]
        self.reference = Reference()
        self.scaling = "global"
        self.contrast = DEFAULT_CONTRAST
        self.bias = DEFAULT_BIAS
        self._energy = None
        self._composite = None
        self.overlays: dict[str, Overlay] = {}
        self.current_overlay: str | None = None
        self._overlay_counter = 0
        self._label_cache: dict[str, dict] = {}
        self._svd = None
        self._readout_band = 0
        self.version = 0  # bumped on every change the page should redraw for
        self.image_version = 0  # bumped when the display image may have changed

    # ------------------------------------------------------------ cube
    def open_cube(self, path: str, name: str | None = None) -> dict:
        cube = load_cube(path)
        if name:
            cube.name = Path(name).stem
        self.set_cube(cube)
        return self.info()

    def load_demo(self) -> dict:
        from .demo import make_demo_cube

        ref, wl, lbl, rgb = make_demo_cube()
        self.set_cube(Cube(ref, wl, "", "demo_cube", reference=rgb))
        self.add_overlay_array(lbl, "demo_cube_segmentation")
        self.set_overlay_visible(self.current_overlay, False)
        return self.info()

    def open_dataset(self, key: str, paths: dict) -> dict:
        """Open a downloaded example scene (see hsview.datasets) with its ground truth as an overlay."""
        cube, seg = OPENERS[key](paths["cube"], paths["ground_truth"])
        self.set_cube(cube)
        self.add_segmentation(seg, f"{cube.name} ground truth")
        info = self.info()
        info["messages"] = [f"{DATASETS[key].title}: {DATASETS[key].description}. {cube.metadata['wavelength_note']}"]
        return info

    def set_cube(self, cube: Cube):
        self.cube = cube
        self.cube_history.clear()
        self.annotation_name = None
        self.store.replace([], undoable=False)
        self.store._undo.clear()
        self.store._redo.clear()
        self.overlays.clear()
        self.current_overlay = None
        self._label_cache.clear()
        self._svd = None
        self._energy = None
        n = cube.n_bands
        for ch in self.channels:
            ch.cache = None
            ch.filter, ch.filter_name = None, ""
            ch.route = "None"
            ch.band = min(ch.band, n - 1)
            ch.width = max(1, min(ch.width, n))
            if ch.source == "Filter":
                ch.source = "Band"
        ref = self.reference
        ref.img = None if cube.reference is None else np.asarray(cube.reference)
        ref.name = "from file" if ref.img is not None else ""
        ref.route = "None"
        if ref.img is not None and cube.band_names in (("red", "green", "blue"), ("gray",)):
            ref.route = "Show"  # plain photos show as themselves
        else:
            for i, ch in enumerate(self.channels):
                ch.band = min(n - 1, (i + 1) * n // 4)
            self.channels[0].band = n // 2
            self.channels[0].route = "Gray"
        self._readout_band = self.channels[0].band
        self._changed()

    def replace_cube(self, cube: Cube):
        """Show a processed cube of the same size, keeping annotations, overlays and display settings."""
        if cube.shape != self.cube.shape:
            raise ValueError("A replacement cube must have the same shape.")
        self.cube = cube
        self._energy = None
        self._svd = None
        self._label_cache.clear()
        for ch in self.channels:
            ch.cache = None
        self._changed()

    def _changed(self):
        self._composite = None
        self.version += 1
        self.image_version += 1

    def info(self) -> dict:
        c = self.cube
        if c is None:
            return {"loaded": False, "title": "hsview"}
        messages = [c.metadata[k] for k in ("wavelength_warning", "reference_warning") if k in c.metadata]
        title = f"{c.name} — hsview"
        if self.store.modified and len(self.store):
            title = "• " + title
        return {
            "loaded": True, "name": c.name, "rows": int(c.shape[0]), "cols": int(c.shape[1]),
            "bands": int(c.n_bands), "has_wavelengths": c.wavelengths is not None,
            "axis": _floats(c.band_axis), "axis_label": c.band_axis_label,
            "band_labels": [c.band_label(b) for b in range(c.n_bands)],
            "messages": messages, "title": title, "history": len(self.cube_history),
            "processed": bool(self.cube_history or c.metadata.get("analysis")),
            "safe_name": _safe_name(c.name),
        }

    def state(self) -> dict:
        """What the page needs to enable menu items and show the window title."""
        has = self.cube is not None
        ref = self.reference.img
        return {
            "has_cube": has,
            "title": self.info()["title"],
            "n_annotations": len(self.store),
            "can_undo": self.store.can_undo,
            "can_redo": self.store.can_redo,
            "modified": bool(self.store.modified and len(self.store)),
            "history": len(self.cube_history),
            "has_wavelengths": has and self.cube.wavelengths is not None,
            "has_reference": ref is not None,
            "reference_rgb": ref is not None and ref.ndim == 3,
            "has_overlays": bool(self.overlays),
            "current_overlay": self.current_overlay,
            "has_segmentation": self.segmentation_overlay() is not None,
            "annotation_name": self.annotation_name,
        }

    @staticmethod
    def database_file(image_id: int, image_dir: str) -> str:
        """The file name of a camera database id, relative to the directory holding image_database.db."""
        return str(resolve_image_id(int(image_id), image_dir).relative_to(image_dir))

    def save_cube(self, path: str) -> str:
        save_cube(path, self.cube)
        return path

    # --------------------------------------------------------- display
    def display_state(self) -> dict:
        cube = self.cube
        chans = []
        for ch in self.channels:
            chans.append({"source": ch.source, "route": ch.route, "band": ch.band, "width": ch.width,
                          "spec": ch.spec, "sum_mode": ch.sum_mode, "filter_name": ch.filter_name,
                          "has_filter": ch.filter is not None, "info": self._channel_info(ch)})
        ref = self.reference
        return {
            "channels": chans, "n_bands": cube.n_bands if cube else 1,
            "reference": {"name": ref.name if ref.img is not None else "None", "route": ref.route,
                          "opacity": ref.opacity, "has": ref.img is not None,
                          "rgb": ref.img is not None and ref.img.ndim == 3},
            "scaling": self.scaling, "contrast": self.contrast, "bias": self.bias,
            "band_marker": self.band_marker(), "readout": self.band_readout(),
        }

    def _bands(self, ch: Channel) -> range:
        return band_window(ch.band, ch.width, self.cube.n_bands if self.cube else 1)

    def _channel_info(self, ch: Channel) -> str:
        cube = self.cube
        if cube is None:
            return ""
        if ch.source == "Band":
            text = cube.band_label(ch.band)
            bands = self._bands(ch)
            if len(bands) > 1:
                text += f", mean of {len(bands)} bands {bands[0]}–{bands[-1]}"
            return text
        if ch.source == "Band sum":
            try:
                n = len(parse_band_spec(ch.spec, cube)) if ch.spec.strip() else 0
                return f"{ch.sum_mode} of {n} bands" if n else "Enter bands to combine"
            except ValueError as exc:
                return str(exc)
        if ch.filter is None:
            return ""
        return f"filter {ch.filter.shape[1] if ch.filter.ndim == 2 else 1} component(s)"

    def band_marker(self):
        """Band to mark on the spectrum plot: the first routed channel showing a single band."""
        ch = next((c for c in self.channels if c.route != "None" and c.source == "Band"), None)
        return ch.band if ch is not None else None

    def band_readout(self) -> str:
        if self.cube is None:
            return "Band: --"
        return self.cube.band_label(min(self._readout_band, self.cube.n_bands - 1))

    def set_channel(self, c: int, changes: dict) -> dict:
        """Change one band channel (keys: source, route, band, width, spec, sum_mode); returns display_state()."""
        ch = self.channels[int(c)]
        n = self.cube.n_bands if self.cube else 1
        route = changes.pop("route", None)
        if "source" in changes:
            src = changes.pop("source")
            if src not in SOURCES:
                raise ValueError(src)
            if src == "Filter" and ch.filter is None:
                raise LoadError("Load a filter first")
            ch.source = src
        if "band" in changes:
            ch.band = max(0, min(n - 1, int(changes.pop("band"))))
            self._readout_band = ch.band
        if "width" in changes:
            ch.width = max(1, min(n, int(changes.pop("width"))))
        if "spec" in changes:
            ch.spec = str(changes.pop("spec"))
        if "sum_mode" in changes:
            ch.sum_mode = changes.pop("sum_mode")
        if changes:
            raise ValueError(f"Unknown channel settings: {', '.join(changes)}")
        ch.cache = None
        if route is not None and route != ch.route:
            ch.route = route
            self._route_changed(ch)
        self._changed()
        return self.display_state()

    def set_reference(self, changes: dict) -> dict:
        ref = self.reference
        if "opacity" in changes:
            ref.opacity = max(0, min(100, int(changes["opacity"])))
        route = changes.get("route")
        if route is not None and route != ref.route:
            if ref.img is None and route != "None":
                raise LoadError("Load a reference image first")
            ref.route = route
            self._route_changed(ref)
        self._changed()
        return self.display_state()

    def set_view(self, changes: dict) -> dict:
        """Scaling, contrast and bias."""
        if "scaling" in changes:
            if changes["scaling"] not in SCALINGS:
                raise ValueError(changes["scaling"])
            self.scaling = changes["scaling"]
            self._composite = None
        if "contrast" in changes:
            self.contrast = max(0, min(89, int(changes["contrast"])))
        if "bias" in changes:
            self.bias = max(-128, min(127, int(changes["bias"])))
        self.version += 1
        self.image_version += 1
        return self.display_state()

    def step_contrast(self, delta: int) -> dict:
        return self.set_view({"contrast": self.contrast + int(delta)})

    def step_bias(self, delta: int) -> dict:
        return self.set_view({"bias": self.bias + int(delta)})

    def reset_contrast(self) -> dict:
        return self.set_view({"contrast": DEFAULT_CONTRAST, "bias": DEFAULT_BIAS})

    def _route_changed(self, source):
        """Keep routing consistent: one source per colour, Gray/Show is exclusive, never nothing."""
        sources = [*self.channels, self.reference]
        choice = source.route
        exclusive = choice in ("Gray", "Show")
        if choice not in ("None", "Overlay"):
            for other in sources:
                if other is source:
                    continue
                o = other.route
                if o == "Overlay" and choice != "Show":
                    continue  # the reference's Overlay blends over whatever the bands show
                if exclusive or o in ("Gray", "Show") or o == choice:
                    other.route = "None"
        if self.reference.route != "Show" and all(c.route == "None" for c in self.channels):
            self.channels[0].route = "Gray"

    def step_band(self, delta: int) -> dict:
        """Move the band of the first channel showing a single band."""
        for ch in self.channels:
            if ch.route != "None" and ch.source == "Band":
                return self.set_channel(self.channels.index(ch), {"band": ch.band + int(delta)})
        ch = self.channels[0]
        state = self.set_channel(0, {"source": "Band", "band": ch.band + int(delta)})
        if all(c.route == "None" for c in self.channels) and self.reference.route == "None":
            state = self.set_channel(0, {"route": "Gray"})
        return state

    def show_rgb(self, specs, mode: str = "sum") -> dict:
        """Three band selections for red, green, blue ('' leaves a channel empty)."""
        for ch in self.channels:
            ch.route = "None"
        self.reference.route = "None"
        for color, (ch, spec) in enumerate(zip(self.channels, specs), start=1):
            if spec:
                ch.source, ch.spec, ch.sum_mode, ch.cache = "Band sum", str(spec), mode, None
                ch.route = ROUTES[color]
        if all(c.route == "None" for c in self.channels):
            self.channels[0].route = "Gray"
        self._changed()
        return self.display_state()

    def true_colour(self) -> dict:
        if self.cube is None or self.cube.wavelengths is None:
            return self.display_state()
        return self.show_rgb(list(TRUE_COLOUR))

    def load_filter(self, c: int, path: str, name: str) -> dict:
        filt = load_filter(path, self.cube.n_bands)
        ch = self.channels[int(c)]
        ch.filter, ch.filter_name, ch.source, ch.cache = filt, name, "Filter", None
        if ch.route == "None":
            ch.route = "Gray"
            self._route_changed(ch)
        self._changed()
        return self.display_state()

    def load_reference(self, path: str, name: str) -> dict:
        img = read_image(path)
        cube = self.cube
        if cube is not None and img.shape[:2] != cube.shape[:2]:
            raise LoadError(f"The image is {img.shape[1]}×{img.shape[0]} but the cube is "
                            f"{cube.shape[1]}×{cube.shape[0]}.")
        self.reference.img, self.reference.name = img, name
        self.reference.route = "Show"
        self._route_changed(self.reference)
        self._changed()
        return self.display_state()

    def reverse_rgb(self) -> dict:
        ref = self.reference
        if ref.img is not None and ref.img.ndim == 3:
            ref.img = np.ascontiguousarray(ref.img[:, :, ::-1])
            self._changed()
        return self.display_state()

    def _energy_image(self):
        if self._energy is None:
            self._energy = spectral_energy(self.cube)
        return self._energy

    def _channel_image(self, ch: Channel):
        cube = self.cube
        if ch.cache is None:
            if ch.source == "Band":
                bands = self._bands(ch)
                if len(bands) == 1:
                    ch.cache = np.asarray(cube.data[:, :, bands[0]], dtype=np.float64)
                else:
                    ch.cache = band_image(cube, bands, "mean")
            elif ch.source == "Band sum":
                if not ch.spec.strip():
                    return None
                try:
                    bands = parse_band_spec(ch.spec, cube)
                except ValueError:
                    return None
                ch.cache = band_image(cube, bands, ch.sum_mode)
            elif ch.filter is not None:
                ch.cache = apply_filter(cube, ch.filter, self._energy_image() if ch.filter.ndim == 2 else None)
        return ch.cache

    def composite(self):
        """The uint8 display image before contrast/bias, like DisplayPanel.composite."""
        if self.cube is None:
            return None
        ref = self.reference
        ref_route = ref.route if ref.img is not None else "None"
        if ref_route == "Show":
            return compose(gray=ref.img, scaling=self.scaling)
        base = self._band_composite()
        if ref_route == "Overlay":
            over = compose(gray=ref.img, scaling=self.scaling)
            if base is None:
                return over
            alpha = ref.opacity / 100
            if over.ndim == 2:
                over = over[:, :, None]
            if base.ndim == 2:
                base = base[:, :, None]
            out = np.rint((1 - alpha) * base + alpha * over).astype(np.uint8)
            return np.ascontiguousarray(out if out.shape[2] == 3 else out[:, :, 0])
        return base

    def _band_composite(self):
        rgb = {}
        for ch in self.channels:
            if ch.route == "None":
                continue
            img = self._channel_image(ch)
            if img is None:
                continue
            if ch.route == "Gray":
                return compose(gray=img, scaling=self.scaling)
            rgb[ROUTES.index(ch.route) - 1] = img
        return compose(rgb=rgb, scaling=self.scaling)

    def render(self) -> dict | None:
        """The display image as RGBA bytes: ``{"width", "height", "rgba"}``, or None when there is nothing."""
        if self._composite is None:
            self._composite = self.composite()
        img = self._composite
        if img is None:
            return None
        if self.contrast != DEFAULT_CONTRAST or self.bias != DEFAULT_BIAS:
            img = contrast_bias_lut(self.contrast, self.bias)[img]
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        rgba = np.empty(img.shape[:2] + (4,), np.uint8)
        rgba[..., :3] = img[..., :3]
        rgba[..., 3] = 255
        return {"width": int(img.shape[1]), "height": int(img.shape[0]), "rgba": rgba.tobytes()}

    # ---------------------------------------------------------- overlays
    def add_overlay_file(self, path: str, name: str) -> dict:
        """An image or array file; hsview segmentation files keep their label colours and metadata."""
        stem = Path(name).stem
        seg = None
        if Path(path).suffix.lower() in (".npz", ".pkl", ".pickle", ".p"):
            try:
                seg = segmentation_from_dict(read_segmentation(path))
            except LoadError:
                seg = None
        if seg is not None:
            return self.add_segmentation(seg, stem)
        return self.add_overlay_array(load_overlay(path), stem)

    def add_segmentation(self, seg, name: str | None = None) -> dict:
        info = self.add_overlay_array(seg.labels, name or seg.name, as_labels=True, colors=seg.label_colors)
        ov = self.overlays[info["key"]]
        ov.segmentation = seg
        return self._overlay_info(ov)

    def add_overlay_array(self, arr, name: str, as_labels: bool | None = None, colors: dict | None = None) -> dict:
        arr = np.asarray(arr)
        if self.cube is not None and arr.shape[:2] != tuple(self.cube.shape[:2]):
            raise LoadError(f"Overlay '{name}' is {arr.shape[1]}×{arr.shape[0]} but the image is "
                            f"{self.cube.shape[1]}×{self.cube.shape[0]}.")
        if as_labels is None:
            as_labels = is_label_image(arr)
        self._overlay_counter += 1
        ov = Overlay(f"ov{self._overlay_counter}", name, arr, as_labels, colors=colors)
        self.overlays[ov.key] = ov
        self.current_overlay = ov.key
        self.version += 1
        return self._overlay_info(ov)

    def _overlay_info(self, ov: Overlay) -> dict:
        return {"key": ov.key, "name": ov.name, "labels": ov.as_labels, "visible": ov.visible,
                "opacity": ov.opacity, "segmentation": ov.segmentation is not None}

    def overlay_list(self) -> list:
        return [self._overlay_info(ov) for ov in self.overlays.values()]

    def overlay_rgba(self, key: str) -> dict:
        ov = self.overlays[key]
        rgba = colorize(ov.data, ov.as_labels, ov.colors)
        return {"width": int(rgba.shape[1]), "height": int(rgba.shape[0]),
                "rgba": np.ascontiguousarray(rgba).tobytes()}

    def select_overlay(self, key):
        self.current_overlay = key if key in self.overlays else None
        self.version += 1

    def set_overlay_visible(self, key, visible: bool):
        if key in self.overlays:
            self.overlays[key].visible = bool(visible)
            self._label_cache.pop(key, None)
            self.version += 1

    def set_all_overlays_visible(self, visible: bool):
        for key in self.overlays:
            self.set_overlay_visible(key, visible)

    def set_overlay_opacity(self, key, opacity: float):
        if key in self.overlays:
            self.overlays[key].opacity = max(0.0, min(1.0, float(opacity)))

    def remove_overlay(self, key):
        self.overlays.pop(key, None)
        self._label_cache.pop(key, None)
        if self.current_overlay == key:
            self.current_overlay = next(reversed(self.overlays), None) if self.overlays else None
        self.version += 1

    def clear_overlays(self):
        self.overlays.clear()
        self._label_cache.clear()
        self.current_overlay = None
        self.version += 1

    def label_overlay(self) -> Overlay | None:
        """The selected overlay if it is a visible segmentation, else the last visible one (spectra panel)."""
        cur = self.overlays.get(self.current_overlay)
        if cur is not None and cur.as_labels and cur.visible:
            return cur
        for ov in reversed(self.overlays.values()):
            if ov.as_labels and ov.visible:
                return ov
        return None

    def segmentation_overlay(self) -> Overlay | None:
        """The selected overlay if it is a label image, else the most recent visible (or any) label overlay."""
        cur = self.overlays.get(self.current_overlay)
        if cur is not None and cur.as_labels:
            return cur
        labels = [ov for ov in self.overlays.values() if ov.as_labels]
        visible = [ov for ov in labels if ov.visible]
        return (visible or labels or [None])[-1]

    # ------------------------------------------------------- annotations
    def annotations(self) -> list:
        return [{"id": a.id, "kind": a.kind, "cls": a.cls, "coords": a.coords.tolist(), "label": a.label}
                for a in self.store]

    def add_annotation(self, kind: str, cls: int, coords) -> int:
        coords = np.asarray(coords, dtype=float).reshape(-1, 2)
        if kind == "point":
            coords = np.floor(coords)
        ann = self.store.add(kind, int(cls), coords)
        self.version += 1
        return ann.id

    def remove_annotations(self, ids) -> int:
        n = self.store.remove([int(i) for i in ids])
        self.version += 1
        return n

    def move_annotations(self, ids, dx: float, dy: float):
        """Shift annotations by (dx, dy) as one undo step; points snap to the nearest pixel."""
        changes = {}
        for i in ids:
            ann = self.store.get(int(i))
            if ann is None:
                continue
            coords = ann.coords + np.array([dx, dy])
            if ann.kind == "point":
                coords = np.floor(coords + 0.5)
            changes[ann.id] = {"coords": coords}
        self.store.update(changes)
        self.version += 1

    def recolor_annotations(self, ids, cls: int):
        self.store.update({int(i): {"cls": int(cls)} for i in ids})
        self.version += 1

    def undo(self):
        self.store.undo()
        self.version += 1

    def redo(self):
        self.store.redo()
        self.version += 1

    def clear_annotations(self):
        self.store.clear()
        self.version += 1

    def load_annotations(self, path: str, name: str) -> int:
        items = ann_io.load_annotations(path)
        self.store.replace(items)
        self.store.modified = False
        # Annotations taken from a segmentation file must not be saved over that file.
        self.annotation_name = None if is_segmentation_file(path) else name
        self.version += 1
        return len(items)

    def save_annotations(self, path: str, name: str) -> int:
        ann_io.save_annotations(path, self.store, self.cube)
        self.annotation_name = name
        self.store.modified = False
        self.version += 1
        return len(self.store)

    def save_spectra(self, path: str):
        ann_io.save_annotation_spectra(path, self.cube, self.store)

    def export_csv(self, path: str) -> int:
        return ann_io.export_spectra_csv(path, self.cube, self.store)

    # ------------------------------------------------------------ spectra
    def cursor_spectrum(self, x: float, y: float):
        if self.cube is None:
            return None
        s = pixel_spectrum(self.cube, x, y)
        return None if s is None else _floats(s)

    def spectra_series(self, show_std: bool = True, by_class: bool = False, show_labels: bool = False) -> list:
        """Lines for the Spectra panel (not the cursor), in the order and styles the Qt panel draws them."""
        cube = self.cube
        if cube is None:
            return []
        out = []
        if show_labels:
            ov = self.label_overlay()
            if ov is not None and ov.data.shape == cube.shape[:2]:
                if ov.key not in self._label_cache:
                    self._label_cache[ov.key] = label_spectra(cube, ov.data)
                for v, spec in self._label_cache[ov.key].items():
                    out.append({"y": _floats(spec), "color": _color_hex(ov.label_color(v)), "width": 1.2,
                                "dash": "dashed", "label": ov.label_text(v)})
        stats = mean_spectra(cube, self.store)
        if by_class:
            for cls in sorted({a.cls for a, *_ in stats}):
                rows = [(a, m, s, k) for a, m, s, k in stats if a.cls == cls]
                pix = sum(k for *_, k in rows)
                mean = sum(m * k for _, m, _, k in rows) / pix
                out.append({"y": _floats(mean), "color": CLASS_PLOT_COLORS[cls], "width": 2,
                            "label": f"{CLASS_NAMES[cls]} ({len(rows)} ann., n={pix})"})
        else:
            for a, mean, std, k in stats:
                col = CLASS_PLOT_COLORS[a.cls % len(CLASS_PLOT_COLORS)]
                if a.kind == "point":
                    out.append({"y": _floats(mean), "color": col, "width": 1.2,
                                "label": f"{a.label} ({int(a.coords[0][0])}, {int(a.coords[0][1])})"})
                else:
                    line = {"y": _floats(mean), "color": col, "width": 2, "label": f"{a.label} mean, n={k}"}
                    if show_std and k > 1:
                        line["lo"], line["hi"] = _floats(mean - std), _floats(mean + std)
                    out.append(line)
        return out

    def class_groups(self, which: str) -> dict:
        """Spectra for Plot → point/region spectra by class: ``{"title", "groups": [{cls, name, color, spectra}]}``."""
        groups: dict[int, list] = {}
        for a in self.store:
            if (which == "point") != (a.kind == "point"):
                continue
            px = a.pixels(self.cube)
            if len(px):
                groups.setdefault(a.cls, []).append(_floats(px.mean(axis=0)))
        return {"title": "Point spectra" if which == "point" else "Region mean spectra",
                "groups": [{"cls": c, "name": CLASS_NAMES[c], "color": CLASS_PLOT_COLORS[c], "spectra": groups[c]}
                           for c in sorted(groups)]}

    # ----------------------------------------------------------- analysis
    def svd_fit(self) -> list:
        """Fit the SVD background model of the current cube; returns each component's variance fraction."""
        self._svd = fit_svd_background(self.cube)
        return _floats(self._svd.variance_fraction)

    def svd_apply(self, k: int, add_mean: bool) -> str:
        model = self._svd or fit_svd_background(self.cube)
        result = remove_svd_background(self.cube, int(k), bool(add_mean), model)
        self.cube_history.append(self.cube)
        self.replace_cube(result)
        return result.name

    def undo_analysis(self):
        if self.cube_history:
            self.replace_cube(self.cube_history.pop())

    def restore_original_cube(self):
        if self.cube_history:
            original = self.cube_history[0]
            self.cube_history.clear()
            self.replace_cube(original)

    # ------------------------------------------------------- segmentation
    def training_summary(self) -> dict:
        """What the Segment from annotations dialog lists: annotations and training pixels per class."""
        annotations = list(self.store)
        training = training_label_image(annotations, self.cube.shape[:2])
        counts = np.bincount(training.ravel(), minlength=len(CLASS_NAMES) + 1)
        n_ann: dict[int, int] = {}
        for a in annotations:
            n_ann[a.cls] = n_ann.get(a.cls, 0) + 1
        rows = [f"{CLASS_NAMES[c]}: {n_ann.get(c, 0)} annotation(s), {counts[c + 1]} pixel(s)"
                for c in range(len(CLASS_NAMES)) if counts[c + 1] or n_ann.get(c)]
        return {"rows": rows, "n_classes": int((counts[1:] > 0).sum())}

    def _show_segmentation(self, seg) -> dict:
        for ov in self.overlays.values():  # only the newest segmentation is shown at first
            if ov.segmentation is not None and ov.visible:
                ov.visible = False
        return self.add_segmentation(seg) | {"title": seg.name}

    def segment_unsupervised(self, params: dict) -> dict:
        return self._show_segmentation(segment_unsupervised(self.cube, **params))

    def segment_supervised(self, params: dict) -> dict:
        return self._show_segmentation(segment_supervised(self.cube, list(self.store), **params))

    def save_segmentation(self, path: str) -> str:
        ov = self.segmentation_overlay()
        if ov is None:
            raise LoadError("There is no segmentation overlay to save")
        seg = ov.segmentation or segmentation_from_labels(ov.data, self.cube, ov.name, ov.colors)
        save_segmentation(path, seg)
        return ov.name

    def segmentation_name(self) -> str | None:
        ov = self.segmentation_overlay()
        return None if ov is None else ov.name


def to_json(obj) -> str:
    """For debugging from the browser console."""
    return json.dumps(obj, default=str)

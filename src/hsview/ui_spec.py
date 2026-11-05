"""The application's commands, menus, default toolbars and icons, without any GUI code.

Both front ends are built from these tables: the Qt app
(:mod:`hsview.gui`) and the browser version (``web/``, which runs this
package in Pyodide). Adding a command here gives it a menu item, a
toolbar-placeable button and a configurable shortcut in both.
"""
from __future__ import annotations

from dataclasses import dataclass

from .annotations import CLASS_NAMES

@dataclass(frozen=True)
class ActionSpec:
    id: str
    text: str
    icon: str = ""
    shortcut: str = ""
    tip: str = ""
    category: str = "General"
    checkable: bool = False
    group: str = ""  # actions sharing a group are mutually exclusive


BASE_SPECS = [
    # File
    ActionSpec("open_cube", "Open cube…", "open", "Ctrl+O", "Open a hyperspectral cube or image", "File"),
    ActionSpec("open_demo", "Synthetic demo cube", "", "", "A small made-up cube with a reference photo and labels", "File"),
    ActionSpec("open_pavia", "Pavia University (download 35 MB)", "", "",
               "ROSIS airborne scene, 610 × 340 pixels, 103 bands, with its 9-class ground truth. "
               "Downloaded once, then opened from the cache", "File"),
    ActionSpec("open_by_id", "Open by database id…", "", "", "Open a file from the camera image database", "File"),
    ActionSpec("open_reference", "Open reference image…", "image", "", "Load a photo of the scene to show or blend", "File"),
    ActionSpec("add_overlay", "Add overlay…", "layers", "", "Add a segmentation or processed image on top", "File"),
    ActionSpec("load_filter", "Load filter into channel 1…", "filter", "", "Show channel 1 through a spectral filter", "File"),
    ActionSpec("load_annotations", "Load annotations…", "load-annotations", "Ctrl+K", "Load points and regions (.pkl or .json)", "File"),
    ActionSpec("save_annotations", "Save annotations", "save", "Ctrl+S", "Save points and regions", "File"),
    ActionSpec("save_annotations_as", "Save annotations as…", "", "Ctrl+Shift+S", "", "File"),
    ActionSpec("save_spectra", "Save annotation spectra…", "spectra-file", "Ctrl+D", "Pickle of all spectra under the annotations", "File"),
    ActionSpec("export_csv", "Export spectra to CSV…", "table", "", "One column per annotation, one row per band", "File"),
    ActionSpec("save_cube", "Save cube as…", "save-cube", "",
               "Save the current (possibly processed) cube as a pickle with ref, wavelengths and lbl", "File"),
    ActionSpec("quit", "Quit", "quit", "Ctrl+Q", "", "File"),
    # Edit
    ActionSpec("undo", "Undo", "undo", "Ctrl+Z", "Undo the last annotation change", "Edit"),
    ActionSpec("redo", "Redo", "redo", "Ctrl+Shift+Z", "", "Edit"),
    ActionSpec("delete_selected", "Delete selected", "delete", "Del", "Delete the selected annotations", "Edit"),
    ActionSpec("select_all", "Select all annotations", "", "Ctrl+A", "", "Edit"),
    ActionSpec("clear_annotations", "Clear annotations…", "trash", "", "Remove every annotation", "Edit"),
    ActionSpec("finish_polygon", "Finish polygon", "check", "", "Close the polygon being drawn (also Enter)", "Edit"),
    ActionSpec("cancel_drawing", "Cancel drawing", "cancel", "", "Discard the shape being drawn (also Esc)", "Edit"),
    ActionSpec("customize", "Customize toolbars and shortcuts…", "settings", "", "", "Edit"),
    # Modes
    ActionSpec("mode_navigate", "Navigate", "hand", "N", "Pan by dragging, zoom with the wheel", "Mode", True, "mode"),
    ActionSpec("mode_select", "Select / move", "pointer", "S", "Select, move and delete annotations", "Mode", True, "mode"),
    ActionSpec("mode_point", "Points", "point", "P", "Click to add points; right-click removes", "Mode", True, "mode"),
    ActionSpec("mode_rect", "Rectangles", "rect", "R", "Drag to add rectangles; right-click removes", "Mode", True, "mode"),
    ActionSpec("mode_polygon", "Polygons", "polygon", "G",
               "Click vertices, double-click or Enter to close, Esc cancels", "Mode", True, "mode"),
    # View
    ActionSpec("zoom_in", "Zoom in", "zoom-in", "Ctrl+=", "", "View"),
    ActionSpec("zoom_out", "Zoom out", "zoom-out", "Ctrl+-", "", "View"),
    ActionSpec("zoom_fit", "Fit to window", "zoom-fit", "Ctrl+0", "", "View"),
    ActionSpec("zoom_actual", "Actual pixels", "zoom-actual", "Ctrl+1", "One image pixel per screen pixel", "View"),
    ActionSpec("prev_band", "Previous band", "prev", "[", "", "Display"),
    ActionSpec("next_band", "Next band", "next", "]", "", "Display"),
    ActionSpec("true_colour", "True colour", "palette", "", "RGB from 620–670, 530–570 and 450–490 nm", "Display"),
    ActionSpec("reverse_rgb", "Reverse reference RGB", "swap", "", "Swap red and blue of the reference image", "Reference"),
    ActionSpec("show_rgb", "RGB from bands…", "rgb", "", "Build red, green and blue from band ranges", "Display"),
    ActionSpec("reset_contrast", "Reset contrast", "contrast", "", "", "Display"),
    ActionSpec("contrast_up", "Increase contrast", "contrast-up", "", "", "Display"),
    ActionSpec("contrast_down", "Decrease contrast", "contrast-down", "", "", "Display"),
    ActionSpec("bias_up", "Brighter (bias +)", "bias-up", "", "", "Display"),
    ActionSpec("bias_down", "Darker (bias −)", "bias-down", "", "", "Display"),
    ActionSpec("scaling_global", "Shared min–max", "", "", "Scale all channels by the same range", "Display", True, "scaling"),
    ActionSpec("scaling_channel", "Per-channel min–max", "", "", "Scale each channel by its own range", "Display", True, "scaling"),
    ActionSpec("scaling_percentile", "Per-channel 2–98 %", "", "", "Clip each channel to its 2–98 % range", "Display", True, "scaling"),
    ActionSpec("ref_route_none", "Hide reference", "", "", "Do not use the reference image", "Reference", True, "ref_route"),
    ActionSpec("ref_route_show", "Show reference", "image", "", "Replace the band display with the reference image",
               "Reference", True, "ref_route"),
    ActionSpec("ref_route_overlay", "Overlay reference", "", "", "Blend the reference image over the bands",
               "Reference", True, "ref_route"),
    ActionSpec("ref_opacity", "Reference opacity…", "", "", "How strongly the overlaid reference covers the bands", "Reference"),
    ActionSpec("toggle_display", "Display panel", "sliders", "", "", "View", True),
    ActionSpec("toggle_spectra", "Spectra panel", "spectrum", "", "", "View", True),
    ActionSpec("toggle_overlays", "Overlays panel", "eye", "", "", "View", True),
    # Plot
    ActionSpec("plot_points", "Plot point spectra by class", "plot-points", "", "", "Plot"),
    ActionSpec("plot_regions", "Plot region spectra by class", "plot-regions", "", "Mean spectrum of each rectangle and polygon", "Plot"),
    ActionSpec("plot_separate", "One window per class", "windows", "",
               "Plot each annotation class in its own window instead of one window of subplots", "Plot", True),
    ActionSpec("close_plots", "Close plot windows", "close-windows", "", "", "Plot"),
    ActionSpec("spectra_cursor", "Show cursor spectrum", "", "", "Show the spectrum of the pixel under the mouse", "Plot", True),
    ActionSpec("spectra_std", "Shade ±1σ", "", "", "Shade one standard deviation around region means", "Plot", True),
    ActionSpec("spectra_by_class", "Spectra by class", "", "",
               "One mean spectrum per annotation class instead of one line per annotation", "Plot", True),
    ActionSpec("spectra_labels", "Overlay class spectra", "", "",
               "Mean spectrum of each class in the selected segmentation overlay", "Plot", True),
    # Overlays
    ActionSpec("remove_overlay", "Remove selected overlay", "", "", "", "Overlays"),
    ActionSpec("clear_overlays", "Remove all overlays", "", "", "", "Overlays"),
    ActionSpec("show_overlays", "Show all overlays", "eye", "", "", "Overlays"),
    ActionSpec("hide_overlays", "Hide all overlays", "", "", "", "Overlays"),
    ActionSpec("overlay_opacity", "Selected overlay opacity…", "", "", "", "Overlays"),
    # Analysis (the analyses themselves are added from gui.analysis.ANALYSES)
    ActionSpec("analysis_undo", "Undo last analysis", "undo", "", "Go back to the cube before the last analysis", "Analysis"),
    ActionSpec("analysis_restore", "Restore original cube", "restore", "", "Discard every analysis applied to the cube",
               "Analysis"),
    # Segment
    ActionSpec("segment_unsupervised", "Segment (unsupervised)…", "segment", "",
               "Split the current cube into regions of similar spectra with k-means clustering", "Segment"),
    ActionSpec("segment_supervised", "Segment from annotations…", "segment-supervised", "",
               "Classify every pixel into the annotated classes (points, rectangles and polygons)", "Segment"),
    ActionSpec("save_segmentation", "Save segmentation…", "save-segmentation", "",
               "Save the selected segmentation overlay (.pkl or .npz) with its annotations and metadata", "Segment"),
    ActionSpec("open_segmentation", "Open segmentation…", "layers", "",
               "Show a saved segmentation (or any label image) as an overlay", "Segment"),
    # Help
    ActionSpec("about", "About hsview", "info", "", "", "Help"),
]


def channel_specs(n_channels: int = 3) -> list[ActionSpec]:
    """The per-channel commands of the Display menu."""
    return [
        ActionSpec(spec_id, text, icon_name, "", tip, f"Channel {c + 1}", checkable, group)
        for c in range(n_channels)
        for spec_id, text, icon_name, tip, checkable, group in [
            (f"ch{c}_band", "Set band…", "", "Show a single band (optionally averaged over a window)", False, ""),
            (f"ch{c}_width", "Band window width…", "", "Average this many bands centred on the selected band", False, ""),
            (f"ch{c}_sum", "Band sum or mean…", "", "Combine bands such as 10-20, 35 or 650-700nm", False, ""),
            (f"ch{c}_filter", "Load filter…", "filter", "Show this channel through a spectral filter", False, ""),
        ] + [(f"ch{c}_route_{r.lower()}", f"→ {r}", "", f"Send channel {c + 1} to {r.lower()}", True, f"ch{c}_route")
             for r in ROUTES]
    ]


def class_specs() -> list[ActionSpec]:
    return [ActionSpec(f"class_{i}", f"Class {i}: {name}", "", str(i + 1), f"Draw new annotations as {name}", "Class",
                       True, "class") for i, name in enumerate(CLASS_NAMES)]


def build_specs(analyses=(), n_channels: int = 3) -> list[ActionSpec]:
    """Every command. ``analyses`` are objects with ``id``, ``text``, ``icon`` and ``tip`` (the Analysis menu)."""
    return (BASE_SPECS + [ActionSpec(a.id, a.text, a.icon, "", a.tip, "Analysis") for a in analyses]
            + channel_specs(n_channels) + class_specs())

SEPARATOR = "|"
SPACER = "<spacer>"

# Items that are widgets rather than buttons.
WIDGET_ITEMS = {
    "class_selector": ("Class selector", "Class"),
    "band_readout": ("Band / wavelength readout", "View"),
}



# ------------------------------------------------------------------ display
ROUTES = ("None", "Red", "Green", "Blue", "Gray")
REF_ROUTES = ("None", "Show", "Overlay")
SOURCES = ("Band", "Band sum", "Filter")
SCALING_LABELS = {"global": "Shared min–max", "channel": "Per-channel min–max", "percentile": "Per-channel 2–98 %"}
DEFAULT_CONTRAST, DEFAULT_BIAS = 45, 0
CONTRAST_STEP, BIAS_STEP = 5, 8
TRUE_COLOUR = ("620-670nm", "530-570nm", "450-490nm")

# Majority filter sizes offered by the segmentation dialogs.
SMOOTHING = [(0, "Off"), (3, "3 × 3"), (5, "5 × 5"), (7, "7 × 7"), (9, "9 × 9")]


# -------------------------------------------------------------------- menus
def menu_layout(analyses=(), n_channels: int = 3) -> dict:
    """Menu bar layout: lists of action ids, None for a separator, (title, [...]) for a submenu.

    The Overlays menu gets one submenu per overlay after these items, and the
    Toolbars menu is filled from the toolbar configuration when it opens.
    """
    channels = [(f"Band channel {c + 1}", [f"ch{c}_band", f"ch{c}_width", f"ch{c}_sum", f"ch{c}_filter", None]
                 + [f"ch{c}_route_{x.lower()}" for x in ROUTES]) for c in range(n_channels)]
    return {
        "&File": ["open_cube", ("Open example", ["open_demo", "open_pavia"]), "open_by_id", "open_reference", "add_overlay", "load_filter", None,
                  "load_annotations", "save_annotations", "save_annotations_as", None,
                  "save_spectra", "export_csv", "save_cube", "save_segmentation", None, "quit"],
        "&Edit": ["undo", "redo", None, "select_all", "delete_selected", "clear_annotations", None,
                  "finish_polygon", "cancel_drawing", None, "customize"],
        "&Mode": ["mode_navigate", "mode_select", "mode_point", "mode_rect", "mode_polygon"],
        "&Class": [f"class_{i}" for i in range(len(CLASS_NAMES))],
        "&View": ["zoom_in", "zoom_out", "zoom_fit", "zoom_actual", None,
                  "toggle_display", "toggle_overlays", "toggle_spectra"],
        "&Display": ["prev_band", "next_band", "true_colour", "show_rgb", None, *channels,
                     ("Reference image", ["open_reference", "reverse_rgb", "ref_opacity", None,
                                          "ref_route_none", "ref_route_show", "ref_route_overlay"]),
                     None, ("Scaling", ["scaling_global", "scaling_channel", "scaling_percentile"]),
                     ("Contrast", ["contrast_up", "contrast_down", "bias_up", "bias_down", None,
                                   "reset_contrast"])],
        "&Overlays": ["add_overlay", "remove_overlay", "clear_overlays", None, "show_overlays", "hide_overlays",
                      "overlay_opacity"],
        "&Plot": ["plot_points", "plot_regions", None, "plot_separate", "close_plots", None,
                  "spectra_cursor", "spectra_std", "spectra_by_class", "spectra_labels"],
        "&Analysis": [cmd.id for cmd in analyses] + [None, "analysis_undo", "analysis_restore"],
        "&Segment": ["segment_unsupervised", "segment_supervised", None, "save_segmentation",
                     "open_segmentation"],
        "&Toolbars": [],
        "&Help": ["about"],
    }


# ----------------------------------------------------------------- toolbars
DEFAULT_TOOLBAR_CONFIG = {
    "icon_size": 22,
    "style": "icon",
    "toolbars": [
        {"name": "File", "visible": True, "items": [
            "open_cube", "open_reference", "add_overlay", SEPARATOR,
            "load_annotations", "save_annotations", "save_spectra", "export_csv", "save_cube"]},
        {"name": "Annotate", "visible": True, "items": [
            "mode_navigate", "mode_select", "mode_point", "mode_rect", "mode_polygon", SEPARATOR,
            "class_selector", SEPARATOR, "undo", "redo", "delete_selected", "clear_annotations"]},
        {"name": "View", "visible": True, "items": [
            "zoom_in", "zoom_out", "zoom_fit", "zoom_actual", SEPARATOR,
            "prev_band", "next_band", "band_readout", SEPARATOR,
            "true_colour", "reset_contrast", SEPARATOR, "plot_points", "plot_regions", "plot_separate"]},
        {"name": "Analysis", "visible": True, "items": [
            "analysis_svd_background", "analysis_undo", "analysis_restore", SEPARATOR,
            "segment_unsupervised", "segment_supervised", "save_segmentation"]},
    ],
}

# Button styles (key -> label) and icon sizes offered by the Toolbars menu.
STYLE_LABELS = {"icon": "Icons only", "text": "Text only", "beside": "Text beside icons",
                "under": "Text under icons"}
ICON_SIZES = (16, 20, 22, 24, 28, 32, 40, 48)


# -------------------------------------------------------------------- icons
# 24x24 line drawings in the style of the Lucide icon set; "currentColor" is the theme's text colour.
ICON_SVG = {
    "open": '<path d="M6 14l1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "filter": '<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>',
    "load-annotations": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M12 18v-6"/><path d="m9 15 3-3 3 3"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "spectra-file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="m7 17 3-4 2 2 3-5 2 3"/>',
    "hand": '<path d="M18 11V6a2 2 0 0 0-4 0v0"/><path d="M14 10V4a2 2 0 0 0-4 0v2"/><path d="M10 10.5V6a2 2 0 0 0-4 0v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>',
    "pointer": '<path d="m3 3 7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="m13 13 6 6"/>',
    "point": '<circle cx="12" cy="12" r="3" fill="currentColor"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
    "rect": '<rect x="4" y="6" width="16" height="12" rx="1"/><circle cx="4" cy="6" r="1.6" fill="currentColor"/><circle cx="20" cy="18" r="1.6" fill="currentColor"/>',
    "polygon": '<path d="M12 3 21 9.5 17.5 20h-11L3 9.5z"/><circle cx="12" cy="3" r="1.6" fill="currentColor"/><circle cx="21" cy="9.5" r="1.6" fill="currentColor"/><circle cx="3" cy="9.5" r="1.6" fill="currentColor"/>',
    "zoom-in": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35M11 8v6M8 11h6"/>',
    "zoom-out": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35M8 11h6"/>',
    "zoom-fit": '<path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "zoom-actual": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7.5 9 9 8v8M15 9l1.5-1v8"/><circle cx="12" cy="10.5" r=".6" fill="currentColor"/><circle cx="12" cy="13.5" r=".6" fill="currentColor"/>',
    "undo": '<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>',
    "redo": '<path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3l3 2.7"/>',
    "trash": '<path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6"/>',
    "delete": '<path d="M20 5H9l-7 7 7 7h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2z"/><path d="m18 9-6 6M12 9l6 6"/>',
    "plot-points": '<path d="M3 3v18h18"/><circle cx="8" cy="15" r="1.3" fill="currentColor"/><circle cx="12" cy="9" r="1.3" fill="currentColor"/><circle cx="16" cy="13" r="1.3" fill="currentColor"/><circle cx="19" cy="6" r="1.3" fill="currentColor"/>',
    "plot-regions": '<path d="M3 3v18h18"/><path d="M7 16c2-6 4-8 6-4s3 2 6-5" /><path d="M7 19c2-4 4-5 6-2s3 1 6-3" opacity=".5"/>',
    "swap": '<path d="M8 3 4 7l4 4M4 7h16M16 21l4-4-4-4M20 17H4"/>',
    "contrast": '<circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20z" fill="currentColor"/>',
    "palette": '<circle cx="13.5" cy="6.5" r="1.2" fill="currentColor"/><circle cx="17.5" cy="10.5" r="1.2" fill="currentColor"/><circle cx="8.5" cy="7.5" r="1.2" fill="currentColor"/><circle cx="6.5" cy="12.5" r="1.2" fill="currentColor"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.93 0 1.65-.75 1.65-1.69 0-.44-.18-.84-.44-1.13-.29-.29-.44-.65-.44-1.13a1.64 1.64 0 0 1 1.67-1.67h2c3.05 0 5.56-2.5 5.56-5.55C21.97 6.01 17.46 2 12 2z"/>',
    "spectrum": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "settings": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "quit": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "prev": '<path d="m15 18-6-6 6-6"/>',
    "next": '<path d="m9 18 6-6-6-6"/>',
    "windows": '<rect x="2" y="7" width="13" height="11" rx="1.5"/><path d="M2 10h13"/><path d="M7 7V5.5A1.5 1.5 0 0 1 8.5 4h12A1.5 1.5 0 0 1 22 5.5v9a1.5 1.5 0 0 1-1.5 1.5H15"/><path d="M15 7h7"/>',
    "close-windows": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 8h18"/><path d="m9.5 11.5 5 5M14.5 11.5l-5 5"/>',
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
    "save-cube": '<path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="m3 7 9 5 9-5M12 12v10"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "cancel": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
    "rgb": '<circle cx="9" cy="9" r="5"/><circle cx="15" cy="9" r="5"/><circle cx="12" cy="15" r="5"/>',
    "contrast-up": '<circle cx="10" cy="12" r="7"/><path d="M10 5a7 7 0 0 1 0 14z" fill="currentColor"/><path d="M20 4v6M17 7h6"/>',
    "contrast-down": '<circle cx="10" cy="12" r="7"/><path d="M10 5a7 7 0 0 1 0 14z" fill="currentColor"/><path d="M17 7h6"/>',
    "bias-up": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "bias-down": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
    "background": '<path d="M3 3v18h18"/><path d="M6 8c3 0 4 4 7 4s4-3 7-3" opacity=".45"/><path d="M6 17c2 0 3-6 5-6s2 6 4 6 2-3 5-3"/>',
    "segment": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 14c4-1 5-5 9-5s5 3 9 2M12 9v12"/>',
    "segment-supervised": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 14c4-1 5-5 9-5s5 3 9 2M12 9v12"/><circle cx="7" cy="7.5" r="1.8" fill="currentColor"/><circle cx="17" cy="17" r="1.8" fill="currentColor"/>',
    "save-segmentation": '<path d="M14 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V10"/><path d="M3 15c4-1 5-5 9-5"/><path d="M18 2v6M15 5.5 18 8.5l3-3"/>',
    "restore": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
}

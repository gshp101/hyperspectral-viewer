"""Command-line entry point: ``hsview cube.hdr --overlay seg.npy``."""
from __future__ import annotations

import argparse

import numpy as np

from .io import Cube, load_cube


def launch(cube=None, overlays=(), annotations=None, block=True):
    import napari

    from . import annotations as ann_io
    from .widget import HyperspectralWidget

    if isinstance(cube, np.ndarray):
        cube = Cube(cube)
    elif cube is not None and not isinstance(cube, Cube):
        cube = load_cube(cube)
    viewer = napari.Viewer(title="hsview")
    widget = HyperspectralWidget(viewer, cube)
    viewer.window.add_dock_widget(widget, name="Hyperspectral", area="right")
    for ov in overlays:
        widget.add_overlay(ov)
    if annotations:
        widget.load_annotation_dict(ann_io.load_annotations(annotations))
    if block:
        napari.run()
    return viewer, widget


def main(argv=None):
    p = argparse.ArgumentParser(description="View and annotate hyperspectral cubes")
    p.add_argument("cube", nargs="?", help="cube file (ENVI .hdr, .npy, .h5, .tif, .pkl)")
    p.add_argument("--overlay", action="append", default=[], help="processed image to overlay (repeatable)")
    p.add_argument("--annotations", help="annotation JSON to load")
    p.add_argument("--band-axis", type=int, help="axis holding the bands for .npy/.h5/.tif/.pkl (default: guess)")
    args = p.parse_args(argv)
    cube = load_cube(args.cube, band_axis=args.band_axis) if args.cube and args.band_axis is not None else args.cube
    launch(cube, args.overlay, args.annotations)


if __name__ == "__main__":
    main()

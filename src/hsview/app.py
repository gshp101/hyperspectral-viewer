"""Command-line entry point: ``hsview cube.pkl --overlay seg.npy``."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .io import Cube, LoadError, load_cube, resolve_image_id


def launch(cube=None, overlays=(), annotations=None, block=True):
    from PyQt6.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("hsview")
    app.setOrganizationName("hsview")
    win = MainWindow()
    win.show()
    if isinstance(cube, np.ndarray):
        cube = Cube(cube)
    if isinstance(cube, Cube):
        win.set_cube(cube)
    elif cube is not None:
        win.open_cube(cube)
    for ov in overlays:
        win.overlays.add_file(ov)
    if annotations:
        win.load_annotations(annotations)
    if block:
        sys.exit(app.exec())
    return win


def main(argv=None):
    p = argparse.ArgumentParser(description="View and annotate hyperspectral cubes")
    p.add_argument("file", nargs="?", help="cube or image file, or a camera database id (a number)")
    p.add_argument("--image-dir", help="directory holding image_database.db, for opening by id "
                                       "(default: $HSVIEW_IMAGE_DIR or /Volumes/HyperspectralData/Images)")
    p.add_argument("--overlay", action="append", default=[], help="processed image to overlay (repeatable)")
    p.add_argument("--annotations", help="annotation file (.pkl or .json) to load")
    p.add_argument("--band-axis", type=int, help="axis holding the bands for .npy/.h5/.tif/.pkl (default: guess)")
    args = p.parse_args(argv)

    target = args.file
    if target is not None and target.isdigit():
        try:
            target = str(resolve_image_id(int(target), args.image_dir))
        except LoadError as exc:
            p.error(str(exc))
    cube = target
    if target is not None and args.band_axis is not None:
        cube = load_cube(target, band_axis=args.band_axis)
    launch(cube, args.overlay, args.annotations)


if __name__ == "__main__":
    main()

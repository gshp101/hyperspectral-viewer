"""hsview: view and annotate hyperspectral image cubes."""
from .io import Cube, LoadError, load_cube, load_overlay

__version__ = "0.2.0"
__all__ = ["Cube", "LoadError", "load_cube", "load_overlay", "launch"]


def launch(cube=None, overlays=(), annotations=None, block=True):
    """Open the viewer. ``cube`` may be a path, a Cube, or a (rows, cols, bands) array."""
    from .app import launch as _launch

    return _launch(cube, overlays, annotations, block)

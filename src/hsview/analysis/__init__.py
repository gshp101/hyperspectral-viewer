"""Analyses that turn one cube into another.

The numerical code lives here and knows nothing about Qt, so it can be used
from scripts and tested directly. The GUI side (parameter dialogs and the
Analysis menu) is in :mod:`hsview.gui.analysis`.
"""
from .background import SVDBackground, fit_svd_background, remove_svd_background

from .segmentation import (
    Segmentation,
    read_segmentation,
    save_segmentation,
    segment_supervised,
    segment_unsupervised,
)

__all__ = ["SVDBackground", "fit_svd_background", "remove_svd_background", "Segmentation", "read_segmentation",
           "save_segmentation", "segment_supervised", "segment_unsupervised"]

"""Toolbar icons: small line-art SVGs tinted to the current palette.

The drawings (24x24 strokes in the style of the Lucide icon set) are in
:data:`hsview.ui_spec.ICON_SVG`, shared with the browser version; this module
renders them with Qt so every icon follows the light/dark theme.
"""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication

from ..ui_spec import ICON_SVG

_SVG = ICON_SVG

ICON_NAMES = tuple(sorted(_SVG))


def _text_color() -> QColor:
    app = QApplication.instance()
    return app.palette().windowText().color() if app else QColor("#333")


@lru_cache(maxsize=None)
def _render(name: str, color: str, size: int) -> QPixmap:
    body = _SVG[name].replace("currentColor", color)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return pm


def icon(name: str) -> QIcon:
    """A theme-tinted icon; unknown names give an empty icon."""
    if name not in _SVG:
        return QIcon()
    color = _text_color().name()
    ic = QIcon()
    for size in (16, 24, 32, 48, 64):  # Qt derives the disabled look from these
        ic.addPixmap(_render(name, color, size))
    return ic


def swatch_icon(rgb: tuple[float, float, float]) -> QIcon:
    """A filled circle in a class colour."""
    ic = QIcon()
    for size in (16, 24, 32, 48):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(0, 0, 0, 160))
        p.setBrush(QColor.fromRgbF(*rgb))
        m = size * 0.15
        p.drawEllipse(QRectF(m, m, size - 2 * m, size - 2 * m))
        p.end()
        ic.addPixmap(pm)
    return ic

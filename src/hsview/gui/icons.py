"""Toolbar icons: small line-art SVGs tinted to the current palette.

The drawings are 24x24 strokes in the style of the Lucide icon set. Keeping
them inline avoids shipping data files and lets every icon follow the
light/dark theme.
"""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication

_SVG = {
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
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
}

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

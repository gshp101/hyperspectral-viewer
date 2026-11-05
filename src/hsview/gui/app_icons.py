"""Candidate application (window/taskbar) icons, chosen under View → Application icon.

The drawings are in :mod:`hsview.app_icon_svg`; this module renders them with Qt.
"""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from ..app_icon_svg import APP_ICONS, DEFAULT_APP_ICON

__all__ = ["APP_ICONS", "DEFAULT_APP_ICON", "render_app_icon", "app_icon"]


def render_app_icon(name: str, size: int) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(APP_ICONS[name][1].encode()))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return pm


def app_icon(name: str = DEFAULT_APP_ICON) -> QIcon:
    if name not in APP_ICONS:
        name = DEFAULT_APP_ICON
    ic = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(render_app_icon(name, size))
    return ic

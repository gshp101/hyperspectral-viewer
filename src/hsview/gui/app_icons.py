"""Candidate application (window/taskbar) icons, chosen under View → Application icon."""
from __future__ import annotations

import math

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

# Shared rainbow gradient, violet to red, as SVG stops.
_RAINBOW = ('<stop offset="0" stop-color="#6a3df0"/><stop offset=".2" stop-color="#2f6bff"/>'
            '<stop offset=".4" stop-color="#18b6c9"/><stop offset=".55" stop-color="#3cc04a"/>'
            '<stop offset=".72" stop-color="#f2d22e"/><stop offset=".86" stop-color="#ff8a1f"/>'
            '<stop offset="1" stop-color="#e8333a"/>')


def _svg(body: str, defs: str = "") -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            f'<defs>{defs}</defs>{body}</svg>')


def _bands(x0, y0, w, h, n=7, vertical=False):
    cols = ["#6a3df0", "#2f6bff", "#18b6c9", "#3cc04a", "#f2d22e", "#ff8a1f", "#e8333a"]
    out = []
    for i in range(n):
        if vertical:
            out.append(f'<rect x="{x0 + i * w / n:.1f}" y="{y0}" width="{w / n + .6:.1f}" height="{h}" fill="{cols[i]}"/>')
        else:
            out.append(f'<rect x="{x0}" y="{y0 + i * h / n:.1f}" width="{w}" height="{h / n + .6:.1f}" fill="{cols[i]}"/>')
    return "".join(out)


# A little landscape drawn in a 100x100 square, mapped onto a cube's top face.
_SCENE = (
    '<rect width="100" height="100" fill="#8fd3ff"/>'
    '<rect y="0" width="100" height="45" fill="#bfe8ff"/>'
    '<circle cx="78" cy="18" r="10" fill="#ffd43b"/>'
    '<path d="M0 58 L22 30 L40 52 L58 24 L84 58 Z" fill="#7d8ca8"/>'
    '<path d="M50 34 L58 24 L66 34 Z" fill="#ffffff"/>'
    '<path d="M0 56 Q30 46 55 56 T100 54 V100 H0 Z" fill="#5cbf4a"/>'
    '<path d="M0 74 Q40 64 100 72 V100 H0 Z" fill="#3e9e3a"/>'
    '<ellipse cx="68" cy="82" rx="20" ry="7" fill="#3aa0e8"/>'
    '<rect x="22" y="62" width="4" height="12" fill="#7a4b2a"/>'
    '<circle cx="24" cy="58" r="9" fill="#2f8f3a"/>'
)
_STRIPES = ["#6a3df0", "#2f6bff", "#18b6c9", "#3cc04a", "#f2d22e", "#ff8a1f", "#e8333a"]


def _scene_cube(cx: float, top: float, w: float, depth: float, outline: str, probe=(62, 70)) -> str:
    """Isometric cube: scene on top, spectral band stripes down the sides, one probed pixel.

    ``cx``/``top`` place the back corner, ``w`` is the half-width of the top face and
    ``depth`` the height of the sides. Returns the SVG and leaves the probed pixel's
    centre in ``_scene_cube.probe_xy``.
    """
    h = w * 46 / 84  # isometric slope of the reference drawing
    L, B, R, F = (cx - w, top + h), (cx, top), (cx + w, top + h), (cx, top + 2 * h)
    a, b, c, d = w / 100, -h / 100, w / 100, h / 100  # unit square -> top face
    out = [f'<g transform="matrix({a:.4f} {b:.4f} {c:.4f} {d:.4f} {L[0]:.2f} {L[1]:.2f})">{_SCENE}'
           f'<rect x="{probe[0] - 5}" y="{probe[1] - 5}" width="10" height="10" fill="#ffffff" stroke="{outline}" stroke-width="3"/></g>']
    n = len(_STRIPES)
    for i, col in enumerate(_STRIPES):  # bands run down the sides: violet at the top
        y0, y1 = depth * i / n, depth * (i + 1) / n + 0.6
        out.append(f'<polygon points="{L[0]},{L[1] + y0} {F[0]},{F[1] + y0} {F[0]},{F[1] + y1} {L[0]},{L[1] + y1}" fill="{col}"/>')
        out.append(f'<polygon points="{F[0]},{F[1] + y0} {R[0]},{R[1] + y0} {R[0]},{R[1] + y1} {F[0]},{F[1] + y1}" fill="{col}" fill-opacity=".82"/>')
        out.append(f'<polygon points="{F[0]},{F[1] + y0} {R[0]},{R[1] + y0} {R[0]},{R[1] + y1} {F[0]},{F[1] + y1}" fill="#000000" fill-opacity=".12"/>')
    out.append(f'<path d="M{L[0]} {L[1]} L{B[0]} {B[1]} L{R[0]} {R[1]} L{F[0]} {F[1]} Z M{L[0]} {L[1]} V{L[1] + depth} '
               f'L{F[0]} {F[1] + depth} L{R[0]} {R[1] + depth} V{R[1]} M{F[0]} {F[1]} V{F[1] + depth}" fill="none" '
               f'stroke="{outline}" stroke-width="{max(3, w / 18):.1f}" stroke-linejoin="round"/>')
    u, v = probe
    _scene_cube.probe_xy = (L[0] + a * u + c * v, L[1] + b * u + d * v)
    return "".join(out)


def _spectrum_card(x, y, w, h, outline, curve="#10151f", bg="#ffffff") -> str:
    pts = [(0.0, .78), (.12, .74), (.24, .40), (.34, .30), (.44, .52), (.56, .62), (.68, .20), (.80, .16), (.90, .46), (1.0, .40)]
    px = lambda t: x + 8 + t * (w - 16)
    py = lambda t: y + 8 + t * (h - 16)
    d = "M" + " L".join(f"{px(t):.1f} {py(v):.1f}" for t, v in pts)
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{bg}" stroke="{outline}" stroke-width="5"/>'
            f'<path d="{d}" fill="none" stroke="url(#spec)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M{x + 8} {y + h - 8} H{x + w - 8}" stroke="{curve}" stroke-opacity=".35" stroke-width="3"/>')


def _probe_cube_icon(variant: str) -> str:
    defs = f'<linearGradient id="spec" x1="0" y1="0" x2="1" y2="0">{_RAINBOW}</linearGradient>'
    if variant == "side":  # cube on the left, spectrum card on the right, leader line from the probed pixel
        cube = _scene_cube(88, 56, 68, 92, "#1d2433")
        px, py = _scene_cube.probe_xy
        body = ('<rect width="256" height="256" rx="56" fill="#1d2433"/>' + cube
                + f'<path d="M{px:.1f} {py:.1f} L180 72" stroke="#ffffff" stroke-width="4" stroke-dasharray="7 6"/>'
                + _spectrum_card(162, 30, 80, 66, "#ffffff"))
    elif variant == "above":  # spectrum card floating above the cube
        cube = _scene_cube(128, 104, 86, 82, "#1d2433")
        px, py = _scene_cube.probe_xy
        body = ('<rect width="256" height="256" rx="56" fill="#1d2433"/>' + cube
                + f'<path d="M{px:.1f} {py:.1f} L150 90" stroke="#ffffff" stroke-width="4" stroke-dasharray="7 6"/>'
                + _spectrum_card(100, 18, 120, 74, "#ffffff"))
    else:  # "light": light tile, larger cube, compact card bottom-right
        cube = _scene_cube(112, 34, 84, 100, "#1d2433")
        px, py = _scene_cube.probe_xy
        body = ('<rect width="256" height="256" rx="56" fill="#eef2f8"/>' + cube
                + f'<path d="M{px:.1f} {py:.1f} L190 168" stroke="#1d2433" stroke-width="4" stroke-dasharray="7 6"/>'
                + _spectrum_card(160, 160, 82, 64, "#1d2433"))
    return _svg(body, defs)


APP_ICONS: dict[str, tuple[str, str]] = {
    "scene-cube": ("Scene cube + probe", _probe_cube_icon("side")),
    "scene-cube-above": ("Scene cube + probe (spectrum above)", _probe_cube_icon("above")),
    "scene-cube-light": ("Scene cube + probe (light)", _probe_cube_icon("light")),
    "hypercube": ("Hypercube", _svg(
        # Isometric cube: photo-like top, rainbow band stripes down the two side faces.
        '<rect width="256" height="256" rx="56" fill="#1d2433"/>'
        '<polygon points="128,40 212,86 128,132 44,86" fill="#cfd8e6"/>'
        '<polygon points="44,86 128,132 128,222 44,176" fill="url(#l)"/>'
        '<polygon points="128,132 212,86 212,176 128,222" fill="url(#r)"/>'
        '<polygon points="128,40 212,86 128,132 44,86" fill="none" stroke="#1d2433" stroke-width="5" stroke-linejoin="round"/>'
        '<path d="M44 86v90l84 46 84-46V86M128 132v90" fill="none" stroke="#1d2433" stroke-width="5" stroke-linejoin="round"/>',
        f'<linearGradient id="l" x1="0" y1="0" x2="0" y2="1">{_RAINBOW}</linearGradient>'
        f'<linearGradient id="r" x1="0" y1="0" x2="0" y2="1">{_RAINBOW}</linearGradient>')),
    "prism": ("Prism", _svg(
        '<rect width="256" height="256" rx="56" fill="#141820"/>'
        '<path d="M20 150 L104 122" stroke="#ffffff" stroke-width="10" stroke-linecap="round"/>'
        '<polygon points="104,122 236,76 236,190" fill="url(#fan)" opacity=".95"/>'
        '<polygon points="128,48 186,170 70,170" fill="#e9eef7" fill-opacity=".22" stroke="#e9eef7" stroke-width="7" stroke-linejoin="round"/>',
        f'<linearGradient id="fan" x1="0" y1="0" x2="0" y2="1">{_RAINBOW}</linearGradient>')),
    "band-stack": ("Band stack", _svg(
        # Image planes stacked through the spectrum.
        '<rect width="256" height="256" rx="56" fill="#f4f6fa"/>'
        + "".join(
            f'<rect x="{34 + 14 * i}" y="{158 - 15 * i}" width="110" height="72" rx="8" fill="{c}" '
            f'stroke="#ffffff" stroke-width="4"/>'
            for i, c in enumerate(["#e8333a", "#ff8a1f", "#f2d22e", "#3cc04a", "#18b6c9", "#2f6bff", "#6a3df0"])))),
    "spectrum-curve": ("Spectrum curve", _svg(
        '<rect width="256" height="256" rx="56" fill="#10151f"/>'
        '<path d="M40 210H222M40 210V44" stroke="#5b667a" stroke-width="6" stroke-linecap="round"/>'
        '<path d="M48 186C70 182 80 120 98 112S122 160 140 150 160 66 178 62 204 140 220 132" fill="none" '
        'stroke="url(#c)" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<linearGradient id="c" x1="0" y1="0" x2="1" y2="0">{_RAINBOW}</linearGradient>')),
    "pixel-probe": ("Pixel probe", _svg(
        # A pixel grid with one pixel picked out and its spectrum drawn beside it.
        '<rect width="256" height="256" rx="56" fill="#1b2230"/>'
        + "".join(f'<rect x="{34 + 36 * c}" y="{70 + 36 * r}" width="32" height="32" rx="4" fill="{"#43506a" if (r + c) % 2 else "#37425a"}"/>'
                  for r in range(4) for c in range(3))
        + '<rect x="70" y="106" width="32" height="32" rx="4" fill="#ffffff"/>'
        '<path d="M102 122 L150 122" stroke="#ffffff" stroke-width="5" stroke-dasharray="6 6"/>'
        f'{_bands(150, 64, 72, 120, vertical=True)}'
        '<path d="M150 170C162 166 168 98 180 96S196 150 206 140 214 110 222 108" fill="none" stroke="#10151f" stroke-width="7" stroke-linecap="round"/>'
        '<rect x="150" y="64" width="72" height="120" rx="6" fill="none" stroke="#ffffff" stroke-width="5"/>')),
    "aperture": ("Spectral aperture", _svg(
        '<rect width="256" height="256" rx="56" fill="#141820"/>'
        + "".join(
            f'<path d="M128 128 L{128 + 92 * math.cos(a):.1f} {128 + 92 * math.sin(a):.1f} '
            f'A92 92 0 0 1 {128 + 92 * math.cos(a + 0.9):.1f} {128 + 92 * math.sin(a + 0.9):.1f} Z" fill="{c}"/>'
            for a, c in zip([i * 0.8976 for i in range(7)],
                            ["#6a3df0", "#2f6bff", "#18b6c9", "#3cc04a", "#f2d22e", "#ff8a1f", "#e8333a"]))
        + '<circle cx="128" cy="128" r="34" fill="#141820"/><circle cx="128" cy="128" r="92" fill="none" stroke="#e9eef7" stroke-width="8"/>')),
    "rainbow-eye": ("Spectral eye", _svg(
        '<rect width="256" height="256" rx="56" fill="#f4f6fa"/>'
        '<path d="M24 128C64 66 192 66 232 128 192 190 64 190 24 128Z" fill="#ffffff" stroke="#1d2433" stroke-width="10" stroke-linejoin="round"/>'
        '<circle cx="128" cy="128" r="46" fill="url(#iris)"/><circle cx="128" cy="128" r="18" fill="#1d2433"/>'
        '<circle cx="140" cy="114" r="7" fill="#ffffff"/>',
        f'<linearGradient id="iris" x1="0" y1="0" x2="1" y2="1">{_RAINBOW}</linearGradient>')),
    "hs-monogram": ("HS monogram", _svg(
        '<rect width="256" height="256" rx="56" fill="url(#bg)"/>'
        '<text x="128" y="164" font-family="Helvetica, Arial, sans-serif" font-size="112" font-weight="bold" '
        'fill="#ffffff" text-anchor="middle">HS</text>',
        f'<linearGradient id="bg" x1="0" y1="1" x2="1" y2="0">{_RAINBOW}</linearGradient>')),
}

DEFAULT_APP_ICON = "scene-cube"


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

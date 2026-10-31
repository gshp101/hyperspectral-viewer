"""The image canvas: a QGraphicsView showing the display image, overlays and annotations.

Rendering is plain Qt raster painting (no OpenGL). Scene units are image
pixels: pixel (row, col) covers [col, col+1) x [row, row+1).
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..annotations import CLASS_COLORS, AnnotationStore

MODES = ("navigate", "select", "point", "rect", "polygon")
ID_ROLE = 0
POINT_RADIUS = 5  # screen pixels
PICK_TOLERANCE = 6  # screen pixels for right-click delete
MIN_ZOOM, MAX_ZOOM = 0.02, 200.0


def array_to_pixmap(img: np.ndarray) -> QPixmap:
    """uint8 (H, W), (H, W, 3) or (H, W, 4) array to a QPixmap (copied)."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        fmt, bpl = QImage.Format.Format_Grayscale8, w
    elif img.shape[2] == 3:
        fmt, bpl = QImage.Format.Format_RGB888, 3 * w
    else:
        fmt, bpl = QImage.Format.Format_RGBA8888, 4 * w
    qimg = QImage(img.data, w, h, bpl, fmt)
    return QPixmap.fromImage(qimg.copy())


def _class_color(cls: int, alpha: int = 255) -> QColor:
    r, g, b = CLASS_COLORS[cls % len(CLASS_COLORS)]
    return QColor.fromRgbF(r, g, b, alpha / 255)


def _segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    d = b - a
    denom = float(d @ d)
    t = 0.0 if denom == 0 else min(1.0, max(0.0, float((p - a) @ d) / denom))
    return float(np.linalg.norm(a + t * d - p))


def _outline_distance(p, vertices) -> float:
    v = np.asarray(vertices, dtype=float)
    return min(_segment_distance(p, v[i], v[(i + 1) % len(v)]) for i in range(len(v)))


class ImageView(QGraphicsView):
    cursorMoved = pyqtSignal(float, float)  # image x, y; NaN when the cursor leaves
    zoomChanged = pyqtSignal(float)
    modeChanged = pyqtSignal(str)

    def __init__(self, store: AnnotationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.current_class = 0
        self._mode = "navigate"
        self._fit = True  # keep the image fitted to the window until the user zooms
        self._overlays: dict[str, QGraphicsPixmapItem] = {}
        self._items: dict[int, QGraphicsItem] = {}
        self._drag_start: QPointF | None = None
        self._pan_last = None
        self._poly: list[QPointF] = []

        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(60, 60, 60))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)  # annotations only; pixels stay sharp
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMouseTracking(True)
        # Pan by dragging instead; scrollbars would only show the margin around the image.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._image = QGraphicsPixmapItem()
        self._image.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._image.setZValue(0)
        self.scene().addItem(self._image)

        pen = QPen(QColor(255, 255, 255), 0, Qt.PenStyle.DashLine)
        self._preview = QGraphicsPathItem()
        self._preview.setPen(pen)
        self._preview.setZValue(100)
        self.scene().addItem(self._preview)

        store.subscribe(self.rebuild_annotations)
        self.set_mode("navigate")

    # ------------------------------------------------------------- image
    @property
    def has_image(self) -> bool:
        return not self._image.pixmap().isNull()

    @property
    def image_rect(self) -> QRectF:
        return self._image.boundingRect()

    def set_image(self, img: np.ndarray | None, refit: bool = False):
        if img is None:
            self._image.setPixmap(QPixmap())
            return
        size_changed = self._image.pixmap().size().width() != img.shape[1] or \
            self._image.pixmap().size().height() != img.shape[0]
        self._image.setPixmap(array_to_pixmap(img))
        if refit or size_changed:
            self.scene().setSceneRect(self.image_rect.adjusted(-0.5 * img.shape[1], -0.5 * img.shape[0],
                                                               0.5 * img.shape[1], 0.5 * img.shape[0]))
            self.zoom_fit()

    # ---------------------------------------------------------- overlays
    def set_overlay(self, key: str, rgba: np.ndarray, opacity: float = 0.5, visible: bool = True):
        item = self._overlays.get(key)
        if item is None:
            item = QGraphicsPixmapItem()
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            item.setZValue(1 + len(self._overlays) * 0.01)
            self.scene().addItem(item)
            self._overlays[key] = item
        item.setPixmap(array_to_pixmap(rgba))
        item.setOpacity(opacity)
        item.setVisible(visible)

    def set_overlay_opacity(self, key: str, opacity: float):
        if key in self._overlays:
            self._overlays[key].setOpacity(opacity)

    def set_overlay_visible(self, key: str, visible: bool):
        if key in self._overlays:
            self._overlays[key].setVisible(visible)

    def remove_overlay(self, key: str):
        item = self._overlays.pop(key, None)
        if item is not None:
            self.scene().removeItem(item)

    def clear_overlays(self):
        for key in list(self._overlays):
            self.remove_overlay(key)

    # --------------------------------------------------------------- zoom
    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def _apply_zoom(self, factor: float):
        new = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        factor = new / self.zoom
        if factor != 1:
            self.scale(factor, factor)
            self.zoomChanged.emit(self.zoom)

    def zoom_in(self):
        self._fit = False
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._fit = False
        self._apply_zoom(0.8)

    def zoom_fit(self):
        self._fit = True
        if self.has_image:
            self.fitInView(self.image_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoomChanged.emit(self.zoom)

    def zoom_actual(self):
        self._fit = False
        self.resetTransform()
        self.zoomChanged.emit(self.zoom)

    def wheelEvent(self, event):
        if not self.has_image:
            return
        self._fit = False
        self._apply_zoom(math.pow(1.0015, event.angleDelta().y()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.zoom_fit()

    # --------------------------------------------------------------- modes
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        self.cancel_drawing()
        self._mode = mode
        self.setDragMode({
            "navigate": QGraphicsView.DragMode.ScrollHandDrag,
            "select": QGraphicsView.DragMode.RubberBandDrag,
        }.get(mode, QGraphicsView.DragMode.NoDrag))
        self._apply_cursor()
        editable = mode == "select"
        for item in self._items.values():
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
            if not editable:
                item.setSelected(False)
        self.modeChanged.emit(mode)

    def _apply_cursor(self):
        if self._mode in ("point", "rect", "polygon"):
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        elif self._mode == "select":
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    def cancel_drawing(self):
        self._drag_start = None
        self._poly = []
        self._preview.setPath(QPainterPath())

    # --------------------------------------------------------- annotations
    def selected_ids(self) -> list[int]:
        return [it.data(ID_ROLE) for it in self.scene().selectedItems() if it.data(ID_ROLE) is not None]

    def select_ids(self, ids):
        ids = set(ids)
        for ann_id, item in self._items.items():
            item.setSelected(ann_id in ids)

    def delete_selected(self) -> int:
        return self.store.remove(self.selected_ids())

    def rebuild_annotations(self):
        selected = set(self.selected_ids())
        for item in self._items.values():
            self.scene().removeItem(item)
        self._items.clear()
        editable = self._mode == "select"
        for ann in self.store:
            color = _class_color(ann.cls)
            if ann.kind == "point":
                x, y = ann.coords[0]
                item = QGraphicsEllipseItem(-POINT_RADIUS, -POINT_RADIUS, 2 * POINT_RADIUS, 2 * POINT_RADIUS)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                item.setPos(x + 0.5, y + 0.5)
                item.setBrush(QBrush(color))
                item.setPen(QPen(QColor(0, 0, 0), 1.2))
                item.setZValue(12)
            else:
                pen = QPen(color, 2)
                pen.setCosmetic(True)
                fill = _class_color(ann.cls, 40)
                if ann.kind == "rect":
                    (x0, y0), (x1, y1) = ann.coords
                    item = QGraphicsRectItem(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
                else:
                    item = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in ann.coords]))
                item.setPen(pen)
                item.setBrush(QBrush(fill))
                item.setZValue(10)
            item.setData(ID_ROLE, ann.id)
            item.setData(1, QPointF(item.pos()))
            item.setToolTip(ann.label)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
            item.setSelected(ann.id in selected)
            self.scene().addItem(item)
            self._items[ann.id] = item

    def _commit_moves(self):
        """Write positions of items dragged in select mode back to the store."""
        changes = {}
        for ann_id, item in self._items.items():
            delta = item.pos() - item.data(1)
            if delta.isNull():
                continue
            ann = self.store.get(ann_id)
            d = np.array([delta.x(), delta.y()])
            coords = ann.coords + d
            if ann.kind == "point":
                coords = np.floor(coords + 0.5)  # snap to the nearest pixel
            changes[ann_id] = {"coords": coords}
        if changes:
            self.store.update(changes)

    def annotation_at(self, pos: QPointF) -> int | None:
        """Id of the annotation nearest to a scene position, within the pick tolerance."""
        tol = PICK_TOLERANCE / max(self.zoom, 1e-9)
        p = (pos.x(), pos.y())
        best, best_d = None, tol
        for ann in self.store:
            if ann.kind == "point":
                d = float(np.hypot(*(ann.coords[0] + 0.5 - p))) - POINT_RADIUS / max(self.zoom, 1e-9)
            elif ann.kind == "rect":
                (x0, y0), (x1, y1) = ann.coords
                d = _outline_distance(p, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            else:
                d = _outline_distance(p, ann.coords)
            if d <= best_d:
                best, best_d = ann.id, d
        return best

    # ------------------------------------------------------------ mouse
    def _in_image(self, p: QPointF) -> bool:
        return self.image_rect.contains(p)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if not self.has_image or self._mode in ("navigate", "select"):
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self._mode == "polygon" and self._poly:
                self._poly.pop()  # right click while drawing removes the last vertex
                self._update_poly_preview(pos)
                return
            ann_id = self.annotation_at(pos)
            if ann_id is not None:
                self.store.remove([ann_id])
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._mode == "point":
            if self._in_image(pos):
                self.store.add("point", self.current_class, [[math.floor(pos.x()), math.floor(pos.y())]])
        elif self._mode == "rect":
            self._drag_start = pos
        elif self._mode == "polygon":
            self._poly.append(pos)
            self._update_poly_preview(pos)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
        if self.has_image:
            if self._in_image(pos):
                self.cursorMoved.emit(pos.x(), pos.y())
            else:
                self.cursorMoved.emit(math.nan, math.nan)
        if self._mode == "rect" and self._drag_start is not None:
            path = QPainterPath()
            path.addRect(QRectF(self._drag_start, pos).normalized())
            self._preview.setPen(self._preview_pen())
            self._preview.setPath(path)
        elif self._mode == "polygon" and self._poly:
            self._update_poly_preview(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self._apply_cursor()
            return
        pos = self.mapToScene(event.position().toPoint())
        if self._mode == "rect" and self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            start, self._drag_start = self._drag_start, None
            self._preview.setPath(QPainterPath())
            screen = self.mapFromScene(start) - self.mapFromScene(pos)
            if abs(screen.x()) > 4 and abs(screen.y()) > 4:
                self.store.add("rect", self.current_class, [[start.x(), start.y()], [pos.x(), pos.y()]])
            return
        super().mouseReleaseEvent(event)
        if self._mode == "select":
            self._commit_moves()

    def mouseDoubleClickEvent(self, event):
        if self._mode == "polygon" and event.button() == Qt.MouseButton.LeftButton:
            self.finish_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        self.cursorMoved.emit(math.nan, math.nan)
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.cancel_drawing()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._mode == "polygon":
            self.finish_polygon()
        elif key == Qt.Key.Key_Backspace and self._mode == "polygon" and self._poly:
            self._poly.pop()
            self._update_poly_preview(self._poly[-1] if self._poly else QPointF())
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._mode == "select":
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    # ---------------------------------------------------------- polygon
    def _preview_pen(self) -> QPen:
        pen = QPen(_class_color(self.current_class), 0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        return pen

    def _update_poly_preview(self, cursor: QPointF):
        path = QPainterPath()
        if self._poly:
            path.moveTo(self._poly[0])
            for p in self._poly[1:]:
                path.lineTo(p)
            path.lineTo(cursor)
        self._preview.setPen(self._preview_pen())
        self._preview.setPath(path)

    def finish_polygon(self):
        pts = self._poly
        # A double click also delivers a press: drop vertices that landed on the previous one.
        tol = 3 / max(self.zoom, 1e-9)
        cleaned = []
        for p in pts:
            if not cleaned or (p - cleaned[-1]).manhattanLength() > tol:
                cleaned.append(p)
        self.cancel_drawing()
        if len(cleaned) >= 3:
            self.store.add("polygon", self.current_class, [[p.x(), p.y()] for p in cleaned])

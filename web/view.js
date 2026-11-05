// The image canvas: display image, overlays and annotations, with zoom, pan and the drawing modes.
// Image units are pixels: pixel (row, col) covers [col, col+1) x [row, row+1), as in the Qt ImageView.

export const MODES = ["navigate", "select", "point", "rect", "polygon"];
const POINT_RADIUS = 5;     // screen pixels
const PICK_TOLERANCE = 6;   // screen pixels for right-click delete
const MIN_ZOOM = 0.02, MAX_ZOOM = 200;

function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const den = dx * dx + dy * dy;
  const t = den === 0 ? 0 : Math.min(1, Math.max(0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den));
  return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
}
function outlineDist(p, v) {
  let d = Infinity;
  for (let i = 0; i < v.length; i++) d = Math.min(d, segDist(p, v[i], v[(i + 1) % v.length]));
  return d;
}
function rgbaCanvas(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const data = new ImageData(new Uint8ClampedArray(img.rgba.buffer, img.rgba.byteOffset, img.rgba.byteLength), img.width, img.height);
  c.getContext("2d").putImageData(data, 0, 0);
  return c;
}
function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export class ImageView {
  constructor(canvas, classColors) {
    this.canvas = canvas;
    this.classColors = classColors;
    this.image = null;          // offscreen canvas of the display image
    this.overlays = new Map();  // key -> {canvas, opacity, visible}
    this.overlayOrder = [];
    this.annotations = [];
    this.selected = new Set();
    this.mode = "navigate";
    this.currentClass = 0;
    this.zoom = 1; this.tx = 0; this.ty = 0;
    this.fit = true;
    this.drag = null;            // current mouse interaction
    this.poly = [];              // polygon vertices being drawn
    this.cursorScene = null;
    this.on = { cursor() {}, add() {}, remove() {}, move() {}, selection() {}, mode() {}, zoom() {}, finishPolygon: null };
    canvas.addEventListener("mousedown", (e) => this._down(e));
    window.addEventListener("mousemove", (e) => this._move(e));
    window.addEventListener("mouseup", (e) => this._up(e));
    canvas.addEventListener("dblclick", (e) => this._dbl(e));
    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("mouseleave", () => { if (!this.drag) this.on.cursor(NaN, NaN); });
    new ResizeObserver(() => { if (this.fit) this.zoomFit(false); this.draw(); }).observe(canvas);
  }

  get hasImage() { return !!this.image; }
  get width() { return this.image ? this.image.width : 0; }
  get height() { return this.image ? this.image.height : 0; }

  setImage(img) {
    if (!img) { this.image = null; this.draw(); return; }
    const changed = !this.image || this.image.width !== img.width || this.image.height !== img.height;
    this.image = rgbaCanvas(img);
    if (changed) this.zoomFit();
    this.draw();
  }

  setOverlay(key, img, opacity, visible) {
    const ov = this.overlays.get(key) || {};
    if (img) ov.canvas = rgbaCanvas(img);
    ov.opacity = opacity; ov.visible = visible;
    if (!this.overlays.has(key)) this.overlayOrder.push(key);
    this.overlays.set(key, ov);
    this.draw();
  }
  removeOverlay(key) { this.overlays.delete(key); this.overlayOrder = this.overlayOrder.filter((k) => k !== key); this.draw(); }
  clearOverlays() { this.overlays.clear(); this.overlayOrder = []; this.draw(); }

  setAnnotations(list) {
    this.annotations = list;
    const ids = new Set(list.map((a) => a.id));
    for (const id of [...this.selected]) if (!ids.has(id)) this.selected.delete(id);
    this.draw();
  }

  // ------------------------------------------------------------ zoom
  _size() { return { w: this.canvas.clientWidth, h: this.canvas.clientHeight }; }
  toScene(sx, sy) { return [(sx - this.tx) / this.zoom, (sy - this.ty) / this.zoom]; }
  toScreen(x, y) { return [x * this.zoom + this.tx, y * this.zoom + this.ty]; }
  _zoomAt(factor, sx, sy) {
    const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.zoom * factor));
    const [x, y] = this.toScene(sx, sy);
    this.zoom = z;
    this.tx = sx - x * z; this.ty = sy - y * z;
    this.on.zoom(this.zoom);
    this.draw();
  }
  _center() { const { w, h } = this._size(); return [w / 2, h / 2]; }
  zoomIn() { this.fit = false; this._zoomAt(1.25, ...this._center()); }
  zoomOut() { this.fit = false; this._zoomAt(0.8, ...this._center()); }
  zoomFit(redraw = true) {
    this.fit = true;
    if (!this.image) return;
    const { w, h } = this._size();
    const m = 2;
    this.zoom = Math.min((w - 2 * m) / this.width, (h - 2 * m) / this.height);
    if (!(this.zoom > 0)) this.zoom = 1;
    this.tx = (w - this.width * this.zoom) / 2; this.ty = (h - this.height * this.zoom) / 2;
    this.on.zoom(this.zoom);
    if (redraw) this.draw();
  }
  zoomActual() {
    this.fit = false;
    const [cx, cy] = this._center();
    this._zoomAt(1 / this.zoom, cx, cy);
  }

  // ------------------------------------------------------------ modes
  setMode(mode) {
    if (!MODES.includes(mode)) throw new Error(mode);
    this.cancelDrawing();
    this.mode = mode;
    if (mode !== "select" && this.selected.size) { this.selected.clear(); this.on.selection([]); }
    this._cursor();
    this.on.mode(mode);
    this.draw();
  }
  _cursor(over) {
    const m = this.mode;
    this.canvas.style.cursor = over || (["point", "rect", "polygon"].includes(m) ? "crosshair" : m === "select" ? "default" : "grab");
  }
  cancelDrawing() { this.poly = []; if (this.drag && this.drag.type === "rect") this.drag = null; this.draw(); }
  selectIds(ids) { this.selected = new Set(ids); this.on.selection([...this.selected]); this.draw(); }
  selectedIds() { return [...this.selected]; }

  finishPolygon() {
    const tol = 3 / Math.max(this.zoom, 1e-9);
    const cleaned = [];
    for (const p of this.poly) {
      const q = cleaned[cleaned.length - 1];
      if (!q || Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) > tol) cleaned.push(p);
    }
    this.cancelDrawing();
    if (cleaned.length >= 3) this.on.add("polygon", cleaned);
  }
  popVertex() { if (this.poly.length) { this.poly.pop(); this.draw(); return true; } return false; }

  annotationAt(p) {
    const tol = PICK_TOLERANCE / Math.max(this.zoom, 1e-9);
    let best = null, bestD = tol;
    for (const a of this.annotations) {
      let d;
      if (a.kind === "point") d = Math.hypot(a.coords[0][0] + 0.5 - p[0], a.coords[0][1] + 0.5 - p[1]) - POINT_RADIUS / Math.max(this.zoom, 1e-9);
      else if (a.kind === "rect") {
        const [[x0, y0], [x1, y1]] = a.coords;
        d = outlineDist(p, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
      } else d = outlineDist(p, a.coords);
      if (d <= bestD) { best = a.id; bestD = d; }
    }
    return best;
  }
  _inside(a, p) {  // for clicking inside a region in select mode
    if (a.kind === "point") return false;
    const v = a.kind === "rect" ? (([[x0, y0], [x1, y1]]) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])(a.coords) : a.coords;
    let inside = false;
    for (let i = 0, j = v.length - 1; i < v.length; j = i++) {
      if ((v[i][1] > p[1]) !== (v[j][1] > p[1]) && p[0] < ((v[j][0] - v[i][0]) * (p[1] - v[i][1])) / (v[j][1] - v[i][1]) + v[i][0]) inside = !inside;
    }
    return inside;
  }
  _bbox(a) {
    if (a.kind === "point") {
      const r = POINT_RADIUS / this.zoom, [x, y] = a.coords[0];
      return [x + 0.5 - r, y + 0.5 - r, x + 0.5 + r, y + 0.5 + r];
    }
    const xs = a.coords.map((c) => c[0]), ys = a.coords.map((c) => c[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }

  // ------------------------------------------------------------ mouse
  _pos(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  _inImage(p) { return this.image && p[0] >= 0 && p[1] >= 0 && p[0] < this.width && p[1] < this.height; }

  _down(e) {
    this.canvas.focus();
    const s = this._pos(e), p = this.toScene(...s);
    if (e.button === 1 || (e.button === 0 && this.mode === "navigate")) {
      e.preventDefault();
      this.drag = { type: "pan", last: s, button: e.button };
      this._cursor("grabbing");
      return;
    }
    if (!this.image) return;
    if (e.button === 2) {
      if (this.mode === "polygon" && this.poly.length) { this.poly.pop(); this.cursorScene = p; this.draw(); return; }
      if (["point", "rect", "polygon"].includes(this.mode)) {
        const id = this.annotationAt(p);
        if (id !== null) this.on.remove([id]);
      }
      return;
    }
    if (e.button !== 0) return;
    if (this.mode === "select") {
      let hit = this.annotationAt(p);
      if (hit === null) for (let i = this.annotations.length - 1; i >= 0; i--) if (this._inside(this.annotations[i], p)) { hit = this.annotations[i].id; break; }
      const additive = e.ctrlKey || e.metaKey || e.shiftKey;
      if (hit !== null) {
        if (additive) { this.selected.has(hit) ? this.selected.delete(hit) : this.selected.add(hit); }
        else if (!this.selected.has(hit)) this.selected = new Set([hit]);
        this.on.selection([...this.selected]);
        this.drag = { type: "move", start: p, dx: 0, dy: 0 };
      } else {
        if (!additive && this.selected.size) { this.selected.clear(); this.on.selection([]); }
        this.drag = { type: "band", start: s, end: s, additive };
      }
      this.draw();
    } else if (this.mode === "point") {
      if (this._inImage(p)) this.on.add("point", [[Math.floor(p[0]), Math.floor(p[1])]]);
    } else if (this.mode === "rect") {
      this.drag = { type: "rect", start: p, startScreen: s, end: p };
    } else if (this.mode === "polygon") {
      this.poly.push(p); this.cursorScene = p; this.draw();
    }
  }

  _move(e) {
    const s = this._pos(e), p = this.toScene(...s);
    const d = this.drag;
    if (d && d.type === "pan") {
      this.tx += s[0] - d.last[0]; this.ty += s[1] - d.last[1]; d.last = s;
      this.fit = false;
      this.draw();
    }
    const overCanvas = e.target === this.canvas || d;
    if (this.image && overCanvas) {
      if (this._inImage(p)) this.on.cursor(p[0], p[1]); else this.on.cursor(NaN, NaN);
    }
    if (!d) {
      if (this.mode === "polygon" && this.poly.length && e.target === this.canvas) { this.cursorScene = p; this.draw(); }
      return;
    }
    if (d.type === "rect") { d.end = p; this.draw(); }
    else if (d.type === "move") { d.dx = p[0] - d.start[0]; d.dy = p[1] - d.start[1]; this.draw(); }
    else if (d.type === "band") { d.end = s; this.draw(); }
  }

  _up(e) {
    const d = this.drag;
    if (!d) return;
    const s = this._pos(e), p = this.toScene(...s);
    if (d.type === "pan" && e.button === d.button) { this.drag = null; this._cursor(); return; }
    if (e.button !== 0) return;
    this.drag = null;
    if (d.type === "rect") {
      if (Math.abs(s[0] - d.startScreen[0]) > 4 && Math.abs(s[1] - d.startScreen[1]) > 4) this.on.add("rect", [d.start, p]);
      this.draw();
    } else if (d.type === "move") {
      if ((d.dx || d.dy) && this.selected.size) this.on.move([...this.selected], d.dx, d.dy);
      this.draw();
    } else if (d.type === "band") {
      const [x0, y0] = this.toScene(Math.min(d.start[0], d.end[0]), Math.min(d.start[1], d.end[1]));
      const [x1, y1] = this.toScene(Math.max(d.start[0], d.end[0]), Math.max(d.start[1], d.end[1]));
      if (Math.abs(d.end[0] - d.start[0]) > 2 || Math.abs(d.end[1] - d.start[1]) > 2) {
        for (const a of this.annotations) {
          const [ax0, ay0, ax1, ay1] = this._bbox(a);
          if (ax1 >= x0 && ax0 <= x1 && ay1 >= y0 && ay0 <= y1) this.selected.add(a.id);
        }
        this.on.selection([...this.selected]);
      }
      this.draw();
    }
  }

  _dbl(e) { if (this.mode === "polygon" && e.button === 0) this.finishPolygon(); }

  _wheel(e) {
    e.preventDefault();
    if (!this.image) return;
    this.fit = false;
    const unit = e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? 800 : 1;
    this._zoomAt(Math.pow(1.0015, -e.deltaY * unit * 1.2), ...this._pos(e));
  }

  // ------------------------------------------------------------ paint
  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._paint(); });
  }

  _paint() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const ctx = c.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "rgb(60,60,60)";
    ctx.fillRect(0, 0, c.width, c.height);
    if (!this.image) return;
    ctx.setTransform(dpr * this.zoom, 0, 0, dpr * this.zoom, dpr * this.tx, dpr * this.ty);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.image, 0, 0);
    for (const key of this.overlayOrder) {
      const ov = this.overlays.get(key);
      if (!ov || !ov.visible || !ov.canvas) continue;
      ctx.globalAlpha = ov.opacity;
      ctx.drawImage(ov.canvas, 0, 0);
    }
    ctx.globalAlpha = 1;
    // Annotations in screen pixels (cosmetic pens, like the Qt view).
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    const move = this.drag && this.drag.type === "move" ? this.drag : null;
    const S = (x, y, sel) => this.toScreen(x + (sel && move ? move.dx : 0), y + (sel && move ? move.dy : 0));
    const regions = this.annotations.filter((a) => a.kind !== "point"), points = this.annotations.filter((a) => a.kind === "point");
    for (const a of [...regions, ...points]) {
      const col = this.classColors[a.cls % this.classColors.length];
      const sel = this.selected.has(a.id);
      ctx.setLineDash([]);
      if (a.kind === "point") {
        const [x, y] = S(a.coords[0][0] + 0.5, a.coords[0][1] + 0.5, sel);
        ctx.beginPath(); ctx.arc(x, y, POINT_RADIUS, 0, 2 * Math.PI);
        ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = 1.2; ctx.strokeStyle = "#000"; ctx.stroke();
        if (sel) this._selBox(ctx, x - POINT_RADIUS - 2, y - POINT_RADIUS - 2, 2 * POINT_RADIUS + 4, 2 * POINT_RADIUS + 4);
      } else {
        const v = a.kind === "rect"
          ? (([[x0, y0], [x1, y1]]) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])(a.coords)
          : a.coords;
        ctx.beginPath();
        v.forEach(([x, y], i) => { const q = S(x, y, sel); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
        ctx.closePath();
        ctx.fillStyle = hexToRgba(col, 40 / 255); ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = col; ctx.stroke();
        if (sel) {
          const pts = v.map(([x, y]) => S(x, y, true));
          const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
          this._selBox(ctx, Math.min(...xs) - 1, Math.min(...ys) - 1, Math.max(...xs) - Math.min(...xs) + 2, Math.max(...ys) - Math.min(...ys) + 2);
        }
      }
    }
    const col = this.classColors[this.currentClass % this.classColors.length];
    if (this.drag && this.drag.type === "rect") {
      const [x0, y0] = this.toScreen(...this.drag.start), [x1, y1] = this.toScreen(...this.drag.end);
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1; ctx.strokeStyle = col;
      ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
    if (this.mode === "polygon" && this.poly.length) {
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1; ctx.strokeStyle = col;
      ctx.beginPath();
      this.poly.forEach((p, i) => { const q = this.toScreen(...p); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
      if (this.cursorScene) ctx.lineTo(...this.toScreen(...this.cursorScene));
      ctx.stroke();
    }
    if (this.drag && this.drag.type === "band") {
      const { start, end } = this.drag;
      ctx.setLineDash([]); ctx.fillStyle = "rgba(48,140,198,0.2)"; ctx.strokeStyle = "rgba(48,140,198,0.9)"; ctx.lineWidth = 1;
      const x = Math.min(start[0], end[0]), y = Math.min(start[1], end[1]);
      ctx.fillRect(x, y, Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1]));
      ctx.strokeRect(x + 0.5, y + 0.5, Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1]));
    }
    ctx.setLineDash([]);
  }

  _selBox(ctx, x, y, w, h) {
    ctx.save();
    ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#000"; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.lineDashOffset = 4; ctx.strokeStyle = "#fff"; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.restore();
  }
}

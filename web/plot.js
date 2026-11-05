// Small canvas plotting, styled like the matplotlib figures of the Qt app.

const FONT = "12px system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif";
const SMALL = "9.5px system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif";
const TITLE = "12px system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif";

/** Size a canvas to its CSS box at the device pixel ratio; returns a context in CSS pixels. */
export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function niceStep(span, n) {
  const raw = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / mag;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
}

function ticks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, n);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function fmt(v, step) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
  const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(Math.min(dec, 6));
}

function extent(values, pad = 0.05) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (v !== null && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === Infinity) return [0, 1];
  if (lo === hi) { const d = Math.abs(lo) * 0.05 || 0.5; return [lo - d, hi + d]; }
  const d = (hi - lo) * pad;
  return [lo - d, hi + d];
}

const DASHES = { dashed: [6, 3], dotted: [1.5, 2.5], solid: [] };

/** Draw one set of axes into the rectangle (x, y, w, h). */
function drawAxes(ctx, box, opt) {
  const { series = [], xlabel = "", ylabel = "", title = "", vline = null, xlim = null, showXTicks = true } = opt;
  const xs = series.length ? series[0].x : [0, 1];
  const xl = xlim || extent(series.flatMap((s) => s.x), 0.05);
  const yl = extent(series.flatMap((s) => [...s.y, ...(s.lo || []), ...(s.hi || [])]), 0.05);
  const left = box.x + 52, right = box.x + box.w - 8;
  const top = box.y + (title ? 20 : 8), bottom = box.y + box.h - (showXTicks ? (xlabel ? 38 : 22) : 8);
  const pw = Math.max(10, right - left), ph = Math.max(10, bottom - top);
  const X = (v) => left + ((v - xl[0]) / (xl[1] - xl[0])) * pw;
  const Y = (v) => bottom - ((v - yl[0]) / (yl[1] - yl[0])) * ph;
  ctx.save();
  ctx.font = FONT;
  ctx.fillStyle = "#222";
  ctx.strokeStyle = "#222";
  // grid and ticks
  const xt = ticks(xl[0], xl[1], Math.max(2, Math.floor(pw / 70)));
  const yt = ticks(yl[0], yl[1], Math.max(2, Math.floor(ph / 40)));
  const xstep = xt.length > 1 ? xt[1] - xt[0] : 1, ystep = yt.length > 1 ? yt[1] - yt[0] : 1;
  ctx.lineWidth = 0.8;
  ctx.strokeStyle = "rgba(176,176,176,0.3)";
  for (const t of xt) { ctx.beginPath(); ctx.moveTo(X(t), top); ctx.lineTo(X(t), bottom); ctx.stroke(); }
  for (const t of yt) { ctx.beginPath(); ctx.moveTo(left, Y(t)); ctx.lineTo(right, Y(t)); ctx.stroke(); }
  ctx.strokeStyle = "#222";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of xt) {
    ctx.beginPath(); ctx.moveTo(X(t), bottom); ctx.lineTo(X(t), bottom + 3.5); ctx.stroke();
    if (showXTicks) ctx.fillText(fmt(t, xstep), X(t), bottom + 5);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of yt) {
    ctx.beginPath(); ctx.moveTo(left, Y(t)); ctx.lineTo(left - 3.5, Y(t)); ctx.stroke();
    ctx.fillText(fmt(t, ystep), left - 6, Y(t));
  }
  if (xlabel && showXTicks) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(xlabel, (left + right) / 2, box.y + box.h - 2); }
  if (ylabel) {
    ctx.save(); ctx.translate(box.x + 10, (top + bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore();
  }
  if (title) { ctx.font = TITLE; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(title, (left + right) / 2, box.y + 3); }
  // data
  ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();
  for (const s of series) {
    if (s.lo && s.hi) {
      ctx.globalAlpha = 0.15; ctx.fillStyle = s.color; ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length; i++) if (s.hi[i] !== null) { started ? ctx.lineTo(X(s.x[i]), Y(s.hi[i])) : ctx.moveTo(X(s.x[i]), Y(s.hi[i])); started = true; }
      for (let i = s.x.length - 1; i >= 0; i--) if (s.lo[i] !== null) ctx.lineTo(X(s.x[i]), Y(s.lo[i]));
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    }
  }
  if (vline !== null && vline !== undefined) {
    ctx.strokeStyle = "#999"; ctx.lineWidth = 0.8; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(X(vline), top); ctx.lineTo(X(vline), bottom); ctx.stroke();
  }
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = (s.width || 1.5) * 1.1; ctx.globalAlpha = s.alpha ?? 1;
    ctx.setLineDash(DASHES[s.dash || "solid"] || []);
    ctx.lineJoin = "round";
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const v = s.y[i];
      if (v === null || !Number.isFinite(v)) { pen = false; continue; }
      pen ? ctx.lineTo(X(s.x[i]), Y(v)) : ctx.moveTo(X(s.x[i]), Y(v));
      pen = true;
    }
    ctx.stroke();
  }
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = "#222"; ctx.lineWidth = 0.8; ctx.strokeRect(left, top, pw, ph);
  ctx.restore();
  return { left, right, top, bottom, xl, yl, xs };
}

function legendSize(ctx, series) {
  ctx.font = SMALL;
  let w = 0;
  for (const s of series) w = Math.max(w, ctx.measureText(s.label || "").width);
  return { w: w + 34, h: series.length * 14 + 6 };
}

function drawLegend(ctx, series, x, y) {
  ctx.save();
  ctx.font = SMALL; ctx.textBaseline = "middle"; ctx.fillStyle = "#222";
  series.forEach((s, i) => {
    const yy = y + 8 + i * 14;
    ctx.strokeStyle = s.color; ctx.lineWidth = Math.max(1, (s.width || 1.5) * 1.1);
    ctx.setLineDash(DASHES[s.dash || "solid"] || []);
    ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(x + 22, yy); ctx.stroke();
    ctx.fillText(s.label || "", x + 28, yy);
  });
  ctx.restore();
}

/** The Spectra panel: one axes, legend outside on the right like `loc="upper left", bbox_to_anchor=(1.01, 1)`. */
export function drawSpectra(canvas, opt) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  if (opt.message) {
    ctx.font = FONT; ctx.fillStyle = "#808080"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(opt.message, w / 2, h / 2);
    return;
  }
  const labelled = opt.series.filter((s) => s.label);
  let lw = 0;
  if (labelled.length) lw = Math.min(legendSize(ctx, labelled).w, Math.max(110, w * 0.38));
  drawAxes(ctx, { x: 4, y: 4, w: w - 8 - lw, h: h - 8 }, opt);
  if (labelled.length) {
    ctx.save(); ctx.beginPath(); ctx.rect(w - lw, 0, lw, h); ctx.clip();
    drawLegend(ctx, labelled, w - lw + 2, 10);
    ctx.restore();
  }
}

/** One subplot per class (Plot → … by class), two columns, shared x axis. */
export function drawClassGrid(canvas, groups, x, xlabel) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  const cols = groups.length === 1 ? 1 : 2;
  const rows = Math.ceil(groups.length / cols);
  const cw = w / cols, rh = h / rows;
  const xlim = extent(x, 0.05);
  groups.forEach((g, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const lastRow = r === rows - 1 || (r === rows - 2 && c >= groups.length - (rows - 1) * cols);
    drawAxes(ctx, { x: c * cw + 4, y: r * rh + 2, w: cw - 8, h: rh - 4 }, {
      series: g.spectra.map((y) => ({ x, y, color: g.color, width: 1, alpha: 0.8 })),
      title: `${g.name} (${g.spectra.length})`, xlim, xlabel: lastRow ? xlabel : "", showXTicks: true,
    });
  });
}

/** Bar chart with a log y axis (the SVD dialog's variance per component). */
export function drawBars(canvas, values, colors, opt = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  const left = 58, right = w - 10, top = 22, bottom = h - 38;
  const pw = right - left, ph = bottom - top;
  const vals = values.map((v) => Math.max(v, 1e-6));
  const lmin = Math.floor(Math.log10(Math.min(...vals))), lmax = Math.ceil(Math.log10(Math.max(...vals)) + 1e-9);
  const lo = lmin, hi = Math.max(lmax, lmin + 1);
  const Y = (v) => bottom - ((Math.log10(v) - lo) / (hi - lo)) * ph;
  const n = vals.length, bw = pw / n;
  ctx.font = FONT; ctx.fillStyle = "#222"; ctx.strokeStyle = "#222";
  ctx.strokeStyle = "rgba(176,176,176,0.3)";
  for (let e = lo; e <= hi; e++) { ctx.beginPath(); ctx.moveTo(left, Y(10 ** e)); ctx.lineTo(right, Y(10 ** e)); ctx.stroke(); }
  vals.forEach((v, i) => {
    ctx.fillStyle = colors[i];
    ctx.fillRect(left + i * bw + bw * 0.1, Y(v), bw * 0.8, bottom - Y(v));
  });
  ctx.fillStyle = "#222"; ctx.strokeStyle = "#222"; ctx.lineWidth = 0.8;
  ctx.strokeRect(left, top, pw, ph);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let e = lo; e <= hi; e++) ctx.fillText(e === 0 ? "1" : e === 1 ? "10" : e === 2 ? "100" : `1e${e}`, left - 6, Y(10 ** e));
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const every = Math.max(1, Math.ceil(n / Math.floor(pw / 28)));
  for (let i = 0; i < n; i++) if ((i + 1) % every === 0 || i === 0) ctx.fillText(String(i + 1), left + (i + 0.5) * bw, bottom + 4);
  ctx.textBaseline = "bottom"; ctx.fillText(opt.xlabel || "", (left + right) / 2, h - 2);
  ctx.save(); ctx.translate(12, (top + bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle";
  ctx.fillText(opt.ylabel || "", 0, 0); ctx.restore();
  ctx.font = SMALL; ctx.textBaseline = "top"; ctx.fillText(opt.title || "", (left + right) / 2, 5);
}

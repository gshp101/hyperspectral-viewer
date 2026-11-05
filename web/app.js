// hsview in the browser: the window, menus, toolbars and panels of the Qt app, driving the
// Python package (src/hsview) in Pyodide. All numerical work is done by hsview.webapp.WebApp.

import { el, buildMenubar, openMenu, closeMenus, dialog, warning, information, question, getInt, getText,
  floatingWindow, isModalOpen, mnemonic, shortcutText } from "./ui.js";
import { ImageView } from "./view.js";
import { drawSpectra, drawClassGrid, drawBars } from "./plot.js";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const PY_ROOT = "../src/hsview/";
const MAC = /Mac|iPhone|iPad/.test(navigator.platform);

// --------------------------------------------------------------- settings
// The Qt app keeps these in QSettings; here they live in the browser's localStorage.
const settings = {
  _data: (() => { try { return JSON.parse(localStorage.getItem("hsview.settings") || "{}"); } catch { return {}; } })(),
  get(key, dflt) { return key in this._data ? this._data[key] : dflt; },
  set(key, value) {
    if (value === undefined) delete this._data[key]; else this._data[key] = value;
    try { localStorage.setItem("hsview.settings", JSON.stringify(this._data)); } catch { /* private mode */ }
  },
};

// ---------------------------------------------------------------- python
let pyodide = null, py = null, T = null;  // pyodide, the WebApp proxy, the UI tables

function toJs(v) {
  if (v && typeof v === "object" && typeof v.toJs === "function") {
    const out = v.toJs({ dict_converter: Object.fromEntries, create_pyproxies: false });
    v.destroy();
    return out;
  }
  return v;
}
function toPy(v) { return v !== null && typeof v === "object" && !(v instanceof Uint8Array) ? pyodide.toPy(v) : v; }

/** Call a WebApp method with JavaScript values and get JavaScript values back. */
function call(method, ...args) {
  const pyArgs = args.map(toPy);
  try {
    return toJs(py[method](...pyArgs));
  } finally {
    for (const a of pyArgs) if (a && typeof a.destroy === "function") a.destroy();
  }
}

/** The message of a Python exception, as the Qt app shows str(exc). */
function errorText(err) {
  const msg = String(err && err.message ? err.message : err);
  const lines = msg.trim().split("\n").filter((l) => l.trim());
  let last = lines[lines.length - 1] || msg;
  const m = last.match(/^[\w.]*(Error|Exception|Warning|Exit)\w*: ([\s\S]*)$/);
  if (m) last = m[2];
  return last;
}

async function busy(text, fn) {
  const b = document.getElementById("busy");
  document.getElementById("busy-text").textContent = text;
  b.hidden = false;
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 20)));
  try { return await fn(); } finally { b.hidden = true; }
}

/** Run fn; on a Python error show a warning box titled `title` and return undefined. */
async function guarded(title, fn) {
  try { return await fn(); } catch (err) {
    console.error(err);
    await warning(title, errorText(err));
    return undefined;
  }
}

const loaded = new Set();
async function ensurePackages(names) {
  const need = names.filter((n) => !loaded.has(n));
  if (!need.length) return;
  const builtin = need.filter((n) => !n.startsWith("pypi:"));
  const pypi = need.filter((n) => n.startsWith("pypi:")).map((n) => n.slice(5));
  await busy(`Loading ${need.map((n) => n.replace("pypi:", "").replace(/==.*/, "")).join(", ")}…`, async () => {
    if (builtin.length) await pyodide.loadPackage(builtin);
    if (pypi.length) {
      const micropip = pyodide.pyimport("micropip");
      await micropip.install(pypi);
      micropip.destroy();
    }
  });
  need.forEach((n) => loaded.add(n));
}

// Pure-Python readers come from PyPI, pinned to releases that work with Pyodide 0.26's numpy 1.26.
const PYPI = { spectral: "pypi:spectral==0.26", hdf5storage: "pypi:hdf5storage==0.2.2", tifffile: "pypi:tifffile==2025.1.10" };

/** Packages a file needs beyond the ones loaded at start, by extension (the Qt app's optional readers). */
function packagesFor(name, saving = false) {
  const dot = name.lastIndexOf(".");
  const ext = dot > 0 ? name.slice(dot).toLowerCase() : "";
  if ([".h5", ".hdf5", ".he5", ".nc"].includes(ext)) return ["h5py"];
  if (ext === ".mat") return ["h5py", "scipy", PYPI.hdf5storage];
  if ([".hdr", ".img", ".dat", ".raw", ".bil", ".bip", ".bsq"].includes(ext) || (!saving && !ext)) return [PYPI.spectral];
  if ([".tif", ".tiff"].includes(ext)) return [PYPI.tifffile];
  return [];
}

async function boot() {
  const status = (t) => { document.getElementById("loading-text").textContent = t; };
  status("Starting Python (Pyodide)…");
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  status("Loading numpy, matplotlib and Pillow…");
  await pyodide.loadPackage(["numpy", "matplotlib", "pillow", "sqlite3", "micropip"]);
  ["numpy", "matplotlib", "pillow", "sqlite3", "micropip"].forEach((n) => loaded.add(n));
  status("Loading hsview…");
  const files = await (await fetch("python-files.json", { cache: "no-cache" })).json();
  for (const f of files) {
    const res = await fetch(PY_ROOT + f, { cache: "no-cache" });
    if (!res.ok) throw new Error(`Could not fetch ${PY_ROOT + f} (${res.status}). Serve the repository root, not only web/.`);
    const path = "/lib/hsview/" + f;
    pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    pyodide.FS.writeFile(path, new Uint8Array(await res.arrayBuffer()));
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/lib')");
  const mod = pyodide.pyimport("hsview.webapp");
  py = mod.WebApp();
  T = toJs(mod.ui_tables());
  window.hsview = { pyodide, call, actions, get py() { return py; } };  // for the browser console
}

// --------------------------------------------------------------- files
let fileCounter = 0;
function writeFiles(files) {
  const dir = `/work/in${++fileCounter}`;
  pyodide.FS.mkdirTree(dir);
  return Promise.all([...files].map(async (f) => {
    pyodide.FS.writeFile(`${dir}/${f.name}`, new Uint8Array(await f.arrayBuffer()));
    return { name: f.name, path: `${dir}/${f.name}` };
  }));
}

/** The browser's file picker. Resolves with the chosen File objects, or null. */
function pickFiles(accept = "", multiple = false) {
  return new Promise((resolve) => {
    const input = el("input", { type: "file", accept, multiple, style: { display: "none" } });
    input.addEventListener("change", () => { resolve(input.files.length ? [...input.files] : null); input.remove(); });
    input.addEventListener("cancel", () => { resolve(null); input.remove(); });
    document.body.append(input);
    input.click();
  });
}

const MIME = { ".json": "application/json", ".csv": "text/csv" };
/**
 * Ask where to save, like QFileDialog.getSaveFileName. filters: [{label, ext}]. Uses the browser's save
 * dialog where there is one (Chrome, Edge); otherwise asks for a name and downloads the file.
 * Resolves with {name, write(bytes)} or null.
 */
async function saveAs(title, suggested, filters) {
  if (window.showSaveFilePicker) {
    try {
      const ext = suggested.slice(suggested.lastIndexOf("."));
      const ordered = [...filters].sort((a, b) => (b.ext === ext) - (a.ext === ext));
      const handle = await window.showSaveFilePicker({
        suggestedName: suggested,
        types: ordered.map((f) => ({ description: f.label, accept: { [MIME[f.ext] || "application/octet-stream"]: [f.ext] } })),
      });
      return { name: handle.name, handle, async write(bytes) { const w = await handle.createWritable(); await w.write(bytes); await w.close(); } };
    } catch (err) {
      if (err && err.name === "AbortError") return null;
      console.warn("Save picker unavailable, using a download instead", err);
    }
  }
  const name = el("input", { type: "text", value: suggested, style: { width: "320px" } });
  const type = el("select", {}, filters.map((f, i) => el("option", { value: i }, `${f.label} (*${f.ext})`)));
  const cur = filters.findIndex((f) => suggested.toLowerCase().endsWith(f.ext));
  if (cur >= 0) type.value = String(cur);
  type.addEventListener("change", () => {
    const f = filters[Number(type.value)], v = name.value, dot = v.lastIndexOf(".");
    name.value = (dot > 0 ? v.slice(0, dot) : v) + f.ext;
  });
  const ok = await dialog({
    title, body: el("div", { class: "form" }, el("label", {}, "File name"), name, el("label", {}, "Type"), type,
      el("p", { class: "full info" }, "The file is saved to your browser's download folder.")),
    buttons: [{ label: "Save", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }],
  });
  if (ok !== "ok" || !name.value.trim()) return null;
  let n = name.value.trim();
  if (!filters.some((f) => n.toLowerCase().endsWith(f.ext))) n += filters[Number(type.value)].ext;
  return {
    name: n, handle: null,
    async write(bytes) {
      const url = URL.createObjectURL(new Blob([bytes]));
      const a = el("a", { href: url, download: n });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    },
  };
}

/** Let Python write a file (writer(path)) and hand it to the user. */
async function saveWith(target, writer, busyText = "Saving…") {
  await ensurePackages(packagesFor(target.name, true));
  const dir = `/work/out${++fileCounter}`;
  pyodide.FS.mkdirTree(dir);
  const path = `${dir}/${target.name}`;
  const result = await busy(busyText, () => writer(path));
  const bytes = pyodide.FS.readFile(path);
  await target.write(bytes);
  return result;
}

// ---------------------------------------------------------------- state
let S = {};          // WebApp.state()
let D = null;        // WebApp.display_state()
let INFO = { loaded: false };
let lastVersion = -1, lastImageVersion = -1;
let view;
const panels = {};
let annotationTarget = null;  // where Ctrl+S writes (a file handle from the save dialog)
const plotWindows = new Set();

function statusMessage(text, ms = 5000) {
  const s = document.getElementById("status-message");
  s.textContent = text;
  clearTimeout(statusMessage.timer);
  if (ms) statusMessage.timer = setTimeout(() => { if (s.textContent === text) s.textContent = ""; }, ms);
}

/** Bring every part of the window up to date with Python's state. */
function sync({ image = false, display = false } = {}) {
  if (!py) return;
  S = call("state");
  document.title = S.title;
  const version = py.version, imageVersion = py.image_version;
  if (display || version !== lastVersion || !D) { D = call("display_state"); refreshDisplay(); }
  if (image || imageVersion !== lastImageVersion) {
    lastImageVersion = imageVersion;
    view.setImage(S.has_cube ? call("render") : null);
  }
  if (version !== lastVersion) {
    lastVersion = version;
    view.setAnnotations(S.has_cube ? call("annotations") : []);
    refreshOverlays();
    scheduleSpectra(true);
  }
  document.getElementById("empty-hint").classList.toggle("hidden", S.has_cube);
  updateToolbarStates();
  document.getElementById("display-panel").classList.toggle("disabled", !S.has_cube);
}

// ------------------------------------------------------------- actions
const actions = {};   // id -> {spec, run, checked(), enabled()}
let shortcuts = {};   // id -> current shortcut

function act(id) { return actions[id]; }
function isEnabled(id) { const a = actions[id]; return !!a && (a.enabled ? a.enabled() : true); }
function isChecked(id) { const a = actions[id]; return a && a.checked ? !!a.checked() : false; }
function trigger(id) {
  const a = actions[id];
  if (!a || !isEnabled(id)) return;
  closeMenus();
  Promise.resolve(a.run()).catch((err) => { console.error(err); warning(a.spec.text.replace("…", ""), errorText(err)); });
}

function iconNode(name, size = 18) {
  if (!name || !T.icons[name]) return null;
  const span = el("span", { class: "icon" });
  span.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${T.icons[name]}</svg>`;
  return span;
}
function swatch(color) { return el("span", { class: "swatch", style: { background: color } }); }
function actionIcon(id, size) {
  if (id.startsWith("class_")) return swatch(T.class_colors[Number(id.split("_")[1])]);
  const spec = actions[id] && actions[id].spec;
  return spec ? iconNode(spec.icon, size) : null;
}

function menuItem(id) {
  const a = actions[id];
  if (!a) return null;
  const spec = a.spec;
  return {
    id, label: spec.text, icon: actionIcon(id, 16), shortcut: shortcuts[id], tip: spec.tip,
    enabled: isEnabled(id), checked: spec.checkable ? isChecked(id) : undefined, radio: !!spec.group,
    action: () => trigger(id),
  };
}

function menuItems(entries) {
  const out = [];
  for (const e of entries) {
    if (e === null) out.push("sep");
    else if (typeof e === "object") out.push({ label: e.title, submenu: () => menuItems(e.items) });
    else { const m = menuItem(e); if (m) out.push(m); }
  }
  return out;
}

// ----------------------------------------------------------- keyboard
const KEY_NAMES = { Equal: "=", Minus: "-", BracketLeft: "[", BracketRight: "]", Delete: "Del", Backspace: "Backspace",
  Escape: "Esc", Enter: "Return", NumpadEnter: "Enter", Space: "Space", Tab: "Tab", Comma: ",", Period: ".", Slash: "/",
  Semicolon: ";", Quote: "'", Backquote: "`", Backslash: "\\", ArrowLeft: "Left", ArrowRight: "Right", ArrowUp: "Up",
  ArrowDown: "Down", Home: "Home", End: "End", PageUp: "PgUp", PageDown: "PgDown", Insert: "Ins", NumpadAdd: "+",
  NumpadSubtract: "-", NumpadMultiply: "*", NumpadDivide: "/", NumpadDecimal: "." };
const MODIFIER_KEYS = ["Control", "Shift", "Alt", "Meta", "OS"];

/** A key event as a Qt portable key sequence ("Ctrl+Shift+Z"); Cmd counts as Ctrl on a Mac, as in Qt. */
export function eventSequence(e) {
  if (MODIFIER_KEYS.includes(e.key)) return "";
  const c = e.code;
  let key;
  if (/^Key[A-Z]$/.test(c)) key = c.slice(3);
  else if (/^Digit\d$/.test(c)) key = c.slice(5);
  else if (/^Numpad\d$/.test(c)) key = c.slice(6);
  else if (/^F\d+$/.test(c)) key = c;
  else key = KEY_NAMES[c] || (e.key.length === 1 ? e.key.toUpperCase() : e.key);
  const ctrl = MAC ? e.metaKey : e.ctrlKey, meta = MAC ? e.ctrlKey : e.metaKey;
  return [ctrl && "Ctrl", meta && "Meta", e.altKey && "Alt", e.shiftKey && "Shift", key].filter(Boolean).join("+");
}
function normalizeSequence(seq) {
  if (!seq) return "";
  const parts = seq.split(/\+(?!$)/);
  const key = parts.pop();
  const mods = new Set(parts.map((p) => p.toLowerCase()));
  return [mods.has("ctrl") && "Ctrl", mods.has("meta") && "Meta", mods.has("alt") && "Alt", mods.has("shift") && "Shift",
    key.length === 1 ? key.toUpperCase() : key === "Delete" ? "Del" : key === "Escape" ? "Esc" : key].filter(Boolean).join("+");
}

function loadShortcuts() {
  shortcuts = {};
  for (const spec of T.specs) shortcuts[spec.id] = settings.get(`shortcuts/${spec.id}`, spec.shortcut);
}

document.addEventListener("keydown", (e) => {
  if (!T || isModalOpen()) return;
  const t = e.target;
  const typing = t instanceof HTMLInputElement && !["checkbox", "radio", "range", "button"].includes(t.type)
    || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement;
  if (e.key === "Escape") {
    if (document.querySelector(".menu")) { closeMenus(); e.preventDefault(); return; }
    if (!typing) { view.cancelDrawing(); updateToolbarStates(); }
    return;
  }
  if (typing) return;
  if ((e.key === "Enter") && view.mode === "polygon") { e.preventDefault(); view.finishPolygon(); return; }
  if (e.key === "Backspace" && !e.ctrlKey && !e.metaKey && !e.altKey) {
    if (view.mode === "polygon" && view.popVertex()) { e.preventDefault(); return; }
    if (view.mode === "select" && view.selected.size) { e.preventDefault(); trigger("delete_selected"); return; }
  }
  const seq = eventSequence(e);
  if (!seq) return;
  const id = Object.keys(shortcuts).find((k) => shortcuts[k] && normalizeSequence(shortcuts[k]) === seq);
  if (id) { e.preventDefault(); trigger(id); }
});

// ------------------------------------------------------------- menubar
function overlaysMenu(base) {
  const items = menuItems(base);
  const ovs = call("overlay_list");
  if (ovs.length) items.push("sep");
  for (const ov of ovs) {
    const current = ov.key === S.current_overlay;
    items.push({
      label: ov.name + (current ? "  (selected)" : ""),
      submenu: () => [
        { label: "Visible", checked: ov.visible, action: () => setOverlayVisible(ov.key, !ov.visible) },
        { label: "Select", action: () => { call("select_overlay", ov.key); sync(); } },
        { label: "Opacity…", action: () => overlayOpacityDialog(ov.key) },
        "sep",
        { label: "Remove", action: () => removeOverlay(ov.key) },
      ],
    });
  }
  return items;
}

function viewMenu(base) {
  const items = menuItems(base);
  items.push("sep", {
    label: "Application icon",
    submenu: () => Object.entries(T.app_icons).map(([key, ic]) => ({
      label: ic.label, radio: true, checked: key === currentAppIcon(),
      icon: el("img", { src: svgUrl(ic.svg), width: 16, height: 16 }),
      action: () => setAppIcon(key),
    })),
  });
  return items;
}

function buildMenus() {
  buildMenubar(document.getElementById("menubar"), T.menus.map((m) => ({
    title: m.title,
    build: () => {
      S = call("state");
      const name = m.title.replace("&", "");
      if (name === "Overlays") return overlaysMenu(m.items);
      if (name === "Toolbars") return toolbarsMenu();
      if (name === "View") return viewMenu(m.items);
      return menuItems(m.items);
    },
  })));
}

// ---------------------------------------------------------- app icon
const svgUrl = (svg) => "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
function currentAppIcon() { const k = settings.get("app/icon", T.default_app_icon); return T.app_icons[k] ? k : T.default_app_icon; }
function setAppIcon(key) {
  if (!T.app_icons[key]) key = T.default_app_icon;
  settings.set("app/icon", key);
  document.getElementById("favicon").href = svgUrl(T.app_icons[key].svg);
}

// ------------------------------------------------------------- toolbars
let tbConfig = null;
const SEP = () => T.separator, SPC = () => T.spacer;

function knownItems() { return new Set([...T.specs.map((s) => s.id), ...Object.keys(T.widget_items)]); }
/** Same rules as hsview.gui.toolbar.validate_config. */
function validateConfig(cfg) {
  const out = JSON.parse(JSON.stringify(T.toolbars));
  if (!cfg || typeof cfg !== "object") return out;
  if (T.icon_sizes.includes(cfg.icon_size)) out.icon_size = cfg.icon_size;
  if (cfg.style in T.styles) out.style = cfg.style;
  const known = knownItems(), names = new Set(), bars = [];
  const norm = (n) => "toolbar_" + n.trim().toLowerCase().replace(/\W+/g, "_");
  for (const tb of cfg.toolbars || []) {
    if (!tb || typeof tb !== "object") continue;
    let name = String(tb.name || "").trim() || "Toolbar";
    const base = name;
    for (let i = 2; names.has(norm(name)); i++) name = `${base} ${i}`;
    names.add(norm(name));
    bars.push({ name, visible: tb.visible !== false, items: (tb.items || []).filter((x) => x === SEP() || x === SPC() || known.has(x)) });
  }
  if ("toolbars" in cfg) out.toolbars = bars;
  return out;
}
function setToolbarConfig(cfg) {
  tbConfig = validateConfig(cfg);
  settings.set("toolbars/config", JSON.stringify(tbConfig));
  renderToolbars();
}
function editToolbars(fn) { const cfg = JSON.parse(JSON.stringify(tbConfig)); fn(cfg); setToolbarConfig(cfg); }

function itemLabel(id) {
  if (id === SEP()) return "Separator";
  if (id === SPC()) return "Spacer";
  const spec = T.specs.find((s) => s.id === id);
  return (spec ? spec.text : (T.widget_items[id] || {}).text || id).replace("…", "");
}
function allItems() {
  return [...T.specs.map((s) => ({ id: s.id, text: s.text, category: s.category })),
    ...Object.entries(T.widget_items).map(([id, w]) => ({ id, text: w.text, category: w.category }))];
}

const toolbarButtons = [];
function renderToolbars() {
  const box = document.getElementById("toolbars");
  box.textContent = "";
  toolbarButtons.length = 0;
  const size = tbConfig.icon_size, style = tbConfig.style;
  for (const bar of tbConfig.toolbars) {
    if (!bar.visible) continue;
    const tb = el("div", { class: "toolbar", "aria-label": bar.name }, el("span", { class: "grip" }));
    for (const item of bar.items) {
      if (item === SEP()) tb.append(el("span", { class: "tb-sep" }));
      else if (item === SPC()) { tb.append(el("span", { class: "tb-spacer" })); tb.classList.add("has-spacer"); }
      else if (item === "class_selector") tb.append(classSelector());
      else if (item === "band_readout") tb.append(el("span", { class: "tb-readout" }, D ? D.readout : "Band: --"));
      else if (actions[item]) {
        const spec = actions[item].spec;
        const icon = actionIcon(item, size);
        let text = spec.text.replace("…", "");
        if (spec.category.startsWith("Channel ")) text = `Ch${spec.category.split(" ")[1]} ${text}`;
        const showIcon = style !== "text" && icon, showText = style !== "icon" || !icon;
        const b = el("button", { class: "tb-button" + (style === "under" ? " under" : ""), type: "button",
          "data-tip": spec.text + (shortcuts[item] ? ` (${shortcutText(shortcuts[item])})` : "") + (spec.tip ? `\n${spec.tip}` : ""),
          "aria-label": spec.text }, showIcon ? icon : null, showText ? el("span", {}, text) : null);
        b.addEventListener("click", () => trigger(item));
        b.dataset.id = item;
        toolbarButtons.push(b);
        tb.append(b);
      }
    }
    tb.addEventListener("contextmenu", (e) => { e.preventDefault(); popupMenu(e.clientX, e.clientY); });
    box.append(tb);
  }
  box.oncontextmenu = (e) => { e.preventDefault(); popupMenu(e.clientX, e.clientY); };
  updateToolbarStates();
}

function updateToolbarStates() {
  for (const b of toolbarButtons) {
    const id = b.dataset.id;
    b.disabled = !isEnabled(id);
    b.classList.toggle("checked", actions[id].spec.checkable && isChecked(id));
  }
  document.querySelectorAll(".tb-readout").forEach((r) => { r.textContent = D ? D.readout : "Band: --"; });
  document.querySelectorAll(".class-selector").forEach((c) => {
    c.firstChild.style.background = T.class_colors[view.currentClass];
    c.lastChild.textContent = T.class_names[view.currentClass];
  });
}

function classSelector() {
  const c = el("button", { class: "class-selector", type: "button", "data-tip": "Class for new annotations (keys 1–7); also recolours selected annotations" },
    swatch(T.class_colors[view.currentClass]), el("span", {}, T.class_names[view.currentClass]));
  c.addEventListener("click", () => {
    const r = c.getBoundingClientRect();
    openMenu(T.class_names.map((n, i) => ({ label: n, icon: swatch(T.class_colors[i]), checked: i === view.currentClass, radio: true, action: () => setClass(i) })), r.left, r.bottom);
  });
  return c;
}

/** Right-click on a toolbar: panels and toolbars to show, like QMainWindow.createPopupMenu. */
function popupMenu(x, y) {
  const items = [
    ...["toggle_display", "toggle_overlays", "toggle_spectra"].map((id) => ({ label: actions[id].spec.text, checked: isChecked(id), action: () => trigger(id) })),
    "sep",
    ...tbConfig.toolbars.map((tb, i) => ({ label: tb.name, checked: tb.visible, action: () => editToolbars((c) => { c.toolbars[i].visible = !tb.visible; }) })),
    "sep", menuItem("customize"),
  ];
  openMenu(items, x, y);
}

/** The Toolbars menu: every toolbar edit, as in hsview.gui.toolbar.ToolbarManager.fill_menu. */
function toolbarsMenu() {
  const items = tbConfig.toolbars.map((spec, bar) => ({ label: spec.name, submenu: () => toolbarSubmenu(bar) }));
  items.push("sep",
    { label: "New toolbar…", action: newToolbarPrompt },
    { label: "Icon size", submenu: () => T.icon_sizes.map((s) => ({ label: `${s} px`, radio: true, checked: s === tbConfig.icon_size, action: () => editToolbars((c) => { c.icon_size = s; }) })) },
    { label: "Button style", submenu: () => Object.entries(T.styles).map(([k, label]) => ({ label, radio: true, checked: k === tbConfig.style, action: () => editToolbars((c) => { c.style = k; }) })) },
    "sep",
    { label: "Show all toolbars", action: () => editToolbars((c) => c.toolbars.forEach((t) => { t.visible = true; })) },
    { label: "Hide all toolbars", action: () => editToolbars((c) => c.toolbars.forEach((t) => { t.visible = false; })) },
    { label: "Reset toolbars to default…", action: async () => { if (await question("Reset toolbars", "Replace all toolbars with the default set?")) setToolbarConfig(T.toolbars); } },
    "sep", menuItem("customize"));
  return items;
}

function toolbarSubmenu(bar) {
  const spec = tbConfig.toolbars[bar];
  const cats = {};
  for (const it of allItems()) (cats[it.category] ||= []).push(it);
  const n = spec.items.length;
  const move = (pos, delta) => editToolbars((c) => {
    const items = c.toolbars[bar].items, to = pos + delta;
    if (to >= 0 && to < items.length) items.splice(to, 0, items.splice(pos, 1)[0]);
  });
  return [
    { label: "Show", checked: spec.visible, action: () => editToolbars((c) => { c.toolbars[bar].visible = !spec.visible; }) },
    "sep",
    { label: "Commands", submenu: () => Object.entries(cats).map(([cat, entries]) => ({
      label: cat,
      submenu: () => entries.map((it) => ({
        label: it.text.replace("…", ""), icon: actions[it.id] ? actionIcon(it.id, 16) : null, tip: actions[it.id] ? actions[it.id].spec.tip : "",
        checked: spec.items.includes(it.id),
        action: () => editToolbars((c) => {
          const items = c.toolbars[bar].items;
          if (items.includes(it.id)) c.toolbars[bar].items = items.filter((x) => x !== it.id); else items.push(it.id);
        }),
      })),
    })) },
    { label: "Add separator", action: () => editToolbars((c) => c.toolbars[bar].items.push(SEP())) },
    { label: "Add spacer", action: () => editToolbars((c) => c.toolbars[bar].items.push(SPC())) },
    { label: "Arrange items", enabled: n > 0, submenu: () => spec.items.map((id, pos) => ({
      label: `${pos + 1}. ${itemLabel(id)}`, icon: actions[id] ? actionIcon(id, 16) : null,
      submenu: () => [
        { label: "Move to start", enabled: pos > 0, action: () => move(pos, -pos) },
        { label: "Move left / up", enabled: pos > 0, action: () => move(pos, -1) },
        { label: "Move right / down", enabled: pos < n - 1, action: () => move(pos, 1) },
        { label: "Move to end", enabled: pos < n - 1, action: () => move(pos, n - 1 - pos) },
        "sep",
        { label: "Remove", action: () => editToolbars((c) => c.toolbars[bar].items.splice(pos, 1)) },
      ],
    })) },
    "sep",
    { label: "Rename…", action: () => renameToolbarPrompt(bar) },
    { label: "Delete toolbar…", action: async () => { if (await question("Delete toolbar", `Delete the toolbar '${spec.name}'?`)) editToolbars((c) => c.toolbars.splice(bar, 1)); } },
  ];
}

const tbNorm = (n) => n.trim().toLowerCase().replace(/\W+/g, "_");
const toolbarTaken = (cfg, name, ignore = -1) => cfg.toolbars.some((t, i) => i !== ignore && tbNorm(t.name) === tbNorm(name));
async function newToolbarPrompt() {
  const name = ((await getText("New toolbar", "Name:")) || "").trim();
  if (!name) return;
  if (toolbarTaken(tbConfig, name)) { await warning("New toolbar", `There is already a toolbar called '${name}'.`); return; }
  editToolbars((c) => c.toolbars.push({ name, visible: true, items: [] }));
}
async function renameToolbarPrompt(bar) {
  const old = tbConfig.toolbars[bar].name;
  const name = ((await getText("Rename toolbar", "Name:", old)) || "").trim();
  if (!name || name === old) return;
  if (toolbarTaken(tbConfig, name, bar)) { await warning("Rename toolbar", `There is already a toolbar called '${name}'.`); return; }
  editToolbars((c) => { c.toolbars[bar].name = name; });
}

// ------------------------------------------------------ customize dialog
async function customizeDialog() {
  const cfg = JSON.parse(JSON.stringify(tbConfig));
  const keys = { ...shortcuts };
  let bar = 0, curSel = -1, availSel = null;

  const barSelect = el("select", { class: "grow" });
  const visible = el("input", { type: "checkbox" });
  const avail = el("div", { class: "list" }), current = el("div", { class: "list" });
  const sizeSel = el("select", {}, T.icon_sizes.map((s) => el("option", { value: s }, `${s} px`)));
  const styleSel = el("select", {}, Object.entries(T.styles).map(([k, l]) => el("option", { value: k }, l)));
  sizeSel.value = cfg.icon_size; styleSel.value = cfg.style;
  sizeSel.onchange = () => { cfg.icon_size = Number(sizeSel.value); };
  styleSel.onchange = () => { cfg.style = styleSel.value; };

  const fillBars = () => {
    barSelect.textContent = "";
    cfg.toolbars.forEach((t, i) => barSelect.append(el("option", { value: i }, t.name)));
    if (bar >= cfg.toolbars.length) bar = cfg.toolbars.length - 1;
    barSelect.value = String(bar);
    fillCurrent();
  };
  const fillAvail = () => {
    avail.textContent = "";
    const add = (id, text, cat) => {
      const row = el("div", { class: "list-item" + (availSel === id ? " current" : "") }, actions[id] ? actionIcon(id, 16) : el("span", { style: { width: "16px", display: "inline-block" } }), text, el("span", { class: "cat" }, cat));
      row.onclick = () => { availSel = id; fillAvail(); };
      row.ondblclick = () => { availSel = id; addItem(); };
      avail.append(row);
    };
    add(SEP(), "— Separator —", ""); add(SPC(), "↔ Spacer", "");
    const used = bar >= 0 ? new Set(cfg.toolbars[bar].items) : new Set();
    for (const it of allItems()) if (!used.has(it.id)) add(it.id, it.text.replace("…", ""), it.category);
  };
  const fillCurrent = () => {
    current.textContent = "";
    if (bar < 0) { visible.checked = false; fillAvail(); return; }
    visible.checked = cfg.toolbars[bar].visible;
    cfg.toolbars[bar].items.forEach((id, i) => {
      const row = el("div", { class: "list-item" + (i === curSel ? " current" : "") }, actions[id] ? actionIcon(id, 16) : el("span", { style: { width: "16px", display: "inline-block" } }), itemLabel(id));
      row.onclick = () => { curSel = i; fillCurrent(); };
      row.ondblclick = () => { curSel = i; removeItem(); };
      current.append(row);
    });
    fillAvail();
  };
  const addItem = () => {
    if (bar < 0 || availSel === null) return;
    const items = cfg.toolbars[bar].items;
    const at = curSel >= 0 ? curSel + 1 : items.length;
    items.splice(at, 0, availSel);
    curSel = at;
    if (availSel !== SEP() && availSel !== SPC()) availSel = null;
    fillCurrent();
  };
  const removeItem = () => {
    if (bar < 0 || curSel < 0) return;
    cfg.toolbars[bar].items.splice(curSel, 1);
    curSel = Math.min(curSel, cfg.toolbars[bar].items.length - 1);
    fillCurrent();
  };
  const moveItem = (d) => {
    if (bar < 0 || curSel < 0) return;
    const items = cfg.toolbars[bar].items, to = curSel + d;
    if (to < 0 || to >= items.length) return;
    items.splice(to, 0, items.splice(curSel, 1)[0]);
    curSel = to;
    fillCurrent();
  };
  barSelect.onchange = () => { bar = Number(barSelect.value); curSel = -1; fillCurrent(); };
  visible.onchange = () => { if (bar >= 0) cfg.toolbars[bar].visible = visible.checked; };
  const btn = (label, fn) => { const b = el("button", { class: "push", type: "button" }, label); b.onclick = fn; return b; };

  const tbTab = el("div", {},
    el("div", { class: "row" }, el("label", {}, "Toolbar"), barSelect,
      btn("New…", async () => {
        const name = ((await getText("New toolbar", "Name:")) || "").trim();
        if (!name) return;
        if (toolbarTaken(cfg, name)) { await warning("New toolbar", `There is already a toolbar called '${name}'.`); return; }
        cfg.toolbars.push({ name, visible: true, items: [] }); bar = cfg.toolbars.length - 1; curSel = -1; fillBars();
      }),
      btn("Rename…", async () => {
        if (bar < 0) return;
        const name = ((await getText("Rename toolbar", "Name:", cfg.toolbars[bar].name)) || "").trim();
        if (!name) return;
        if (toolbarTaken(cfg, name, bar)) { await warning("Rename toolbar", `There is already a toolbar called '${name}'.`); return; }
        cfg.toolbars[bar].name = name; fillBars();
      }),
      btn("Delete", () => { if (bar >= 0) { cfg.toolbars.splice(bar, 1); bar = Math.min(bar, cfg.toolbars.length - 1); curSel = -1; fillBars(); } })),
    el("div", { class: "row" }, el("label", { class: "check" }, visible, "Show this toolbar")),
    el("div", { class: "cust-grid" },
      el("div", {}, el("div", {}, "Available commands"), avail),
      el("div", { class: "buttons" }, btn("Add →", addItem), btn("← Remove", removeItem), btn("Move up", () => moveItem(-1)), btn("Move down", () => moveItem(1))),
      el("div", {}, el("div", {}, "Items on this toolbar"), current)),
    el("div", { class: "row" }, el("label", {}, "Icon size"), sizeSel, el("label", {}, "Button style"), styleSel));

  // shortcuts tab
  const rows = [];
  const table = el("table", { class: "shortcuts" }, el("tr", {}, el("th", {}, "Command"), el("th", {}, "Category"), el("th", {}, "Shortcut")));
  const checkConflicts = () => {
    const seen = {};
    for (const r of rows) { const s = normalizeSequence(keys[r.id]); if (s) (seen[s] ||= []).push(r.id); }
    for (const r of rows) r.tr.classList.toggle("conflict", (seen[normalizeSequence(keys[r.id])] || []).length > 1);
    return Object.values(seen).filter((ids) => ids.length > 1);
  };
  for (const spec of T.specs) {
    const input = el("input", { type: "text", value: keys[spec.id] || "", placeholder: "press keys", readonly: true });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Tab") return;
      e.preventDefault(); e.stopPropagation();
      if ((e.key === "Backspace" || e.key === "Delete") && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && input.value) { input.value = ""; keys[spec.id] = ""; checkConflicts(); return; }
      const seq = eventSequence(e);
      if (!seq) return;
      input.value = seq; keys[spec.id] = seq; checkConflicts();
    });
    const clear = el("button", { class: "dock-close", type: "button", title: "Clear" }, "×");
    clear.onclick = () => { input.value = ""; keys[spec.id] = ""; checkConflicts(); };
    const tr = el("tr", {}, el("td", {}, el("span", { style: { display: "inline-flex", gap: "6px", alignItems: "center" } }, actionIcon(spec.id, 16), spec.text.replace("…", ""))), el("td", { class: "info" }, spec.category), el("td", {}, input, clear));
    rows.push({ id: spec.id, input, tr });
    table.append(tr);
  }
  const restore = btn("Restore default shortcuts", () => { for (const r of rows) { keys[r.id] = T.specs.find((s) => s.id === r.id).shortcut; r.input.value = keys[r.id]; } checkConflicts(); });
  const scTab = el("div", {}, el("div", { class: "shortcut-scroll" }, table), el("div", { class: "row" }, restore,
    el("span", { class: "info" }, MAC ? "Ctrl means ⌘ on a Mac. Click a field and press the keys; Backspace clears it." : "Click a field and press the keys; Backspace clears it.")));
  checkConflicts();

  const tabs = el("div", { class: "tabs" });
  const pages = [["Toolbars", tbTab], ["Shortcuts", scTab]];
  const showTab = (i) => { [...tabs.children].forEach((t, j) => t.classList.toggle("current", i === j)); pages.forEach(([, p], j) => { p.hidden = i !== j; }); };
  pages.forEach(([name], i) => { const t = el("div", { class: "tab" }, name); t.onclick = () => showTab(i); tabs.append(t); });
  showTab(0);
  fillBars();
  const ok = await dialog({
    title: "Customize toolbars and shortcuts", body: el("div", {}, tabs, tbTab, scTab),
    validate: (v) => {
      if (v !== "ok") return true;
      const clashes = checkConflicts();
      if (clashes.length) {
        warning("Shortcuts", "These commands share a shortcut:\n\n" + clashes.map((ids) => ids.map(itemLabel).join(", ")).join("\n"));
        return false;
      }
      return true;
    },
    buttons: [{ label: "OK", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }],
  });
  if (ok !== "ok") return;
  for (const spec of T.specs) {
    const seq = keys[spec.id] || "";
    settings.set(`shortcuts/${spec.id}`, seq === spec.shortcut ? undefined : seq);
  }
  loadShortcuts();
  setToolbarConfig(cfg);
}

// ------------------------------------------------------------ docks
const DOCKS = { toggle_display: "dock-display", toggle_overlays: "dock-overlays", toggle_spectra: "dock-spectra" };
function dockVisible(id) { return settings.get(`docks/${id}`, true); }
function setDockVisible(id, on) {
  settings.set(`docks/${id}`, on);
  layoutDocks();
}
function layoutDocks() {
  const d = dockVisible("toggle_display"), o = dockVisible("toggle_overlays"), s = dockVisible("toggle_spectra");
  document.getElementById("dock-display").classList.toggle("hidden", !d);
  document.getElementById("dock-overlays").classList.toggle("hidden", !o);
  document.getElementById("split-left").classList.toggle("hidden", !(d && o));
  document.getElementById("left").classList.toggle("hidden", !(d || o));
  document.getElementById("split-l").classList.toggle("hidden", !(d || o));
  document.getElementById("right").classList.toggle("hidden", !s);
  document.getElementById("split-r").classList.toggle("hidden", !s);
  document.getElementById("dock-overlays").style.flex = d ? `0 0 ${settings.get("layout/overlays_h", 180)}px` : "1 1 0";
  updateToolbarStates();
  scheduleSpectra();
}
function setupSplitters() {
  const left = document.getElementById("left"), right = document.getElementById("right");
  left.style.width = settings.get("layout/left_w", 330) + "px";
  right.style.width = settings.get("layout/right_w", 460) + "px";
  const drag = (handle, fn) => handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const move = (ev) => fn(ev);
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); scheduleSpectra(); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
  drag(document.getElementById("split-l"), (e) => {
    const w = Math.max(200, Math.min(window.innerWidth * 0.5, e.clientX - left.getBoundingClientRect().left));
    left.style.width = w + "px"; settings.set("layout/left_w", Math.round(w));
  });
  drag(document.getElementById("split-r"), (e) => {
    const w = Math.max(220, Math.min(window.innerWidth * 0.6, right.getBoundingClientRect().right - e.clientX));
    right.style.width = w + "px"; settings.set("layout/right_w", Math.round(w));
  });
  drag(document.getElementById("split-left"), (e) => {
    const r = left.getBoundingClientRect();
    const h = Math.max(80, Math.min(r.height - 120, r.bottom - e.clientY));
    document.getElementById("dock-overlays").style.flex = `0 0 ${h}px`; settings.set("layout/overlays_h", Math.round(h));
  });
  document.querySelectorAll(".dock-close").forEach((b) => b.addEventListener("click", () => setDockVisible(b.dataset.toggle, false)));
}

// ------------------------------------------------------- display panel
function slider(min, max, value) { return el("input", { type: "range", min, max, value, class: "grow" }); }
function spin(min, max, value) { return el("input", { type: "number", min, max, value, step: 1 }); }
function select(options, value) {
  const s = el("select", {}, options.map((o) => (typeof o === "string" ? el("option", { value: o }, o) : el("option", { value: o.value }, o.label))));
  if (value !== undefined) s.value = value;
  return s;
}
function group(title, ...children) { return el("div", { class: "group" }, el("div", { class: "group-title" }, title), ...children); }

function displayChange(fn) {
  if (!S.has_cube) return;
  try { D = fn(); } catch (err) { console.error(err); warning("Display", errorText(err)); D = call("display_state"); }
  refreshDisplay();
  sync();
}

function buildDisplayPanel() {
  const root = document.getElementById("display-panel");
  panels.channels = [];
  for (let c = 0; c < 3; c++) {
    const w = {};
    w.source = select(T.sources); w.route = select(T.routes);
    w.route.dataset.tip = "Display channel this image feeds; Gray shows it alone";
    w.slider = slider(0, 0, 0); w.spin = spin(0, 0, 0);
    w.wslider = slider(1, 1, 1); w.wspin = spin(1, 1, 1);
    w.wslider.dataset.tip = w.wspin.dataset.tip = "Average this many bands centred on the selected band";
    w.spec = el("input", { type: "text", class: "grow", placeholder: "10-20, 35  or  650-700nm" });
    w.mode = select(["sum", "mean"]);
    w.filterLabel = el("span", { class: "grow" }, "No filter");
    w.filterBtn = el("button", { class: "push", type: "button" }, "Load…");
    w.info = el("div", { class: "info" });
    w.pages = [
      el("div", {}, el("div", { class: "row" }, w.slider, w.spin), el("div", { class: "row" }, el("span", {}, "Width"), w.wslider, w.wspin)),
      el("div", { class: "row" }, w.spec, w.mode),
      el("div", { class: "row" }, w.filterLabel, w.filterBtn),
    ];
    root.append(group(`Band channel ${c + 1}`, el("div", { class: "row" }, el("div", { class: "grow", style: { display: "flex" } }, w.source), el("span", {}, "→"), el("div", { class: "grow", style: { display: "flex" } }, w.route)), ...w.pages, w.info));
    w.source.style.flex = w.route.style.flex = "1 1 auto";
    w.source.onchange = async () => {
      if (w.source.value === "Filter" && !D.channels[c].has_filter) { await loadFilterDialog(c); refreshDisplay(); return; }
      displayChange(() => call("set_channel", c, { source: w.source.value }));
    };
    w.route.onchange = () => displayChange(() => call("set_channel", c, { route: w.route.value }));
    const band = (v) => displayChange(() => call("set_channel", c, { band: Number(v) }));
    w.slider.oninput = () => { w.spin.value = w.slider.value; band(w.slider.value); };
    w.spin.onchange = () => band(w.spin.value);
    const width = (v) => displayChange(() => call("set_channel", c, { width: Number(v) }));
    w.wslider.oninput = () => { w.wspin.value = w.wslider.value; width(w.wslider.value); };
    w.wspin.onchange = () => width(w.wspin.value);
    w.spec.onchange = () => displayChange(() => call("set_channel", c, { spec: w.spec.value }));
    w.spec.onkeydown = (e) => { if (e.key === "Enter") w.spec.blur(); };
    w.mode.onchange = () => displayChange(() => call("set_channel", c, { sum_mode: w.mode.value }));
    w.filterBtn.onclick = () => loadFilterDialog(c);
    panels.channels.push(w);
  }
  const r = {};
  r.label = el("span", { class: "grow" }, "None");
  r.route = select(T.ref_routes);
  r.route.dataset.tip = "Show replaces the band display; Overlay blends the image over it";
  r.load = el("button", { class: "push grow", type: "button" }, "Load…");
  r.reverse = el("button", { class: "push grow", type: "button", "data-tip": "Swap red and blue (for BGR images)" }, "Reverse RGB");
  r.opacity = slider(0, 100, 50);
  r.opacity.dataset.tip = "How strongly the reference image covers the bands when routed to Overlay";
  root.append(group("Reference image", el("div", { class: "row" }, r.label, r.route), el("div", { class: "row" }, r.load, r.reverse),
    el("div", { class: "row" }, el("span", {}, "Opacity"), r.opacity)));
  r.route.onchange = () => displayChange(() => call("set_reference", { route: r.route.value }));
  r.load.onclick = () => trigger("open_reference");
  r.reverse.onclick = () => trigger("reverse_rgb");
  r.opacity.oninput = () => displayChange(() => call("set_reference", { opacity: Number(r.opacity.value) }));
  panels.reference = r;

  const k = {};
  k.scaling = select(T.scalings.map((s) => ({ value: s.key, label: s.label })));
  k.contrast = slider(0, 89, T.default_contrast);
  k.bias = slider(-128, 127, T.default_bias);
  k.reset = el("button", { class: "push full", type: "button" }, "Reset contrast");
  root.append(group("Contrast", el("div", { class: "form" }, el("span", {}, "Scaling"), k.scaling, el("span", {}, "Contrast"), k.contrast,
    el("span", {}, "Bias"), k.bias, k.reset)));
  k.scaling.onchange = () => displayChange(() => call("set_view", { scaling: k.scaling.value }));
  k.contrast.oninput = () => displayChange(() => call("set_view", { contrast: Number(k.contrast.value) }));
  k.bias.oninput = () => displayChange(() => call("set_view", { bias: Number(k.bias.value) }));
  k.reset.onclick = () => trigger("reset_contrast");
  panels.contrast = k;
}

function refreshDisplay() {
  if (!D || !panels.channels) return;
  const has = !!S.has_cube;
  const n = D.n_bands;
  D.channels.forEach((ch, c) => {
    const w = panels.channels[c];
    w.source.value = ch.source; w.route.value = ch.route;
    for (const s of [w.slider, w.spin]) { s.max = n - 1; if (document.activeElement !== s) s.value = ch.band; }
    for (const s of [w.wslider, w.wspin]) { s.max = n; if (document.activeElement !== s) s.value = ch.width; }
    if (document.activeElement !== w.spec) w.spec.value = ch.spec;
    w.mode.value = ch.sum_mode;
    w.filterLabel.textContent = ch.filter_name || "No filter";
    w.info.textContent = ch.info;
    w.pages.forEach((p, i) => { p.hidden = T.sources[i] !== ch.source; });
    for (const x of [w.source, w.route, w.slider, w.spin, w.wslider, w.wspin, w.spec, w.mode, w.filterBtn]) x.disabled = !has;
  });
  const r = panels.reference, ref = D.reference;
  r.label.textContent = ref.name; r.route.value = ref.route; r.opacity.value = ref.opacity;
  r.route.disabled = !has || !ref.has; r.reverse.disabled = !has || !ref.rgb; r.opacity.disabled = !has || !ref.has; r.load.disabled = !has;
  const k = panels.contrast;
  k.scaling.value = D.scaling; k.contrast.value = D.contrast; k.bias.value = D.bias;
  for (const x of [k.scaling, k.contrast, k.bias, k.reset]) x.disabled = !has;
  updateToolbarStates();
}

// ------------------------------------------------------- overlays panel
function buildOverlayPanel() {
  const root = document.getElementById("overlay-panel");
  const o = {};
  o.list = el("div", { class: "list" });
  o.opacity = slider(0, 100, 50);
  o.add = el("button", { class: "push grow", type: "button" }, "Add…");
  o.remove = el("button", { class: "push grow", type: "button" }, "Remove");
  root.append(o.list, el("div", { class: "row" }, el("span", {}, "Opacity"), o.opacity), el("div", { class: "row" }, o.add, o.remove));
  o.add.onclick = () => trigger("add_overlay");
  o.remove.onclick = () => trigger("remove_overlay");
  o.opacity.oninput = () => {
    if (!S.current_overlay) return;
    call("set_overlay_opacity", S.current_overlay, Number(o.opacity.value) / 100);
    const ov = view.overlays.get(S.current_overlay);
    if (ov) { ov.opacity = Number(o.opacity.value) / 100; view.draw(); }
  };
  panels.overlays = o;
}

function refreshOverlays() {
  const list = S.has_cube ? call("overlay_list") : [];
  const keys = new Set(list.map((ov) => ov.key));
  for (const key of [...view.overlays.keys()]) if (!keys.has(key)) view.removeOverlay(key);
  for (const ov of list) {
    const img = view.overlays.has(ov.key) ? null : call("overlay_rgba", ov.key);
    view.setOverlay(ov.key, img, ov.opacity, ov.visible);
  }
  const o = panels.overlays;
  o.list.textContent = "";
  for (const ov of list) {
    const cb = el("input", { type: "checkbox", checked: ov.visible });
    cb.onclick = (e) => { e.stopPropagation(); setOverlayVisible(ov.key, cb.checked); };
    const row = el("div", { class: "list-item" + (ov.key === S.current_overlay ? " current" : "") }, cb, `${ov.name}  (${ov.labels ? "labels" : "image"})`);
    row.onclick = () => { call("select_overlay", ov.key); sync(); };
    o.list.append(row);
  }
  const cur = list.find((ov) => ov.key === S.current_overlay);
  o.opacity.disabled = !cur; o.remove.disabled = !cur; o.add.disabled = !S.has_cube;
  if (cur) o.opacity.value = Math.round(cur.opacity * 100);
}

function setOverlayVisible(key, on) { call("set_overlay_visible", key, on); sync(); }
function removeOverlay(key) { call("remove_overlay", key); sync(); }

// -------------------------------------------------------- spectra panel
let spectraTimer = null, seriesCache = null, cursor = null;
function buildSpectraPanel() {
  const root = document.getElementById("spectra-panel");
  const p = {};
  const box = (label, tip, on) => { const cb = el("input", { type: "checkbox", checked: on }); return [cb, el("label", { class: "check", "data-tip": tip }, cb, label)]; };
  let lab;
  [p.cursor, lab] = box("Cursor", "Show the spectrum of the pixel under the mouse", true); const l1 = lab;
  [p.std, lab] = box("±1σ", "Shade one standard deviation around region means", true); const l2 = lab;
  [p.byClass, lab] = box("By class", "One mean spectrum per annotation class instead of one line per annotation", false); const l3 = lab;
  [p.labels, lab] = box("Overlay classes", "Mean spectrum of each class in the selected segmentation overlay", false); const l4 = lab;
  p.canvas = el("canvas");
  root.append(el("div", { class: "row" }, l1, l2, l3, l4), el("div", { class: "plot-wrap" }, p.canvas));
  for (const cb of [p.cursor, p.std, p.byClass, p.labels]) cb.onchange = () => { scheduleSpectra(true); updateToolbarStates(); };
  new ResizeObserver(() => scheduleSpectra()).observe(p.canvas);
  panels.spectra = p;
}

function scheduleSpectra(dirty = false) {
  if (dirty) seriesCache = null;
  clearTimeout(spectraTimer);
  spectraTimer = setTimeout(drawSpectraPanel, 40);
}

function drawSpectraPanel() {
  const p = panels.spectra;
  if (!p || !p.canvas.clientWidth) return;
  if (!S.has_cube) { drawSpectra(p.canvas, { message: "Open a cube to see spectra" }); return; }
  try {
    if (!seriesCache) seriesCache = call("spectra_series", p.std.checked, p.byClass.checked, p.labels.checked);
    const x = INFO.axis;
    const series = seriesCache.map((s) => ({ ...s, x }));
    if (p.cursor.checked && cursor) {
      const y = call("cursor_spectrum", cursor[0], cursor[1]);
      if (y) series.push({ x, y, color: "#595959", width: 1, dash: "dotted", label: `cursor (${cursor[0]}, ${cursor[1]})` });
    }
    const marker = D && D.band_marker !== null && D.band_marker !== undefined ? x[D.band_marker] : null;
    drawSpectra(p.canvas, { series, xlabel: INFO.axis_label, ylabel: "Value", vline: marker });
  } catch (err) { console.error(err); }
}

// ------------------------------------------------------------- plots
function plotByClass(which) {
  const res = call("class_groups", which);
  if (!res.groups.length) { information("Nothing to plot", which === "point" ? "There are no points." : "There are no rectangles or polygons."); return; }
  const make = (groups, title, x, y) => {
    const canvas = el("canvas");
    const save = el("button", { class: "push", type: "button", "data-tip": "Save the figure as a PNG image" }, "Save image…");
    const n = groups.length;
    const win = floatingWindow({
      title, x, y, w: n === 1 ? 720 : 900, h: n === 1 ? 440 : 300 + 160 * Math.max(0, Math.floor((n - 1) / 2)), actions: [save],
      onResize: () => drawClassGrid(canvas, groups, INFO.axis, INFO.axis_label),
      onClose: () => plotWindows.delete(win),
    });
    win.body.append(canvas);
    save.onclick = async () => {
      const target = await saveAs("Save figure", title.replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "") + ".png", [{ label: "PNG image", ext: ".png" }]);
      if (target) canvas.toBlob(async (b) => target.write(new Uint8Array(await b.arrayBuffer())), "image/png");
    };
    plotWindows.add(win);
  };
  if (isChecked("plot_separate")) {
    res.groups.forEach((g, i) => make([g], `${res.title} — ${g.name} — ${INFO.name}`, 120 + 30 * i, 100 + 30 * i));
  } else make(res.groups, `${res.title} by class — ${INFO.name}`);
}

// ------------------------------------------------------------ dialogs
function bandSpecDialog(title, labels, values, mode) {
  const edits = labels.map((_, i) => el("input", { type: "text", value: values[i] || "", placeholder: "10-20, 35  or  650-700nm", style: { width: "240px" } }));
  const combine = select(["sum", "mean"], mode);
  const body = el("div", { class: "form" }, ...labels.flatMap((l, i) => [el("label", {}, l), edits[i]]), el("label", {}, "Combine by"), combine);
  return dialog({ title, body, buttons: [{ label: "OK", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }] })
    .then((v) => (v === "ok" ? { values: edits.map((e) => e.value.trim()), mode: combine.value } : null));
}

async function svdDialog() {
  const fractions = await busy("Computing the SVD…", () => call("svd_fit"));
  const n = fractions.length;
  const k = el("input", { type: "number", min: 0, max: Math.max(0, n - 1), value: Math.min(settings.get("analysis/svd_k", 1), Math.max(0, n - 1)), step: 1, "data-tip": "Number of the largest components removed as background" });
  const addMean = el("input", { type: "checkbox", checked: settings.get("analysis/svd_add_mean", false) });
  const summary = el("span", {});
  const canvas = el("canvas");
  const update = () => {
    const kk = Math.max(0, Math.min(n - 1, Math.round(Number(k.value) || 0)));
    summary.textContent = `Removes ${(fractions.slice(0, kk).reduce((a, b) => a + b, 0) * 100).toFixed(2)} % of the variance around the mean`;
    const shown = Math.min(n, Math.max(20, kk + 5));
    drawBars(canvas, fractions.slice(0, shown).map((f) => f * 100), Array.from({ length: shown }, (_, i) => (i + 1 <= kk ? "#c0504d" : "#4f81bd")),
      { xlabel: "Component", ylabel: "% of variance", title: "Red: removed as background" });
  };
  k.oninput = update;
  const body = el("div", { style: { width: "520px" } },
    el("p", {}, "The mean spectrum is subtracted and the SVD of the cube (pixels × bands) is computed. The largest components are treated as background and projected out of every spectrum."),
    el("div", { class: "svd-chart" }, canvas),
    el("div", { class: "form" }, el("label", {}, "Background components (k)"), k, el("span", {}), summary, el("span", {}),
      el("label", { class: "check", "data-tip": "Off: the result is the residual around zero. On: the mean spectrum is added back so values stay in the original range." }, addMean, "Add the mean spectrum back")));
  const res = await dialog({ title: "Remove background (SVD)", body, onOpen: () => requestAnimationFrame(update),
    buttons: [{ label: "OK", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }] });
  if (res !== "ok") return;
  const kk = Math.max(0, Math.min(n - 1, Math.round(Number(k.value) || 0)));
  settings.set("analysis/svd_k", kk);
  settings.set("analysis/svd_add_mean", addMean.checked);
  const name = await busy("Removing the background…", () => call("svd_apply", kk, addMean.checked));
  INFO = call("info");
  sync({ image: true, display: true });
  statusMessage(`Remove background (SVD): showing ${name}`, 8000);
}

const ANALYSIS_RUNNERS = { analysis_svd_background: svdDialog };

function segmentCommon(prefix, nBands) {
  const g = (k, d) => settings.get(`${prefix}/${k}`, d);
  const comps = el("input", { type: "number", min: 0, max: nBands, step: 1, value: Math.min(g("components", 10), nBands),
    "data-tip": "Cluster or classify on this many principal components of the cube (faster and less noisy); 'All bands' (0) uses the spectra directly" });
  const compsNote = el("span", { class: "info" });
  const upd = () => { compsNote.textContent = Number(comps.value) === 0 ? "All bands" : ""; };
  comps.oninput = upd; upd();
  const intensity = select(T.intensity_modes.map((m) => ({ value: m.key, label: m.label })), g("intensity", "use"));
  intensity.dataset.tip = "Use intensity: brightness counts, so the same material in light and shadow can split.\nIgnore intensity: every spectrum is scaled to unit length, so only its shape decides the class\n(spectral angle); a pixel gets the same class at any brightness.\nIgnore intensity and offset: also subtracts each spectrum's own mean first, so a constant\noffset such as stray light or a dark level is ignored too (correlation between spectra).";
  const smooth = select(T.smoothing.map((s) => ({ value: s.size, label: s.label })), String(g("smooth", 0)));
  smooth.dataset.tip = "Majority filter applied to the result to remove isolated pixels";
  const rows = [el("label", {}, "Principal components"), el("div", { class: "row" }, comps, compsNote), el("label", {}, "Intensity"), intensity, el("label", {}, "Smoothing"), smooth];
  const params = () => {
    const p = { n_components: Math.max(0, Math.min(nBands, Math.round(Number(comps.value) || 0))), intensity: intensity.value, smooth: Number(smooth.value) };
    settings.set(`${prefix}/components`, p.n_components); settings.set(`${prefix}/intensity`, p.intensity); settings.set(`${prefix}/smooth`, p.smooth);
    return p;
  };
  return { rows, params };
}

function intField(min, max, value, tip) { return el("input", { type: "number", min, max, step: 1, value, "data-tip": tip || null }); }
const clampInt = (input) => Math.max(Number(input.min), Math.min(Number(input.max), Math.round(Number(input.value) || 0)));

async function segmentUnsupervisedDialog() {
  const P = "segment_unsupervised";
  const common = segmentCommon(P, INFO.bands);
  const clusters = intField(2, 64, settings.get(`${P}/clusters`, 6));
  const iters = intField(1, 1000, settings.get(`${P}/iterations`, 50));
  const seed = intField(0, 1000000, settings.get(`${P}/seed`, 0), "Random seed for the starting cluster centres; change it to try another start");
  const body = el("div", { style: { width: "460px" } }, el("p", {}, "Group the pixels into regions of similar spectra with k-means clustering. No annotations are needed."),
    el("div", { class: "form" }, el("label", {}, "Number of regions"), clusters, ...common.rows, el("label", {}, "Maximum iterations"), iters, el("label", {}, "Random seed"), seed));
  if (await dialog({ title: "Segment (unsupervised)", body, buttons: [{ label: "OK", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }] }) !== "ok") return;
  const p = common.params();
  Object.assign(p, { n_clusters: clampInt(clusters), max_iter: clampInt(iters), seed: clampInt(seed) });
  settings.set(`${P}/clusters`, p.n_clusters); settings.set(`${P}/iterations`, p.max_iter); settings.set(`${P}/seed`, p.seed);
  await runSegmentation(() => call("segment_unsupervised", p));
}

async function segmentSupervisedDialog() {
  const P = "segment_supervised";
  const summary = call("training_summary");
  const common = segmentCommon(P, INFO.bands);
  const method = select(T.supervised_methods.map((m) => ({ value: m.key, label: m.label })), settings.get(`${P}/method`, "nearest_mean"));
  method.dataset.tip = "Nearest class mean: each pixel goes to the class with the closest mean spectrum.\nLDA: linear discriminant with a shared covariance; better when classes overlap.\nk-NN: majority vote of the closest training pixels; needs several pixels per class.";
  const neighbors = intField(1, 50, settings.get(`${P}/neighbors`, 5));
  const ok = summary.n_classes >= 2;
  const warn = el("p", { class: "warning" }, ok ? "" : "Annotate at least two classes (pick the class with keys 1–7) inside the image to train on.");
  warn.hidden = ok;
  const syncKnn = () => { neighbors.disabled = method.value !== "knn"; };
  method.onchange = syncKnn; syncKnn();
  const body = el("div", { style: { width: "480px" } },
    el("p", {}, "Classify every pixel into one of the annotated classes. Points give one training pixel, rectangles and polygons every pixel inside them."),
    el("p", { style: { whiteSpace: "pre" } }, "Training data\n  " + (summary.rows.length ? summary.rows.join("\n  ") : "none")),
    el("div", { class: "form" }, el("label", {}, "Method"), method, el("label", {}, "Neighbours (k-NN)"), neighbors, ...common.rows), warn);
  const res = await dialog({ title: "Segment from annotations", body, buttons: [{ label: "OK", value: "ok", default: true, disabled: !ok, id: "ok" }, { label: "Cancel", value: null, cancel: true }] });
  if (res !== "ok") return;
  const p = common.params();
  Object.assign(p, { method: method.value, k_neighbors: clampInt(neighbors) });
  settings.set(`${P}/method`, p.method); settings.set(`${P}/neighbors`, p.k_neighbors);
  await runSegmentation(() => call("segment_supervised", p));
}

async function runSegmentation(fn) {
  const res = await guarded("Segmentation", () => busy("Segmenting…", fn));
  if (!res) return;
  sync();
  statusMessage(`Segmentation: ${res.title}. Save it with Segment → Save segmentation…`, 8000);
}

// ------------------------------------------------------------- commands
const CUBE_EXTS = [".hdr", ".img", ".dat", ".raw", ".bil", ".bip", ".bsq", ".npy", ".npz", ".h5", ".hdf5", ".he5", ".nc", ".mat",
  ".tif", ".tiff", ".pkl", ".pickle", ".p", ".jpg", ".jpeg", ".png", ".bmp", ".gif"];
const OVERLAY_EXTS = ".npy,.npz,.tif,.tiff,.png,.jpg,.bmp,.h5,.hdf5,.pkl";
const IMAGE_EXTS = ".jpg,.jpeg,.png,.bmp,.tif,.tiff";

async function confirmDiscard() {
  if (!S.modified) return true;
  const v = await dialog({
    title: "Unsaved annotations", body: el("div", { class: "msg" }, el("div", { class: "msg-icon" }, "❓"), el("p", {}, "Save the current annotations first?")),
    buttons: [{ label: "Save", value: "save", default: true }, { label: "Discard", value: "discard" }, { label: "Cancel", value: null, cancel: true }],
  });
  if (v === "save") return saveAnnotations();
  return v === "discard";
}

async function openCubeFiles(files) {
  const written = await writeFiles(files);
  // ENVI: open the header; the data file next to it is found by name.
  const main = written.find((f) => f.name.toLowerCase().endsWith(".hdr")) || written[0];
  const info = await guarded("Could not open file", async () => {
    await ensurePackages(packagesFor(main.name));
    return busy(`Opening ${main.name}…`, () => call("open_cube", main.path, null));
  });
  if (!info) return false;
  cubeLoaded(info);
  return true;
}

function cubeLoaded(info) {
  INFO = info;
  annotationTarget = null;
  view.clearOverlays();
  view.selected.clear();
  lastVersion = -1;
  sync({ image: true, display: true });
  view.zoomFit();
  for (const m of info.messages) statusMessage(m, 8000);
}

async function openCubeDialog() {
  if (!(await confirmDiscard())) return;
  const files = await pickFiles(CUBE_EXTS.join(","), true);
  if (files) await openCubeFiles(files);
}

async function openDemo() {
  if (!(await confirmDiscard())) return;
  const info = await busy("Making the demo cube…", () => call("load_demo"));
  cubeLoaded(info);
  statusMessage("Demo cube: 160 × 120 pixels, 100 bands, with a reference photo and a hidden label overlay", 8000);
}

async function openByIdDialog() {
  if (!window.showDirectoryPicker) {
    await warning("Open by database id", "Opening by database id needs a browser that can open folders (Chrome or Edge).");
    return;
  }
  const id = await getInt("Open by database id", "Image id:", 1, 0, 1e9);
  if (id === null) return;
  if (!(await confirmDiscard())) return;
  let dir;
  try { dir = await window.showDirectoryPicker({ id: "hsview-images" }); } catch { return; }
  await guarded("Open by id", async () => {
    let dbFile;
    try { dbFile = await (await dir.getFileHandle("image_database.db")).getFile(); } catch { throw new Error(`No image database (image_database.db) in ${dir.name}`); }
    const [db] = await writeFiles([dbFile]);
    const rel = call("database_file", id, db.path.slice(0, db.path.lastIndexOf("/")));
    let handle = dir;
    const parts = rel.split("/");
    for (const p of parts.slice(0, -1)) handle = await handle.getDirectoryHandle(p);
    const file = await (await handle.getFileHandle(parts[parts.length - 1])).getFile();
    await openCubeFiles([file]);
  });
}

async function loadFilterDialog(c) {
  if (!S.has_cube) return;
  const files = await pickFiles(".pkl");
  if (!files) return;
  const [f] = await writeFiles(files);
  await guarded("Bad filter file", () => { D = call("load_filter", c, f.path, f.name); });
  refreshDisplay(); sync();
}

async function openReferenceDialog() {
  const files = await pickFiles(IMAGE_EXTS);
  if (!files) return;
  const [f] = await writeFiles(files);
  await guarded("Could not open image", async () => { await ensurePackages(packagesFor(f.name)); D = call("load_reference", f.path, f.name); });
  sync({ display: true });
}

async function addOverlayDialog(accept = OVERLAY_EXTS) {
  const files = await pickFiles(accept);
  if (!files) return;
  const [f] = await writeFiles(files);
  await guarded("Could not open overlay", async () => {
    await ensurePackages(packagesFor(f.name));
    return busy("Adding overlay…", () => call("add_overlay_file", f.path, f.name));
  });
  sync();
}

async function loadAnnotationsDialog() {
  const files = await pickFiles(".pkl,.json,.npz");
  if (!files) return;
  const [f] = await writeFiles(files);
  const n = await guarded("Could not load annotations", () => call("load_annotations", f.path, f.name));
  if (n === undefined) return;
  annotationTarget = null;
  sync();
  statusMessage(`Loaded ${n} annotations from ${f.name}`);
}

const ANNOTATION_FILTERS = [{ label: "Pickle", ext: ".pkl" }, { label: "JSON", ext: ".json" }];
async function saveAnnotations() {
  if (annotationTarget) return writeAnnotations(annotationTarget);
  return saveAnnotationsAs();
}
async function saveAnnotationsAs() {
  const target = await saveAs("Save annotations", S.annotation_name || `${INFO.name}_annotations.pkl`, ANNOTATION_FILTERS);
  return !!target && writeAnnotations(target);
}
async function writeAnnotations(target) {
  const n = await guarded("Could not save annotations", () => saveWith(target, (path) => call("save_annotations", path, target.name)));
  if (n === undefined) return false;
  annotationTarget = target.handle ? target : null;
  sync();
  statusMessage(`Saved ${n} annotations to ${target.name}`);
  return true;
}

async function saveCubeDialog() {
  const target = await saveAs(INFO.processed ? "Save processed cube" : "Save cube", `${INFO.safe_name}.pkl`,
    [{ label: "Pickle", ext: ".pkl" }, { label: "NumPy archive", ext: ".npz" }, { label: "HDF5", ext: ".h5" }]);
  if (!target) return;
  if (await guarded("Could not save cube", () => saveWith(target, (path) => call("save_cube", path), "Saving the cube…")) !== undefined)
    statusMessage(`Saved cube to ${target.name}`);
}

async function saveSegmentationDialog() {
  const name = call("segmentation_name");
  if (name === null) return;
  const target = await saveAs(`Save segmentation '${name}'`, `${INFO.safe_name}_segmentation.pkl`,
    [{ label: "Pickle", ext: ".pkl" }, { label: "NumPy archive", ext: ".npz" }]);
  if (!target) return;
  if (await guarded("Could not save segmentation", () => saveWith(target, (path) => call("save_segmentation", path))) !== undefined)
    statusMessage(`Saved segmentation to ${target.name}`);
}

async function saveSpectraDialog() {
  const target = await saveAs("Save annotation spectra", `${INFO.name}_spectra.pkl`, [{ label: "Pickle", ext: ".pkl" }]);
  if (!target) return;
  if (await guarded("Could not save spectra", () => saveWith(target, (path) => call("save_spectra", path))) !== undefined)
    statusMessage(`Saved spectra to ${target.name}`);
}

async function exportCsvDialog() {
  const target = await saveAs("Export spectra", `${INFO.name}_spectra.csv`, [{ label: "CSV", ext: ".csv" }]);
  if (!target) return;
  const n = await guarded("Could not export spectra", () => saveWith(target, (path) => call("export_csv", path)));
  if (n !== undefined) statusMessage(`Wrote ${n} spectra to ${target.name}`);
}

async function channelBandDialog(c) {
  const ch = D.channels[c], n = INFO.bands;
  const b = await getInt(`Band channel ${c + 1}`, `Band (0–${n - 1}):`, ch.band, 0, n - 1);
  if (b === null) return;
  displayChange(() => {
    call("set_channel", c, { source: "Band", band: b });
    return D.channels[c].route === "None" ? call("set_channel", c, { route: "Gray" }) : call("display_state");
  });
}
async function channelWidthDialog(c) {
  const w = await getInt(`Band channel ${c + 1}`, "Average this many bands:", D.channels[c].width, 1, INFO.bands);
  if (w !== null) displayChange(() => call("set_channel", c, { width: w }));
}
async function channelSumDialog(c) {
  const ch = D.channels[c];
  const r = await bandSpecDialog(`Band channel ${c + 1}`, ["Bands"], [ch.spec], ch.sum_mode);
  if (!r || !r.values[0]) return;
  displayChange(() => {
    call("set_channel", c, { source: "Band sum", spec: r.values[0], sum_mode: r.mode });
    return D.channels[c].route === "None" ? call("set_channel", c, { route: "Gray" }) : call("display_state");
  });
}
async function showRgbDialog() {
  let defaults = D.channels.map((c) => c.spec);
  if (!defaults.some(Boolean) && INFO.has_wavelengths) defaults = T.true_colour;
  const r = await bandSpecDialog("RGB from bands", ["Red", "Green", "Blue"], defaults, D.channels[0].sum_mode);
  if (r) displayChange(() => call("show_rgb", r.values, r.mode));
}
async function referenceOpacityDialog() {
  const v = await getInt("Reference image", "Opacity (%):", D.reference.opacity, 0, 100);
  if (v === null) return;
  displayChange(() => {
    call("set_reference", { opacity: v });
    return D.reference.route === "None" ? call("set_reference", { route: "Overlay" }) : call("display_state");
  });
}
async function overlayOpacityDialog(key) {
  key = key || S.current_overlay;
  const ov = call("overlay_list").find((o) => o.key === key);
  if (!ov) return;
  const v = await getInt(ov.name, "Opacity (%):", Math.round(ov.opacity * 100), 0, 100);
  if (v === null) return;
  call("set_overlay_opacity", key, v / 100);
  const vo = view.overlays.get(key);
  if (vo) { vo.opacity = v / 100; view.draw(); }
  refreshOverlays();
}

function setClass(cls) {
  view.currentClass = cls;
  const ids = view.selectedIds();
  if (ids.length) call("recolor_annotations", ids, cls);
  sync();
  view.draw();
  updateToolbarStates();
}

async function quit() {
  if (!(await confirmDiscard())) return;
  for (const w of [...plotWindows]) w.close();
  py.destroy();
  py = pyodide.pyimport("hsview.webapp").WebApp();
  INFO = { loaded: false };
  annotationTarget = null;
  view.clearOverlays();
  lastVersion = -1;
  sync({ image: true, display: true });
  statusMessage("Closed the cube. Open another with File → Open cube…");
}

function about() {
  information("About hsview", `hsview ${T.version}\nView and annotate hyperspectral cubes.\n\nEvery command is in the menus. Use the Toolbars menu (or right-click a toolbar) to change the toolbars.\n\nThis is the browser version: the hsview Python package running in Pyodide ${pyodide.version}. Files you open stay in this browser tab.`);
}

function defineActions() {
  const has = () => !!S.has_cube;
  const run = {
    open_cube: openCubeDialog, open_by_id: openByIdDialog, open_reference: openReferenceDialog,
    add_overlay: () => addOverlayDialog(), load_filter: () => loadFilterDialog(0), load_annotations: loadAnnotationsDialog,
    save_annotations: saveAnnotations, save_annotations_as: saveAnnotationsAs, save_spectra: saveSpectraDialog,
    export_csv: exportCsvDialog, save_cube: saveCubeDialog, quit,
    undo: () => { call("undo"); sync(); }, redo: () => { call("redo"); sync(); },
    delete_selected: () => { call("remove_annotations", view.selectedIds()); view.selectIds([]); sync(); },
    select_all: () => { view.setMode("select"); view.selectIds(view.annotations.map((a) => a.id)); },
    clear_annotations: async () => { if (S.n_annotations && await question("Clear annotations", "Remove all annotations?")) { call("clear_annotations"); sync(); } },
    finish_polygon: () => view.finishPolygon(), cancel_drawing: () => view.cancelDrawing(),
    customize: customizeDialog,
    zoom_in: () => view.zoomIn(), zoom_out: () => view.zoomOut(), zoom_fit: () => view.zoomFit(), zoom_actual: () => view.zoomActual(),
    prev_band: () => displayChange(() => call("step_band", -1)), next_band: () => displayChange(() => call("step_band", 1)),
    true_colour: () => displayChange(() => call("true_colour")), show_rgb: showRgbDialog,
    reverse_rgb: () => displayChange(() => call("reverse_rgb")), reset_contrast: () => displayChange(() => call("reset_contrast")),
    contrast_up: () => displayChange(() => call("step_contrast", 5)), contrast_down: () => displayChange(() => call("step_contrast", -5)),
    bias_up: () => displayChange(() => call("step_bias", 8)), bias_down: () => displayChange(() => call("step_bias", -8)),
    ref_opacity: referenceOpacityDialog,
    remove_overlay: () => { if (S.current_overlay) removeOverlay(S.current_overlay); },
    clear_overlays: async () => { if (S.has_overlays && await question("Remove overlays", "Remove all overlays?")) { call("clear_overlays"); sync(); } },
    show_overlays: () => { call("set_all_overlays_visible", true); sync(); },
    hide_overlays: () => { call("set_all_overlays_visible", false); sync(); },
    overlay_opacity: () => overlayOpacityDialog(),
    analysis_undo: () => { call("undo_analysis"); INFO = call("info"); sync({ image: true, display: true }); },
    analysis_restore: () => { call("restore_original_cube"); INFO = call("info"); sync({ image: true, display: true }); },
    segment_unsupervised: segmentUnsupervisedDialog, segment_supervised: segmentSupervisedDialog,
    save_segmentation: saveSegmentationDialog,
    open_segmentation: () => addOverlayDialog(".pkl,.npz," + OVERLAY_EXTS),
    plot_points: () => plotByClass("point"), plot_regions: () => plotByClass("region"),
    close_plots: () => { for (const w of [...plotWindows]) w.close(); },
    about,
  };
  const enabled = {
    open_reference: has, add_overlay: has, load_filter: has, load_annotations: has, save_annotations: has,
    save_annotations_as: has, save_spectra: has, export_csv: has, plot_points: has, plot_regions: has,
    zoom_in: has, zoom_out: has, zoom_fit: has, zoom_actual: has, prev_band: has, next_band: has, reset_contrast: has,
    clear_annotations: has, select_all: has, save_cube: has, show_rgb: has, contrast_up: has, contrast_down: has,
    bias_up: has, bias_down: has,
    segment_unsupervised: has, segment_supervised: () => has() && S.n_annotations > 0, open_segmentation: has,
    save_segmentation: () => !!S.has_segmentation,
    analysis_undo: () => S.history > 0, analysis_restore: () => S.history > 0,
    true_colour: () => has() && S.has_wavelengths,
    reverse_rgb: () => !!S.reference_rgb, ref_route_show: () => !!S.has_reference, ref_route_overlay: () => !!S.has_reference,
    ref_opacity: () => !!S.has_reference,
    remove_overlay: () => !!S.current_overlay, overlay_opacity: () => !!S.current_overlay,
    clear_overlays: () => !!S.has_overlays, show_overlays: () => !!S.has_overlays, hide_overlays: () => !!S.has_overlays,
    finish_polygon: () => view.mode === "polygon", cancel_drawing: () => ["polygon", "rect"].includes(view.mode),
    undo: () => !!S.can_undo, redo: () => !!S.can_redo, delete_selected: () => view.selected.size > 0,
  };
  const checked = {
    plot_separate: () => settings.get("plot/separate_windows", false),
    spectra_cursor: () => panels.spectra.cursor.checked, spectra_std: () => panels.spectra.std.checked,
    spectra_by_class: () => panels.spectra.byClass.checked, spectra_labels: () => panels.spectra.labels.checked,
  };
  run.plot_separate = () => settings.set("plot/separate_windows", !settings.get("plot/separate_windows", false));
  const box = { spectra_cursor: "cursor", spectra_std: "std", spectra_by_class: "byClass", spectra_labels: "labels" };
  for (const [id, key] of Object.entries(box)) run[id] = () => { panels.spectra[key].checked = !panels.spectra[key].checked; scheduleSpectra(true); };
  for (const [id, dock] of Object.entries(DOCKS)) { run[id] = () => setDockVisible(id, !dockVisible(id)); checked[id] = () => dockVisible(id); void dock; }
  for (const m of ["navigate", "select", "point", "rect", "polygon"]) { run[`mode_${m}`] = () => view.setMode(m); checked[`mode_${m}`] = () => view.mode === m; }
  T.class_names.forEach((_, i) => { run[`class_${i}`] = () => setClass(i); checked[`class_${i}`] = () => view.currentClass === i; });
  for (const s of T.scalings) { run[`scaling_${s.key}`] = () => displayChange(() => call("set_view", { scaling: s.key })); checked[`scaling_${s.key}`] = () => D && D.scaling === s.key; enabled[`scaling_${s.key}`] = has; }
  for (const r of T.ref_routes) { run[`ref_route_${r.toLowerCase()}`] = () => displayChange(() => call("set_reference", { route: r })); checked[`ref_route_${r.toLowerCase()}`] = () => D && D.reference.route === r; }
  for (let c = 0; c < 3; c++) {
    run[`ch${c}_band`] = () => channelBandDialog(c);
    run[`ch${c}_width`] = () => channelWidthDialog(c);
    run[`ch${c}_sum`] = () => channelSumDialog(c);
    run[`ch${c}_filter`] = () => loadFilterDialog(c);
    for (const r of T.routes) {
      const id = `ch${c}_route_${r.toLowerCase()}`;
      run[id] = () => displayChange(() => call("set_channel", c, { route: r }));
      checked[id] = () => D && D.channels[c].route === r;
    }
  }
  for (const [id, fn] of Object.entries(ANALYSIS_RUNNERS)) run[id] = fn;
  for (const spec of T.specs) {
    if (spec.category.startsWith("Channel ") || spec.category === "Analysis") enabled[spec.id] ||= has;
    actions[spec.id] = { spec, run: run[spec.id] || (() => warning(spec.text, "This command is not available in the browser version.")),
      enabled: enabled[spec.id], checked: checked[spec.id] };
  }
}

// ------------------------------------------------------------- start
async function main() {
  const icon = document.getElementById("loading-icon");
  try {
    await boot();
  } catch (err) {
    console.error(err);
    document.querySelector(".loading-box").classList.add("error");
    document.getElementById("loading-text").textContent = "hsview could not start.\n\n" + String(err && err.message ? err.message : err);
    return;
  }
  icon.src = svgUrl(T.app_icons[currentAppIcon()].svg);
  setAppIcon(currentAppIcon());
  view = new ImageView(document.getElementById("view"), T.class_colors);
  view.on = {
    cursor: (x, y) => {
      const p = Number.isNaN(x) ? null : [Math.floor(x), Math.floor(y)];
      document.getElementById("status-permanent").textContent = p ? `x ${p[0]}  y ${p[1]}   zoom ${view.zoom.toFixed(2)}×` : "";
      if ((p && cursor && p[0] === cursor[0] && p[1] === cursor[1]) || (!p && !cursor)) return;
      cursor = p;
      if (panels.spectra.cursor.checked) scheduleSpectra();
    },
    add: (kind, coords) => { call("add_annotation", kind, view.currentClass, coords); sync(); },
    remove: (ids) => { call("remove_annotations", ids); sync(); },
    move: (ids, dx, dy) => { call("move_annotations", ids, dx, dy); sync(); },
    selection: () => updateToolbarStates(),
    mode: () => updateToolbarStates(),
    zoom: (z) => { if (cursor) document.getElementById("status-permanent").textContent = `x ${cursor[0]}  y ${cursor[1]}   zoom ${z.toFixed(2)}×`; },
  };
  loadShortcuts();
  defineActions();
  buildDisplayPanel();
  buildOverlayPanel();
  buildSpectraPanel();
  buildMenus();
  let saved = null;
  try { saved = JSON.parse(settings.get("toolbars/config", "null")); } catch { saved = null; }
  tbConfig = validateConfig(saved || T.toolbars);
  renderToolbars();
  setupSplitters();
  layoutDocks();
  sync({ image: true, display: true });
  document.getElementById("loading").classList.add("hidden");
  document.getElementById("demo-button").onclick = openDemo;

  // Open a file dropped on the canvas.
  const center = document.getElementById("center");
  center.addEventListener("dragover", (e) => { e.preventDefault(); center.classList.add("drop"); });
  center.addEventListener("dragleave", () => center.classList.remove("drop"));
  center.addEventListener("drop", async (e) => {
    e.preventDefault(); center.classList.remove("drop");
    if (e.dataTransfer.files.length && await confirmDiscard()) await openCubeFiles([...e.dataTransfer.files]);
  });
  window.addEventListener("beforeunload", (e) => { if (S.modified) { e.preventDefault(); e.returnValue = ""; } });

  // ?demo opens the demo cube straight away.
  if (new URLSearchParams(location.search).has("demo")) await openDemo();
}

main();

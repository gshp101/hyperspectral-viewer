// Generic desktop-style widgets: menus, dialogs, message boxes, floating windows and tooltips.

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else if (v === true) e.setAttribute(k, "");
    else e.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

export const layer = () => document.getElementById("layer");

/** "&File" -> "File" with the mnemonic letter underlined. */
export function mnemonic(text) {
  const span = document.createElement("span");
  const i = text.indexOf("&");
  if (i < 0) { span.textContent = text; return span; }
  span.append(text.slice(0, i), el("u", {}, text[i + 1]), text.slice(i + 2));
  return span;
}

/** Display a Qt portable shortcut ("Ctrl+Shift+Z") the platform's way. */
export function shortcutText(seq) {
  if (!seq) return "";
  const mac = /Mac|iPhone|iPad/.test(navigator.platform);
  return mac ? seq.replace(/Ctrl\+/g, "⌘").replace(/Shift\+/g, "⇧").replace(/Alt\+/g, "⌥").replace(/Meta\+/g, "⌃") : seq;
}

// ------------------------------------------------------------------ menus
let openMenus = [];
let menubarActive = null;

export function closeMenus(depth = 0) {
  while (openMenus.length > depth) openMenus.pop().remove();
  if (depth === 0 && menubarActive) { menubarActive.classList.remove("open"); menubarActive = null; }
}

document.addEventListener("mousedown", (e) => {
  if (!openMenus.length) return;
  if (openMenus.some((m) => m.contains(e.target)) || (menubarActive && menubarActive.contains(e.target))) return;
  closeMenus();
}, true);
window.addEventListener("blur", () => closeMenus());
window.addEventListener("resize", () => closeMenus());

/**
 * Show a popup menu. Items: "sep", or {label, icon (Node), shortcut, checked (true/false/undefined),
 * enabled, tip, action(), submenu: () => items}.
 */
export function openMenu(items, x, y, depth = 0) {
  closeMenus(depth);
  const menu = el("div", { class: "menu", role: "menu" });
  let timer = null;
  for (const it of items) {
    if (it === "sep") {
      if (menu.lastChild && !menu.lastChild.classList.contains("menu-sep")) menu.append(el("div", { class: "menu-sep" }));
      continue;
    }
    const enabled = it.enabled !== false;
    const row = el("div", { class: "menu-item" + (enabled ? "" : " disabled"), role: "menuitem", "data-tip": it.tip || null },
      el("span", { class: "check" }, it.checked === true ? (it.radio ? "●" : "✓") : ""),
      el("span", { class: "icon" }, it.icon || ""),
      typeof it.label === "string" ? el("span", {}, it.label) : it.label,
      el("span", { class: "shortcut" }, shortcutText(it.shortcut || "")),
      el("span", { class: "arrow" }, it.submenu ? "▶" : ""));
    if (it.id) row.dataset.id = it.id;
    row.addEventListener("mouseenter", () => {
      clearTimeout(timer);
      menu.querySelectorAll(".menu-item.active").forEach((r) => r.classList.remove("active"));
      if (it.submenu && enabled) {
        row.classList.add("active");
        timer = setTimeout(() => {
          const r = row.getBoundingClientRect();
          openMenu(it.submenu(), r.right - 2, r.top - 4, depth + 1);
        }, 120);
      } else timer = setTimeout(() => closeMenus(depth + 1), 200);
    });
    row.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!enabled) return;
      if (it.submenu) { const r = row.getBoundingClientRect(); openMenu(it.submenu(), r.right - 2, r.top - 4, depth + 1); return; }
      closeMenus();
      if (it.action) it.action();
    });
    menu.append(row);
  }
  if (menu.lastChild && menu.lastChild.classList.contains("menu-sep")) menu.lastChild.remove();
  document.body.append(menu);
  const r = menu.getBoundingClientRect();
  menu.style.left = Math.max(0, Math.min(x, window.innerWidth - r.width - 2)) + "px";
  menu.style.top = Math.max(0, Math.min(y, window.innerHeight - r.height - 2)) + "px";
  if (depth > 0 && x + r.width > window.innerWidth) {
    const parent = openMenus[depth - 1].getBoundingClientRect();
    menu.style.left = Math.max(0, parent.left - r.width + 2) + "px";
  }
  openMenus.push(menu);
  return menu;
}

/** A menu bar; each menu is {title: "&File", build: () => items}. */
export function buildMenubar(container, menus) {
  container.textContent = "";
  for (const m of menus) {
    const title = el("div", { class: "menu-title", role: "menuitem" }, mnemonic(m.title));
    const show = () => {
      closeMenus();
      menubarActive = title;
      title.classList.add("open");
      const r = title.getBoundingClientRect();
      openMenu(m.build(), r.left, r.bottom);
      openMenus[0].addEventListener("remove", () => {});
    };
    title.addEventListener("mousedown", (e) => {
      e.preventDefault();
      if (menubarActive === title) closeMenus(); else show();
    });
    title.addEventListener("mouseenter", () => { if (menubarActive && menubarActive !== title) show(); });
    title.dataset.title = m.title.replace("&", "");
    container.append(title);
  }
}

// ---------------------------------------------------------------- dialogs
function dragBy(handle, box) {
  handle.addEventListener("mousedown", (e) => {
    if (e.target.closest("button")) return;
    const r = box.getBoundingClientRect();
    const ox = e.clientX - r.left, oy = e.clientY - r.top;
    box.style.position = "fixed"; box.style.margin = "0";
    box.style.left = r.left + "px"; box.style.top = r.top + "px";
    const move = (ev) => {
      box.style.left = Math.max(0, Math.min(window.innerWidth - 60, ev.clientX - ox)) + "px";
      box.style.top = Math.max(0, Math.min(window.innerHeight - 30, ev.clientY - oy)) + "px";
    };
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    e.preventDefault();
  });
}

let modalDepth = 0;
export const isModalOpen = () => modalDepth > 0;

/**
 * A modal dialog. buttons: [{label, value, default, cancel, disabled, id}]. Resolves with the value of the
 * button pressed (or of the default button for Enter; the cancel button or null for Esc). ``validate``
 * may return false to keep the dialog open.
 */
export function dialog({ title, body, buttons = [{ label: "OK", value: true, default: true }, { label: "Cancel", value: null, cancel: true }], onOpen, validate, width }) {
  return new Promise((resolve) => {
    closeMenus();
    const footer = el("footer");
    const box = el("div", { class: "dialog", role: "dialog", "aria-label": title, style: width ? { width } : null },
      el("header", {}, el("span", {}, title)), el("div", { class: "body" }, body), footer);
    const backdrop = el("div", { class: "modal-backdrop" }, box);
    const prevFocus = document.activeElement;
    modalDepth++;
    const finish = (value) => {
      if (value !== null && value !== undefined && validate && validate(value) === false) return;
      modalDepth--;
      backdrop.remove();
      document.removeEventListener("keydown", onKey, true);
      if (prevFocus && prevFocus.focus) prevFocus.focus();
      resolve(value);
    };
    const btns = {};
    for (const b of buttons) {
      const btn = el("button", { class: "push" + (b.default ? " default" : ""), type: "button", disabled: b.disabled }, b.label);
      btn.addEventListener("click", () => finish(b.value));
      if (b.id) btns[b.id] = btn;
      footer.append(btn);
    }
    const onKey = (e) => {
      if (backdrop !== layer().lastElementChild) return;
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); const c = buttons.find((b) => b.cancel); finish(c ? c.value : null); }
      else if (e.key === "Enter" && !(e.target instanceof HTMLTextAreaElement) && !(e.target instanceof HTMLButtonElement)) {
        const d = buttons.find((b) => b.default);
        if (d && !(d.id && btns[d.id] && btns[d.id].disabled)) { e.preventDefault(); e.stopPropagation(); finish(d.value); }
      }
      e.stopPropagation();
    };
    document.addEventListener("keydown", onKey, true);
    dragBy(box.querySelector("header"), box);
    layer().append(backdrop);
    if (onOpen) onOpen({ box, buttons: btns, close: finish });
    const first = box.querySelector("input, select, textarea") || footer.querySelector(".default");
    if (first) setTimeout(() => { first.focus(); if (first.select) first.select(); }, 0);
  });
}

function messageBody(icon, text) {
  const t = el("div", {});
  for (const para of String(text).split("\n\n")) t.append(el("p", {}, para));
  return el("div", { class: "msg" }, el("div", { class: "msg-icon" }, icon), t);
}

export const warning = (title, text) => dialog({ title, body: messageBody("⚠️", text), buttons: [{ label: "OK", value: true, default: true, cancel: true }] });
export const information = (title, text) => dialog({ title, body: messageBody("ℹ️", text), buttons: [{ label: "OK", value: true, default: true, cancel: true }] });
export const question = async (title, text) => (await dialog({
  title, body: messageBody("❓", text),
  buttons: [{ label: "Yes", value: true, default: true }, { label: "No", value: false, cancel: true }],
})) === true;

export function getInt(title, label, value, min, max) {
  const input = el("input", { type: "number", value, min, max, step: 1, style: { width: "120px" } });
  return dialog({
    title, body: el("div", { class: "form" }, el("label", {}, label), input),
    buttons: [{ label: "OK", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }],
  }).then((v) => {
    if (v !== "ok") return null;
    const n = Math.round(Number(input.value));
    return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : null;
  });
}

export function getText(title, label, value = "") {
  const input = el("input", { type: "text", value, style: { width: "260px" } });
  return dialog({
    title, body: el("div", { class: "form" }, el("label", {}, label), input),
    buttons: [{ label: "OK", value: "ok", default: true }, { label: "Cancel", value: null, cancel: true }],
  }).then((v) => (v === "ok" ? input.value : null));
}

// -------------------------------------------------------- floating windows
let windowCount = 0;
/** A non-modal, movable, resizable window (the plot windows). Returns {box, body, close}. */
export function floatingWindow({ title, x, y, w = 720, h = 440, actions = [], onResize, onClose }) {
  const body = el("div", { class: "body" });
  const close = el("button", { class: "dock-close", title: "Close" }, "×");
  const header = el("header", {}, el("span", {}, title), el("span", { class: "actions" }, ...actions, close));
  const box = el("div", { class: "window plot-window" }, header, body);
  const W = Math.min(w, window.innerWidth - 20), H = Math.min(h, window.innerHeight - 20);
  Object.assign(box.style, {
    left: Math.max(0, Math.min(x ?? (80 + 30 * windowCount), window.innerWidth - W)) + "px",
    top: Math.max(0, Math.min(y ?? (80 + 30 * windowCount), window.innerHeight - H)) + "px",
    width: W + "px", height: H + "px",
  });
  windowCount = (windowCount + 1) % 10;
  const doClose = () => { box.remove(); if (onClose) onClose(); };
  close.addEventListener("click", doClose);
  box.addEventListener("mousedown", () => layer().append(box));
  dragBy(header, box);
  if (onResize) new ResizeObserver(() => onResize()).observe(body);
  layer().append(box);
  return { box, body, close: doClose };
}

// --------------------------------------------------------------- tooltips
let tipEl = null, tipTimer = null;
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  clearTimeout(tipTimer);
  if (tipEl) { tipEl.remove(); tipEl = null; }
  if (!t || !t.dataset.tip) return;
  tipTimer = setTimeout(() => {
    const r = t.getBoundingClientRect();
    tipEl = el("div", { class: "tooltip" }, t.dataset.tip);
    document.body.append(tipEl);
    const tr = tipEl.getBoundingClientRect();
    const inMenu = t.classList.contains("menu-item");
    let x = inMenu ? r.right + 4 : r.left, y = inMenu ? r.top : r.bottom + 4;
    if (x + tr.width > window.innerWidth) x = Math.max(0, (inMenu ? r.left - tr.width - 4 : window.innerWidth - tr.width - 4));
    if (y + tr.height > window.innerHeight) y = r.top - tr.height - 4;
    tipEl.style.left = x + "px"; tipEl.style.top = y + "px";
  }, 600);
});
document.addEventListener("mousedown", () => { clearTimeout(tipTimer); if (tipEl) { tipEl.remove(); tipEl = null; } }, true);

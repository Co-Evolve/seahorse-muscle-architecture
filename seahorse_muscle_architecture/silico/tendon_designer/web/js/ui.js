// Small DOM helpers shared by the UI-shell modules (app, panels, plots, experiments).
// No framework: h() builds elements, tooltip() gives plain-language hover help.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "tip") el.dataset.tip = v;
    else if (v === true) el.setAttribute(k, "");
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v);
  }
  appendChildren(el, children);
  return el;
}

function appendChildren(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

export function debounce(fn, ms) {
  let t = null;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => { t = null; fn(...args); }, ms); };
  d.flush = (...args) => { clearTimeout(t); t = null; fn(...args); };
  d.cancel = () => { clearTimeout(t); t = null; };
  return d;
}

/** Parse a number typed by a person: accepts "0,25" as well as "0.25" and "1e-3". */
export function parseNum(v) {
  const t = String(v ?? "").trim().replace(/\s/g, "").replace(",", ".");
  if (!t) return NaN;
  const n = Number(t);
  return Number.isFinite(n) ? n : NaN;
}

export function deepCopy(x) { return x === undefined ? undefined : JSON.parse(JSON.stringify(x)); }

export function fmt(v, digits = 1) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const r = Number(v.toFixed(digits));
  return (Object.is(r, -0) ? 0 : r).toFixed(digits);
}

/** Format a number for a parameter input: compact but exact enough. */
export function fmtParam(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2).replace(/\.?0+e/, "e");
  return String(Number(v.toPrecision(4)));
}

/** Icons: small inline SVG line icons (24-unit grid, stroke = currentColor). */
const ICONS = {
  play: "M8 5.5v13l10.5-6.5z",
  pause: "M8 5h3v14H8zM13.5 5h3v14h-3z",
  reset: "M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v4h4",
  plus: "M12 5v14M5 12h14",
  copy: "M8 8h11v11H8zM5 16V5h11",
  trash: "M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12",
  undo: "M9 14 4 9l5-5M4 9h10a5.5 5.5 0 0 1 0 11h-3",
  redo: "M15 14l5-5-5-5M20 9H10a5.5 5.5 0 0 0 0 11h3",
  file: "M6 3h8l4 4v14H6zM14 3v4h4",
  open: "M3.5 7.5h6l2 2h9v9.5h-17zM3.5 7.5V5h6l2 2.5",
  save: "M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19h14",
  help: "M9.3 9.2a2.8 2.8 0 1 1 3.8 2.6c-.7.3-1.1.9-1.1 1.7v.7M12 17.4v.2",
  code: "M9 7l-5 5 5 5M15 7l5 5-5 5",
  chevron: "M8 10l4 4 4-4",
  close: "M6 6l12 12M18 6 6 18",
  info: "M12 11v6M12 7.6v.2",
  flask: "M9.5 3.5h5M10.5 3.5v6L5 19.5h14l-5.5-10v-6M7.6 15h8.8",
  edit: "M5 19h4L19 9l-4-4L5 15zM13.5 6.5l4 4",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M5.2 18.8l1.4-1.4M17.4 6.6l1.4-1.4",
  moon: "M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10z",
  library: "M5 4h3v16H5zM10 4h3v16h-3zM15.2 4.8l2.9-.8 3.4 15.4-2.9.8z",
  download: "M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19h14",
  split: "M6 4v6a4 4 0 0 0 4 4h0M6 10v10M10 14h0a4 4 0 0 1 4 4v2M14 4l4 4-4 4M18 8h-6",
  stop: "M7 7h10v10H7z",
  eye: "M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z",
  dots: "M6 12h.01M12 12h.01M18 12h.01",
  gravity: "M12 4v13M7.5 12.5 12 17l4.5-4.5M6 20.5h12",
  contrast: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18zM12 3v18",
  mirror: "M12 3v18M8.5 7.5 4 12l4.5 4.5M15.5 7.5 20 12l-4.5 4.5",
  upright: "M12 20V5M7.5 9.5 12 5l4.5 4.5",
  hanging: "M12 4v15M7.5 14.5 12 19l4.5-4.5",
  layers: "M12 4 3.5 8.5 12 13l8.5-4.5zM3.5 12.5 12 17l8.5-4.5M3.5 16.5 12 21l8.5-4.5",
  zap: "M13 3 5 13.5h6L10 21l8-10.5h-6z",
};
export function icon(name, size = 16) {
  const p = ICONS[name] || ICONS.dots;
  return s("svg", { class: "icon", width: size, height: size, viewBox: "0 0 24 24", "aria-hidden": "true",
    fill: name === "play" || name === "pause" || name === "stop" ? "currentColor" : "none",
    stroke: "currentColor", "stroke-width": name === "dots" ? 3 : 1.7, "stroke-linecap": "round", "stroke-linejoin": "round" },
    s("path", { d: p }));
}

// ---------------------------------------------------------------- tooltip
let tipEl = null;
let tipTarget = null;
export function installTooltips(root = document.body) {
  tipEl = h("div", { class: "tooltip", role: "tooltip", id: "app-tooltip" });
  document.body.append(tipEl);
  const show = (target) => {
    const text = target.dataset.tip;
    if (!text) return;
    tipTarget = target;
    tipEl.textContent = "";
    const parts = text.split("\n");
    parts.forEach((p, i) => { if (i) tipEl.append(h("br")); tipEl.append(p); });
    tipEl.classList.add("show");
    const r = target.getBoundingClientRect();
    const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
    let x = r.left + r.width / 2 - tw / 2;
    x = Math.max(8, Math.min(window.innerWidth - tw - 8, x));
    let y = r.bottom + 8;
    if (y + th > window.innerHeight - 8) y = r.top - th - 8;
    tipEl.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  };
  const hide = () => { tipTarget = null; tipEl.classList.remove("show"); };
  root.addEventListener("pointerover", (e) => {
    const t = e.target.closest?.("[data-tip]");
    if (t && t !== tipTarget) show(t);
    else if (!t && tipTarget) hide();
  });
  root.addEventListener("pointerleave", hide);
  root.addEventListener("focusin", (e) => { const t = e.target.closest?.("[data-tip]"); if (t) show(t); });
  root.addEventListener("focusout", hide);
  root.addEventListener("pointerdown", hide);
}

// ---------------------------------------------------------------- toasts
export function toast(message, kind = "info", ms = 3200) {
  let host = document.getElementById("toasts");
  if (!host) { host = h("div", { id: "toasts", class: "toasts", "aria-live": "polite" }); document.body.append(host); }
  const el = h("div", { class: `toast toast-${kind}` }, message);
  host.append(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 300); }, ms);
}

// ---------------------------------------------------------------- files
export function downloadText(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: filename, style: { display: "none" } });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
}

export function pickFile(accept = ".json,application/json") {
  return new Promise((resolve) => {
    const input = h("input", { type: "file", accept, style: { display: "none" } });
    input.addEventListener("change", async () => {
      const f = input.files && input.files[0];
      input.remove();
      if (!f) return resolve(null);
      resolve({ name: f.name, text: await f.text() });
    });
    document.body.append(input);
    input.click();
  });
}

export function safeFileName(name, ext) {
  const base = String(name || "config").trim().replace(/[^A-Za-z0-9_\-]+/g, "_").replace(/^_+|_+$/g, "") || "config";
  return `${base}.${ext}`;
}

// ---------------------------------------------------------------- menus
/** Attach a small dropdown menu to a button. items: [{label, hint, onClick, disabled}] or () => items. */
export function attachMenu(button, itemsOrFn, { align = "left" } = {}) {
  let menu = null;
  const close = () => {
    if (!menu) return;
    menu.remove(); menu = null;
    button.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", outside, true);
    document.removeEventListener("keydown", onKey, true);
  };
  const outside = (e) => { if (menu && !menu.contains(e.target) && !button.contains(e.target)) close(); };
  const onKey = (e) => { if (e.key === "Escape") { close(); button.focus(); } };
  button.setAttribute("aria-haspopup", "menu");
  button.setAttribute("aria-expanded", "false");
  button.addEventListener("click", async () => {
    if (menu) return close();
    const items = typeof itemsOrFn === "function" ? await itemsOrFn() : itemsOrFn;
    menu = h("div", { class: "menu", role: "menu" });
    if (!items.length) menu.append(h("div", { class: "menu-empty" }, "Nothing here yet"));
    for (const it of items) {
      if (it.separator) { menu.append(h("div", { class: "menu-sep" })); continue; }
      if (it.heading) { menu.append(h("div", { class: "menu-heading" }, it.heading)); continue; }
      const b = h("button", { class: "menu-item", role: "menuitem", disabled: !!it.disabled,
        onClick: () => { close(); it.onClick && it.onClick(); } },
        h("span", { class: "menu-label" }, it.label),
        it.hint ? h("span", { class: "menu-hint" }, it.hint) : null);
      menu.append(b);
    }
    document.body.append(menu);
    const r = button.getBoundingClientRect();
    const mw = menu.offsetWidth;
    let x = align === "right" ? r.right - mw : r.left;
    x = Math.max(8, Math.min(window.innerWidth - mw - 8, x));
    menu.style.left = `${x}px`;
    const mh = menu.offsetHeight;
    let y = r.bottom + 6;
    if (y + mh > window.innerHeight - 8) y = r.top - mh - 6 >= 8 ? r.top - mh - 6 : Math.max(8, window.innerHeight - mh - 8);
    menu.style.top = `${y}px`;
    button.setAttribute("aria-expanded", "true");
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", onKey, true);
    menu.querySelector(".menu-item:not([disabled])")?.focus();
  });
  return { close };
}

/** Segmented control (radio group styled as pills). */
export function segmented(options, value, onChange, { label = "", small = false } = {}) {
  const wrap = h("div", { class: `segmented${small ? " small" : ""}`, role: "radiogroup", "aria-label": label });
  const buttons = options.map((o) => {
    const b = h("button", { type: "button", role: "radio", class: "seg-btn", "aria-checked": String(o.value === value),
      tip: o.tip, onClick: () => { set(o.value); onChange(o.value); } }, o.icon ? icon(o.icon, 14) : null, o.label);
    b.dataset.value = o.value;
    return b;
  });
  wrap.append(...buttons);
  const set = (v) => buttons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === String(v))));
  wrap.setValue = set;
  return wrap;
}

/** Toggle switch. */
export function toggle(labelText, checked, onChange, tip) {
  const input = h("input", { type: "checkbox", role: "switch", checked });
  input.addEventListener("change", () => onChange(input.checked));
  const el = h("label", { class: "switch", tip }, input, h("span", { class: "switch-track" }, h("span", { class: "switch-thumb" })),
    h("span", { class: "switch-label" }, labelText));
  el.setChecked = (v) => { input.checked = !!v; };
  return el;
}

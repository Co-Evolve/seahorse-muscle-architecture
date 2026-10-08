// SliceEditor: route tendons by clicking holes on per-segment cross-section ("slice") views.
//
// Contract (DESIGN.md):
//   new SliceEditor(container, catalog, {onChange, onSelectTendon})
//   setConfig(config) / setActiveTendon(name) / setSegment(index)
// Extras: getSegment(), getActiveTendon(), setTool("hole" | "free"), setAutoAdvance(bool),
//         focus(), destroy(); option callback onSegmentChange(index).
//
// The editor never mutates the config it is given; after every edit it calls
// onChange(newConfig) with a fresh deep copy.
//
// Drawing orientation (stated by the compass): the slice is seen from the base, looking
// towards the tail tip. Dorsal is up, ventral is down, dextral (the animal's right) is on the
// right, sinistral on the left. SVG units are millimetres:  X = -y * 1000,  Y = x * 1000.

import {
  deepCopy, catalogIndex, getSegment, pointInfo, pointInPolygons, plateAt, nearestTap,
  describeTendon, uniqueFreePointId, removeTrunkPoint, moveTrunkPoint, SNAP_DISTANCE,
} from "./config.js";

const NS = "http://www.w3.org/2000/svg";
const BASE_VIEW = { x: -61, y: -64.5, w: 122, h: 130 };
const CORNER_TEXT = {
  ventral_dextral: "ventral · dextral",
  ventral_sinistral: "ventral · sinistral",
  dorsal_sinistral: "dorsal · sinistral",
  dorsal_dextral: "dorsal · dextral",
};
const CORNER_WORDS = {
  ventral_dextral: "ventral-dextral", ventral_sinistral: "ventral-sinistral",
  dorsal_sinistral: "dorsal-sinistral", dorsal_dextral: "dorsal-dextral",
};
const KIND_TEXT = { intermediate: "hole", ghost: "midline hole", end: "corner anchor", mvm: "MVM hole", free: "free point" };
const BRANCH_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const toSvg = (xy) => [-xy[1] * 1000, xy[0] * 1000];
const fromSvg = (p) => [p[1] / 1000, -p[0] / 1000];
const r2 = (v) => Math.round(v * 100) / 100;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function polysToPath(polys) {
  let d = "";
  for (const poly of polys || []) {
    poly.forEach((pt, i) => {
      const [X, Y] = toSvg(pt);
      d += `${i ? "L" : "M"}${r2(X)} ${r2(Y)}`;
    });
    d += "Z";
  }
  return d;
}

/** Readable text colour (#fff or dark) on a tendon colour. */
function inkOn(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ""));
  if (!m) return "#fff";
  const n = parseInt(m[1], 16);
  const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const L = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return L > 0.36 ? "#1b2230" : "#ffffff";
}

function tapShape(kind, X, Y) {
  switch (kind) {
    case "ghost": {
      const s = 1.15;
      return `<path class="se-tap-shape" d="M${r2(X)} ${r2(Y - s)}L${r2(X + s)} ${r2(Y)}L${r2(X)} ${r2(Y + s)}L${r2(X - s)} ${r2(Y)}Z"/>`;
    }
    case "end":
      return `<rect class="se-tap-shape" x="${r2(X - 1)}" y="${r2(Y - 1)}" width="2" height="2" rx="0.45"/>`;
    case "mvm": {
      let d = "";
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i + Math.PI / 6;
        d += `${i ? "L" : "M"}${r2(X + Math.cos(a) * 1.05)} ${r2(Y + Math.sin(a) * 1.05)}`;
      }
      return `<path class="se-tap-shape" d="${d}Z"/>`;
    }
    default:
      return `<circle class="se-tap-shape" cx="${r2(X)}" cy="${r2(Y)}" r="0.9"/>`;
  }
}

function legendShape(kind) {
  const inner = tapShape(kind, 0, 0).replace("se-tap-shape", "se-tap-shape se-legend-shape");
  return `<svg viewBox="-1.6 -1.6 3.2 3.2" class="se-legend-icon se-kind-${kind}" aria-hidden="true">${inner}</svg>`;
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

const ICON = {
  hole: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="5.2" fill="none" stroke="currentColor" stroke-width="1.7"/><circle cx="10" cy="10" r="1.6" fill="currentColor"/></svg>',
  free: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v4M10 13v4M3 10h4M13 10h4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><circle cx="10" cy="10" r="2.1" fill="currentColor"/></svg>',
  step: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h9M10 6l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.5 5v10" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
  plus: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 5v10M5 10h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  minus: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 10h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  fit: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 8V4h4M16 8V4h-4M4 12v4h4M16 12v4h-4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  left: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12 5l-5 5 5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  right: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M8 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  up: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 15V5M6 9l4-4 4 4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  down: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 5v10M6 11l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  trash: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11M8 6V4.5h4V6M6 6l.8 9.5h6.4L14 6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cut: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4v12M5 10c0-3 3-4 6-4h4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M12.5 12.5l4 4M16.5 12.5l-4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  edit: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 16l1-4 8-8 3 3-8 8zM11.5 5.5l3 3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
  branch: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4v12M5 10c0-3 3-4 6-4h4M12.5 3.5 15 6l-2.5 2.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

export class SliceEditor {
  /**
   * @param {HTMLElement} container
   * @param {object} catalog  parsed catalog.json
   * @param {{onChange?: Function, onSelectTendon?: Function, onSegmentChange?: Function, autoAdvance?: boolean}} options
   */
  constructor(container, catalog, { onChange = () => {}, onSelectTendon = () => {}, onSegmentChange = () => {}, autoAdvance = true } = {}) {
    this.container = container;
    this.catalog = catalog;
    this.onChange = onChange;
    this.onSelectTendon = onSelectTendon;
    this.onSegmentChange = onSegmentChange;
    this.config = { format: "seahorse-tendon-config", version: 1, free_points: [], tendons: [] };
    this.active = null;
    this.segment = 0;
    this.tool = "hole";
    this.shift = false;
    this.autoAdvance = autoAdvance;
    this.target = null; // null = trunk, {branch: i} = existing branch, {draft: k} = new branch from trunk point k
    this.view = { ...BASE_VIEW };
    this.drag = null;
    this.hover = null;
    this.flash = null; // {kind, text} transient hint
    this.lastAdded = null;
    this.numSegments = catalog.num_segments ?? catalog.segments.length;
    this._thumbCache = new Map();
    this._staticCache = new Map();
    this._build();
    this._bind();
    this.setSegment(0, { silent: true });
  }

  // -------------------------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------------------------

  setConfig(config) {
    this.config = deepCopy(config || {});
    this.config.free_points = this.config.free_points || [];
    this.config.tendons = this.config.tendons || [];
    if (this.active && !this._tendon()) this.active = null;
    this._checkTarget();
    this._closePopover();
    this._render();
  }

  setActiveTendon(name) {
    const changed = name !== this.active;
    this.active = name || null;
    if (!this._tendon()) this.active = null;
    if (changed) {
      this.target = null;
      this._closePopover();
      // Jump to the tendon's start if the current segment shows none of it.
      const t = this._tendon();
      if (t && t.path.length) {
        const segs = this._segmentsOf(t);
        if (!segs.all.has(this.segment)) {
          const s = pointInfo(t.path[0], this.catalog, this.config)?.segment;
          if (s !== undefined) { this.setSegment(s); return; }
        }
      }
    }
    this._render();
  }

  setSegment(index, { silent = false } = {}) {
    clearTimeout(this._advanceTimer);
    const i = Math.max(0, Math.min(this.numSegments - 1, Number(index) || 0));
    const dir = i - this.segment;
    const changed = i !== this.segment;
    this.segment = i;
    this._closePopover();
    this._renderStatic();
    this._render();
    if (changed) {
      this._animateSwap(dir);
      if (!silent) this.onSegmentChange(i);
    }
  }

  getSegment() { return this.segment; }
  getActiveTendon() { return this.active; }

  setTool(tool) {
    this.tool = tool === "free" ? "free" : "hole";
    this._clearPreview();
    this._render();
  }

  setAutoAdvance(on) { this.autoAdvance = !!on; this._render(); }

  focus() { this.root.focus({ preventScroll: true }); }

  destroy() {
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("keyup", this._onKeyUp);
    window.removeEventListener("blur", this._onBlur);
    document.removeEventListener("pointerdown", this._onDocPointer, true);
    this._ro?.disconnect();
    this.root.remove();
  }

  // -------------------------------------------------------------------------------------------
  // Model helpers
  // -------------------------------------------------------------------------------------------

  _tendon(cfg = this.config) { return (cfg.tendons || []).find((t) => t.name === this.active) || null; }
  _info(id) { return pointInfo(id, this.catalog, this.config); }

  _segmentsOf(t) {
    const segOf = (id) => this._info(id)?.segment;
    const trunk = (t.path || []).map(segOf);
    const branches = (t.branches || []).map((b) => (b.path || []).map(segOf));
    return { trunk, branches, all: new Set([...trunk, ...branches.flat()].filter((s) => s !== undefined)) };
  }

  _checkTarget() {
    const t = this._tendon();
    if (!t || !this.target) { if (!t) this.target = null; return; }
    if (this.target.branch !== undefined && !t.branches?.[this.target.branch]) this.target = null;
    if (this.target?.draft !== undefined && !(this.target.draft < t.path.length)) this.target = null;
  }

  /** Every place a point id is used: [{tendon, where: "trunk"|"branch", branch, index, label}]. */
  _usages(id) {
    const out = [];
    for (const t of this.config.tendons) {
      t.path.forEach((p, i) => { if (p === id) out.push({ tendon: t, where: "trunk", index: i, label: String(i + 1) }); });
      (t.branches || []).forEach((b, bi) => b.path.forEach((p, i) => {
        if (p === id) out.push({ tendon: t, where: "branch", branch: bi, index: i, label: `${BRANCH_LETTERS[bi] || bi + 1}${i + 1}` });
      }));
    }
    return out;
  }

  _commit(mutator, { added = null } = {}) {
    const cfg = deepCopy(this.config);
    const t = this._tendon(cfg);
    const res = mutator(cfg, t);
    if (res === false) return;
    // Drop free points that nothing uses any more (only those this edit orphaned).
    const used = new Set();
    for (const tt of cfg.tendons) { tt.path.forEach((p) => used.add(p)); (tt.branches || []).forEach((b) => b.path.forEach((p) => used.add(p))); }
    const before = new Set(this.config.free_points.map((f) => f.id));
    const prevUsed = new Set();
    for (const tt of this.config.tendons) { tt.path.forEach((p) => prevUsed.add(p)); (tt.branches || []).forEach((b) => b.path.forEach((p) => prevUsed.add(p))); }
    cfg.free_points = cfg.free_points.filter((f) => used.has(f.id) || !before.has(f.id) || !prevUsed.has(f.id));
    this.config = cfg;
    this.lastAdded = added;
    clearTimeout(this._newTimer);
    this._newTimer = setTimeout(() => { this.lastAdded = null; }, 450);
    this._checkTarget();
    this._render();
    this.onChange(deepCopy(cfg));
  }

  // -------------------------------------------------------------------------------------------
  // Editing operations
  // -------------------------------------------------------------------------------------------

  _append(id, { freePoint = null } = {}) {
    const t = this._tendon();
    if (!t) return;
    const info = freePoint ? { segment: freePoint.segment, kind: "free" } : this._info(id);
    let added = id;
    this._commit((cfg, tt) => {
      if (freePoint) cfg.free_points.push(freePoint);
      if (this.target?.draft !== undefined) {
        tt.branches = tt.branches || [];
        tt.branches.push({ from: this.target.draft, path: [id] });
        this.target = { branch: tt.branches.length - 1 };
      } else if (this.target?.branch !== undefined) {
        tt.branches[this.target.branch].path.push(id);
      } else {
        if (tt.path[tt.path.length - 1] === id) { this._flash("warn", "That hole is already the last point."); return false; }
        tt.path.push(id);
      }
    }, { added });
    if (this.autoAdvance && info && info.kind !== "end" && !freePoint) this._advanceAfter(info.segment);
  }

  _advanceAfter(seg) {
    const t = this._tendon();
    if (!t) return;
    const segs = this._segmentsOf(t);
    let dir = 1;
    const path = this.target?.branch !== undefined ? [segs.trunk[t.branches[this.target.branch].from], ...segs.branches[this.target.branch]] : segs.trunk;
    if (path.length >= 2 && path[path.length - 1] < path[0]) dir = -1;
    const next = seg + dir;
    if (next < 0 || next >= this.numSegments || seg !== this.segment) return;
    clearTimeout(this._advanceTimer);
    this._advanceTimer = setTimeout(() => this.setSegment(next), 260);
  }

  _removePoint(ref) {
    this._commit((cfg, t) => {
      if (ref.where === "trunk") {
        Object.assign(t, removeTrunkPoint(t, ref.index));
      } else {
        const b = t.branches[ref.branch];
        b.path.splice(ref.index, 1);
        if (!b.path.length) {
          t.branches.splice(ref.branch, 1);
          if (this.target?.branch === ref.branch) this.target = null;
          else if (this.target?.branch > ref.branch) this.target = { branch: this.target.branch - 1 };
        }
      }
    });
  }

  _movePoint(ref, delta) {
    this._commit((cfg, t) => {
      if (ref.where === "trunk") Object.assign(t, moveTrunkPoint(t, ref.index, delta));
      else {
        const p = t.branches[ref.branch].path;
        const j = ref.index + delta;
        if (j < 0 || j >= p.length) return false;
        [p[ref.index], p[j]] = [p[j], p[ref.index]];
      }
    });
  }

  _removeBranch(bi) {
    if (this.target?.branch === bi) this.target = null;
    this._commit((cfg, t) => { t.branches.splice(bi, 1); });
  }

  _startBranch(k) {
    this.target = { draft: k };
    this._closePopover();
    this._render();
    const info = this._info(this._tendon().path[k]);
    if (this.autoAdvance && info && info.segment === this.segment) this._advanceAfter(info.segment);
  }

  _createFreePoint(xy, plate) {
    const fp = { id: uniqueFreePointId(this.config), segment: this.segment, plate: plate.index, xy: [Number(xy[0].toFixed(6)), Number(xy[1].toFixed(6))] };
    this._append(fp.id, { freePoint: fp });
  }

  /** Replace every use of free point `fid` by tap `tapId` (free point is then dropped). */
  _snapFreeToTap(fid, tapId) {
    this._commit((cfg) => {
      for (const t of cfg.tendons) {
        t.path = t.path.map((p) => (p === fid ? tapId : p));
        (t.branches || []).forEach((b) => { b.path = b.path.map((p) => (p === fid ? tapId : p)); });
      }
      cfg.free_points = cfg.free_points.filter((f) => f.id !== fid);
    }, { added: tapId });
  }

  _moveFreePoint(fid, xy) {
    this._commit((cfg) => {
      const fp = cfg.free_points.find((f) => f.id === fid);
      if (fp) fp.xy = [Number(xy[0].toFixed(6)), Number(xy[1].toFixed(6))];
    });
  }

  // -------------------------------------------------------------------------------------------
  // DOM construction
  // -------------------------------------------------------------------------------------------

  _build() {
    const root = el("div", "se");
    root.tabIndex = 0;
    root.setAttribute("aria-label", "Tendon routing editor");
    this.root = root;

    // Header: active tendon + tools
    const head = el("div", "se-head");
    this.chip = el("div", "se-chip");
    const tools = el("div", "se-tools");
    this.toolSeg = el("div", "se-seg");
    this.toolSeg.setAttribute("role", "radiogroup");
    this.toolSeg.setAttribute("aria-label", "Click tool");
    this.btnHole = el("button", "se-seg-btn", `${ICON.hole}<span>Holes</span>`);
    this.btnFree = el("button", "se-seg-btn", `${ICON.free}<span>Free point</span>`);
    this.btnHole.type = this.btnFree.type = "button";
    this.btnHole.title = "Click holes to add them to the tendon";
    this.btnFree.title = "Place a point anywhere on a plate (or hold Shift while clicking)";
    this.toolSeg.append(this.btnHole, this.btnFree);
    this.btnStep = el("button", "se-icon-btn se-step", `${ICON.step}<span>Step</span>`);
    this.btnStep.type = "button";
    tools.append(this.toolSeg, this.btnStep);
    head.append(this.chip, tools);

    // Filmstrip
    const stripWrap = el("div", "se-strip-wrap");
    this.btnPrev = el("button", "se-icon-btn se-strip-nav", ICON.left);
    this.btnNext = el("button", "se-icon-btn se-strip-nav", ICON.right);
    this.btnPrev.type = this.btnNext.type = "button";
    this.btnPrev.title = "Previous segment (←)";
    this.btnNext.title = "Next segment (→)";
    this.strip = el("div", "se-strip");
    this.strip.setAttribute("role", "tablist");
    this.strip.setAttribute("aria-label", "Segments, base to tip");
    this.stripCells = [];
    for (let i = 0; i < this.numSegments; i++) {
      const b = el("button", "se-cell");
      b.type = "button";
      b.dataset.seg = i;
      b.setAttribute("role", "tab");
      b.innerHTML = `<svg class="se-thumb" viewBox="-60 -56 120 112" aria-hidden="true">${this._thumbStatic(i)}<g class="se-thumb-dots"></g></svg><span class="se-cell-num">${i}</span><span class="se-cell-bar"></span>`;
      this.stripCells.push(b);
      this.strip.append(b);
    }
    const stripEnds = el("div", "se-strip-ends", "<span>base</span><span>tip</span>");
    const stripMid = el("div", "se-strip-mid");
    stripMid.append(this.strip, stripEnds);
    stripWrap.append(this.btnPrev, stripMid, this.btnNext);

    // Mode banner
    this.banner = el("div", "se-banner");

    // Stage
    const stage = el("div", "se-stage");
    this.stage = stage;
    this.svg = document.createElementNS(NS, "svg");
    this.svg.setAttribute("class", "se-slice");
    this.svg.setAttribute("role", "img");
    this.svg.innerHTML = `
      <defs>
        <pattern id="se-hatch" width="1.6" height="1.6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="1.6" class="se-hatch-line"/>
        </pattern>
        <pattern id="se-grid" width="5" height="5" patternUnits="userSpaceOnUse">
          <circle cx="0" cy="0" r="0.2" class="se-grid-dot"/>
        </pattern>
        <radialGradient id="se-plate-grad" cx="50%" cy="50%" r="75%">
          <stop offset="0%" class="se-plate-stop-a"/><stop offset="100%" class="se-plate-stop-b"/>
        </radialGradient>
      </defs>
      <g class="se-world">
        <g class="se-axes"></g>
        <g class="se-plates"></g>
        <g class="se-vertebra"></g>
        <g class="se-corner-labels"></g>
        <g class="se-prev"></g>
        <g class="se-taps"></g>
        <g class="se-others"></g>
        <g class="se-lines"></g>
        <g class="se-used"></g>
        <g class="se-preview"></g>
      </g>`;
    this.g = {};
    for (const k of ["world", "axes", "plates", "vertebra", "corner-labels", "prev", "taps", "others", "lines", "used", "preview"]) {
      this.g[k] = this.svg.querySelector(`.se-${k}`);
    }
    this.title = el("div", "se-title");
    this.compass = el("div", "se-compass", `
      <svg viewBox="-34 -27 68 54" aria-hidden="true">
        <circle r="13" class="se-compass-ring"/>
        <path d="M0 -11 L3 -4 L-3 -4Z" class="se-compass-d"/>
        <path d="M0 11 L3 4 L-3 4Z" class="se-compass-v"/>
        <path d="M-11 0 L-4 3 L-4 -3Z" class="se-compass-s"/>
        <path d="M11 0 L4 3 L4 -3Z" class="se-compass-x"/>
        <circle r="1.6" class="se-compass-c"/>
        <text y="-17" class="se-compass-t">dorsal</text>
        <text y="23" class="se-compass-t">ventral</text>
        <text x="-16" y="2.4" class="se-compass-t se-end">sin.</text>
        <text x="16" y="2.4" class="se-compass-t se-start">dex.</text>
      </svg>`);
    this.compass.title = "Seen from the base of the tail, looking towards the tip: dorsal up, ventral down, sinistral (left side of the animal) left, dextral (right side) right.";
    const zoom = el("div", "se-zoom");
    this.btnZoomIn = el("button", "se-icon-btn", ICON.plus);
    this.btnZoomOut = el("button", "se-icon-btn", ICON.minus);
    this.btnZoomFit = el("button", "se-icon-btn", ICON.fit);
    this.btnZoomIn.title = "Zoom in (pinch or Alt + scroll)";
    this.btnZoomOut.title = "Zoom out";
    this.btnZoomFit.title = "Show the whole slice";
    for (const b of [this.btnZoomIn, this.btnZoomOut, this.btnZoomFit]) { b.type = "button"; zoom.append(b); }
    this.hint = el("div", "se-hint");
    this.hint.setAttribute("aria-live", "polite");
    this.tip = el("div", "se-tip");
    this.pop = el("div", "se-pop");
    this.pop.setAttribute("role", "menu");
    const stageHead = el("div", "se-stage-head");
    stageHead.append(this.title, this.compass);
    const stageFoot = el("div", "se-stage-foot");
    stageFoot.append(this.hint, zoom);
    stage.append(stageHead, this.svg, stageFoot, this.tip, this.pop);

    // Legend
    const legend = el("div", "se-legend");
    legend.innerHTML = ["intermediate", "ghost", "end", "mvm"].map((k) =>
      `<span class="se-legend-item">${legendShape(k)}${KIND_TEXT[k]}</span>`).join("") +
      `<span class="se-legend-item"><svg viewBox="-1.6 -1.6 3.2 3.2" class="se-legend-icon se-kind-free" aria-hidden="true"><circle r="0.9" class="se-legend-free"/><path d="M0 -1.5V-1M0 1V1.5M-1.5 0H-1M1 0H1.5" class="se-legend-free-tick"/></svg>free point</span>`;

    // Route overview
    const route = el("section", "se-route");
    const rhead = el("div", "se-route-head", `<span class="se-route-title">Route overview</span><span class="se-route-sub">base → tip, click to jump</span>`);
    this.routeSvg = document.createElementNS(NS, "svg");
    this.routeSvg.setAttribute("class", "se-route-svg");
    this.routeSvg.setAttribute("viewBox", "0 0 460 172");
    this.routeSvg.setAttribute("role", "img");
    this.routeSvg.setAttribute("aria-label", "Side and top view of the tendon routes along the tail");
    route.append(rhead, this.routeSvg);

    root.append(head, stripWrap, this.banner, stage, legend, route);
    this.container.append(root);
  }

  _thumbStatic(i) {
    const seg = getSegment(this.catalog, i);
    if (!seg) return "";
    const plates = seg.plates.map((p) => `<path d="${polysToPath(p.outline.slice(0, 1))}" class="se-thumb-plate"/>`).join("");
    const vert = `<path d="${polysToPath(seg.vertebra_outline.slice(0, 1))}" class="se-thumb-vert"/>`;
    return plates + vert;
  }

  _bind() {
    this.btnHole.addEventListener("click", () => this.setTool("hole"));
    this.btnFree.addEventListener("click", () => this.setTool(this.tool === "free" ? "hole" : "free"));
    this.btnStep.addEventListener("click", () => this.setAutoAdvance(!this.autoAdvance));
    this.btnPrev.addEventListener("click", () => this.setSegment(this.segment - 1));
    this.btnNext.addEventListener("click", () => this.setSegment(this.segment + 1));
    this.strip.addEventListener("click", (e) => {
      const b = e.target.closest(".se-cell");
      if (b) this.setSegment(Number(b.dataset.seg));
    });
    this.btnZoomIn.addEventListener("click", () => this._zoomBy(1.5));
    this.btnZoomOut.addEventListener("click", () => this._zoomBy(1 / 1.5));
    this.btnZoomFit.addEventListener("click", () => { this.view = { ...BASE_VIEW }; this._applyView(); });
    this.banner.addEventListener("click", (e) => {
      const a = e.target.closest("[data-act]");
      if (!a) return;
      if (a.dataset.act === "trunk") { this.target = null; this._render(); }
      if (a.dataset.act === "free-off") this.setTool("hole");
    });

    this.svg.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    this.svg.addEventListener("pointermove", (e) => this._onPointerMove(e));
    this.svg.addEventListener("pointerup", (e) => this._onPointerUp(e));
    this.svg.addEventListener("pointerleave", () => { if (!this.drag) { this._hideTip(); this._clearPreview(); } });
    this.svg.addEventListener("wheel", (e) => {
      if (!(e.ctrlKey || e.altKey || e.metaKey)) return;
      e.preventDefault();
      this._zoomBy(Math.exp(-e.deltaY * 0.01), this._svgPoint(e));
    }, { passive: false });

    this.pop.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-act]");
      if (b) this._popAction(b.dataset.act);
    });

    this.routeSvg.addEventListener("click", (e) => {
      const n = e.target.closest("[data-seg]");
      if (n) this.setSegment(Number(n.dataset.seg));
    });

    this._onKeyDown = (e) => {
      if (e.key === "Shift" && !this.shift) { this.shift = true; this._syncToolUi(); this._renderHint(this._tendon()); }
      if (!this._keyScope(e)) return;
      if (e.key === "ArrowLeft") { this.setSegment(this.segment - 1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { this.setSegment(this.segment + 1); e.preventDefault(); }
      else if (e.key === "Escape") {
        if (this.pop.classList.contains("is-open")) this._closePopover();
        else if (this.target) { this.target = null; this._render(); }
        else if (this.tool === "free") this.setTool("hole");
      } else if ((e.key === "f" || e.key === "F") && !e.metaKey && !e.ctrlKey) {
        this.setTool(this.tool === "free" ? "hole" : "free");
      }
    };
    this._onKeyUp = (e) => { if (e.key === "Shift") { this.shift = false; this._syncToolUi(); this._clearPreview(); this._renderHint(this._tendon()); } };
    this._onBlur = () => { this.shift = false; this._syncToolUi(); this._renderHint(this._tendon()); };
    this._onDocPointer = (e) => {
      if (this.pop.classList.contains("is-open") && !this.pop.contains(e.target) && !this.svg.contains(e.target)) this._closePopover();
    };
    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("keyup", this._onKeyUp);
    window.addEventListener("blur", this._onBlur);
    document.addEventListener("pointerdown", this._onDocPointer, true);
    if (typeof ResizeObserver !== "undefined") {
      this._ro = new ResizeObserver(() => this.root.classList.toggle("is-narrow", this.root.clientWidth < 440));
      this._ro.observe(this.root);
    }
  }

  /** Arrow keys etc. act when focus is inside the editor, or nowhere in particular. */
  _keyScope(e) {
    const a = document.activeElement;
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return false;
    if (e.target?.isContentEditable) return false;
    if (!this.root.isConnected || this.root.offsetParent === null) return false;
    return this.root.contains(a) || a === document.body || a === null;
  }

  // -------------------------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------------------------

  _renderStatic() {
    const seg = getSegment(this.catalog, this.segment);
    if (!seg) return;
    let cached = this._staticCache.get(this.segment);
    if (!cached) {
      const plates = seg.plates.map((p) =>
        `<path class="se-plate" data-plate="${p.index}" fill-rule="evenodd" d="${polysToPath(p.outline)}"/>`).join("");
      const vert = `<path class="se-vert" fill-rule="evenodd" d="${polysToPath(seg.vertebra_outline)}"/>` +
        `<path class="se-vert-hatch" fill-rule="evenodd" d="${polysToPath(seg.vertebra_outline)}"/>`;
      const taps = seg.plates.flatMap((p) => p.taps.map((t) => {
        const [X, Y] = toSvg(t.xy);
        return `<g class="se-tap se-kind-${t.kind}" data-tap="${esc(t.id)}"><circle class="se-hit" cx="${r2(X)}" cy="${r2(Y)}" r="1.36"/>${tapShape(t.kind, X, Y)}</g>`;
      })).join("");
      // Corner labels sit outside the plates, near each corner.
      const labels = seg.plates.map((p) => {
        const [sx, sy] = p.corner.split("_");
        const X = sy === "dextral" ? 34 : -34;
        const Y = sx === "ventral" ? 62.3 : -59.6;
        return `<text class="se-corner-label" data-zoom-plate="${p.index}" x="${X}" y="${Y}">${CORNER_TEXT[p.corner] || p.corner}</text>`;
      }).join("");
      const axes = `<rect x="-200" y="-200" width="400" height="400" fill="url(#se-grid)" pointer-events="none"/><line class="se-axis" x1="0" y1="-58" x2="0" y2="58"/><line class="se-axis" x1="-62" y1="0" x2="62" y2="0"/>`;
      cached = { plates, vert, taps, labels, axes };
      this._staticCache.set(this.segment, cached);
    }
    this.g.plates.innerHTML = cached.plates;
    this.g.vertebra.innerHTML = cached.vert;
    this.g.taps.innerHTML = cached.taps;
    this.g["corner-labels"].innerHTML = cached.labels;
    this.g.axes.innerHTML = cached.axes;
    this._applyView();
  }

  _render() {
    const t = this._tendon();
    this._syncToolUi();
    this._renderChip(t);
    this._renderStrip(t);
    this._renderBanner(t);
    this._renderTitle(t);
    this._renderSliceDynamic(t);
    this._renderHint(t);
    this._renderRoute(t);
    this.root.style.setProperty("--se-tendon", t?.color || "var(--se-accent)");
    this.root.style.setProperty("--se-tendon-ink", inkOn(t?.color));
    this.root.classList.toggle("has-tendon", !!t);
  }

  _syncToolUi() {
    const free = this.tool === "free" || this.shift;
    this.root.classList.toggle("is-free", free);
    this.btnHole.classList.toggle("is-on", !free);
    this.btnFree.classList.toggle("is-on", free);
    this.btnHole.setAttribute("aria-checked", String(!free));
    this.btnFree.setAttribute("aria-checked", String(free));
    this.btnStep.classList.toggle("is-on", this.autoAdvance);
    this.btnStep.setAttribute("aria-pressed", String(this.autoAdvance));
    this.btnStep.title = this.autoAdvance
      ? "Step on: after each click the view moves to the next segment. Click to turn off."
      : "Step off: the view stays on this segment after a click. Click to turn on.";
  }

  _renderChip(t) {
    if (!t) {
      const n = this.config.tendons.length;
      this.chip.innerHTML = `<span class="se-chip-swatch is-empty"></span><span class="se-chip-text"><span class="se-chip-name">${n ? "No tendon selected" : "No tendons yet"}</span><span class="se-chip-sum">${n ? "Pick a tendon in the list, or click one of its points." : "Add a tendon to start routing."}</span></span>`;
      return;
    }
    const sum = describeTendon(t, this.catalog, this.config);
    this.chip.innerHTML = `<span class="se-chip-swatch" style="background:${esc(t.color)}"></span><span class="se-chip-text"><span class="se-chip-name">${esc(t.name)}${t.group ? `<span class="se-chip-group">${esc(t.group)}</span>` : ""}</span><span class="se-chip-sum" title="${esc(sum)}">${esc(sum)}</span></span>`;
  }

  _renderStrip(t) {
    const segs = t ? this._segmentsOf(t) : null;
    const trunkSet = new Set(segs?.trunk || []);
    const branchSet = new Set((segs?.branches || []).flat());
    const lo = segs && segs.all.size ? Math.min(...segs.all) : null;
    const hi = segs && segs.all.size ? Math.max(...segs.all) : null;
    this.stripCells.forEach((cell, i) => {
      cell.classList.toggle("is-current", i === this.segment);
      cell.setAttribute("aria-selected", String(i === this.segment));
      cell.classList.toggle("is-trunk", trunkSet.has(i));
      cell.classList.toggle("is-branch", !trunkSet.has(i) && branchSet.has(i));
      cell.classList.toggle("in-span", lo !== null && i >= lo && i <= hi);
      cell.classList.toggle("span-start", lo === i);
      cell.classList.toggle("span-end", hi === i);
      const n = t ? [...(t.path || []), ...(t.branches || []).flatMap((b) => b.path)].filter((id) => this._info(id)?.segment === i).length : 0;
      cell.title = `Segment ${i}${i === 0 ? " (base, fixed)" : i === this.numSegments - 1 ? " (tip)" : ""}${n ? ` · ${n} point${n > 1 ? "s" : ""} of ${t.name}` : ""}`;
      const dots = cell.querySelector(".se-thumb-dots");
      let html = "";
      if (t) {
        const add = (id, branch) => {
          const p = this._info(id);
          if (!p || p.segment !== i) return;
          const [X, Y] = toSvg(p.xy);
          html += `<circle cx="${r2(X)}" cy="${r2(Y)}" r="11" class="se-thumb-dot${branch ? " is-branch" : ""}"/>`;
        };
        t.path.forEach((id) => add(id, false));
        (t.branches || []).forEach((b) => b.path.forEach((id) => add(id, true)));
      }
      dots.innerHTML = html;
    });
    this.btnPrev.disabled = this.segment <= 0;
    this.btnNext.disabled = this.segment >= this.numSegments - 1;
  }

  _renderBanner(t) {
    let html = "";
    if (t && this.target) {
      const bi = this.target.branch;
      const k = bi !== undefined ? t.branches[bi].from : this.target.draft;
      const letter = BRANCH_LETTERS[bi !== undefined ? bi : (t.branches || []).length] || "?";
      const at = this._info(t.path[k]);
      html = `<div class="se-banner-inner is-branch">${ICON.branch}<span><b>Branch ${letter}</b>: ${bi !== undefined ? "clicks add to this branch" : "click the next hole of the branch"}. It splits off at point ${k + 1}${at ? ` (segment ${at.segment})` : ""}.</span><button type="button" class="se-banner-btn" data-act="trunk">Back to main path</button></div>`;
    } else if (this.tool === "free" && t) {
      html = `<div class="se-banner-inner is-free">${ICON.free}<span><b>Free point</b>: click anywhere on a plate. Within 1.5 mm of a hole the point snaps to it.</span><button type="button" class="se-banner-btn" data-act="free-off">Done</button></div>`;
    }
    this.banner.innerHTML = html;
    this.banner.classList.toggle("is-open", !!html);
  }

  _renderTitle(t) {
    const where = this.segment === 0 ? "base · fixed to the body" : this.segment === this.numSegments - 1 ? "tip of the tail" : `${this.segment} of ${this.numSegments - 1}, base → tip`;
    let here = "";
    if (t) {
      const n = [...t.path, ...(t.branches || []).flatMap((b) => b.path)].filter((id) => this._info(id)?.segment === this.segment).length;
      here = n ? `<span class="se-title-count"><i style="background:${esc(t.color)}"></i>${n} point${n > 1 ? "s" : ""} here</span>` : "";
    }
    this.title.innerHTML = `<span class="se-title-main">Segment ${this.segment}</span><span class="se-title-sub">${where}</span>${here}`;
  }

  _renderSliceDynamic(t) {
    const seg = this.segment;
    const usedHere = new Map(); // tap/free id -> usages
    for (const tt of this.config.tendons) {
      const add = (id, u) => {
        const p = this._info(id);
        if (!p || p.segment !== seg) return;
        if (!usedHere.has(id)) usedHere.set(id, []);
        usedHere.get(id).push({ ...u, tendon: tt, info: p });
      };
      tt.path.forEach((id, i) => add(id, { where: "trunk", index: i, label: String(i + 1) }));
      (tt.branches || []).forEach((b, bi) => b.path.forEach((id, i) => add(id, { where: "branch", branch: bi, index: i, label: `${BRANCH_LETTERS[bi] || bi + 1}${i + 1}` })));
    }

    // Other tendons: small coloured dots around the tap.
    let others = "";
    for (const [id, us] of usedHere) {
      const foreign = us.filter((u) => u.tendon.name !== this.active);
      if (!foreign.length) continue;
      const [X, Y] = toSvg(us[0].info.xy);
      const mine = us.some((u) => u.tendon.name === this.active);
      foreign.forEach((u, j) => {
        let cx = X, cy = Y, r = 1.15;
        if (mine || foreign.length > 1) {
          const a = -Math.PI / 2 + (j - (foreign.length - 1) / 2) * 0.9 + (mine ? Math.PI / 4 : 0);
          const off = mine ? 2.75 : 1.25;
          cx = X + Math.cos(a) * off; cy = Y + Math.sin(a) * off; r = mine ? 0.85 : 0.95;
        }
        others += `<g class="se-other${this.active ? "" : " is-solo"}" data-other="${esc(u.tendon.name)}" data-point="${esc(id)}"><circle cx="${r2(cx)}" cy="${r2(cy)}" r="${r + 0.6}" class="se-hit"/><circle cx="${r2(cx)}" cy="${r2(cy)}" r="${r}" fill="${esc(u.tendon.color)}" class="se-other-dot"/></g>`;
      });
    }
    this.g.others.innerHTML = others;

    // Active tendon: previous-segment ghosts, in-segment lines, numbered markers.
    let prev = "", lines = "", used = "";
    if (t) {
      const color = esc(t.color);
      const ink = inkOn(t.color);
      const chains = [{ ids: t.path, where: "trunk", offset: 0 }];
      (t.branches || []).forEach((b, bi) => chains.push({ ids: [t.path[b.from], ...b.path], where: "branch", branch: bi, offset: -1 }));
      const prevShown = new Set();
      const usedPos = [];
      for (const id of [...t.path, ...(t.branches || []).flatMap((bb) => bb.path)]) {
        const ip = this._info(id);
        if (ip && ip.segment === seg) usedPos.push(toSvg(ip.xy));
      }
      const labelFree = (X, Y) => !usedPos.some(([ux, uy]) => Math.abs(ux - X) < 6 && Math.abs(uy - (Y - 2.6)) < 3.2);
      const ghostText = (X, Y, color, text) => (labelFree(X, Y) ? `<text x="${r2(X)}" y="${r2(Y - 2.6)}" fill="${color}">${text}</text>` : "");
      for (const ch of chains) {
        const pts = ch.ids.map((id) => (id === undefined ? null : this._info(id)));
        const dash = ch.where === "branch" ? ' stroke-dasharray="1.4 1"' : "";
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          const q = pts[i - 1];
          if (!p || !q) continue;
          const [X, Y] = toSvg(p.xy);
          const [qX, qY] = toSvg(q.xy);
          if (p.segment === seg && q.segment === seg) {
            lines += `<line x1="${r2(qX)}" y1="${r2(qY)}" x2="${r2(X)}" y2="${r2(Y)}" stroke="${color}" class="se-line${ch.where === "branch" ? " is-branch" : ""}"${dash}/>`;
          } else if (p.segment === seg && q.segment !== seg) {
            // Thread coming in from another segment: faint ghost + dotted lead line.
            lines += `<line x1="${r2(qX)}" y1="${r2(qY)}" x2="${r2(X)}" y2="${r2(Y)}" stroke="${color}" class="se-lead"/>`;
            const key = `${q.id}`;
            if (!prevShown.has(key)) {
              prevShown.add(key);
              prev += `<g class="se-ghost"><circle cx="${r2(qX)}" cy="${r2(qY)}" r="1.7" stroke="${color}"/>${ghostText(qX, qY, color, `from seg ${q.segment}`)}</g>`;
            }
          }
        }
      }
      // Points of the neighbouring (previous) segment even if not directly linked.
      const prevSeg = seg - 1;
      if (prevSeg >= 0) {
        const all = [...t.path, ...(t.branches || []).flatMap((b) => b.path)];
        for (const id of all) {
          const p = this._info(id);
          if (!p || p.segment !== prevSeg || prevShown.has(id)) continue;
          prevShown.add(id);
          const [X, Y] = toSvg(p.xy);
          prev += `<g class="se-ghost"><circle cx="${r2(X)}" cy="${r2(Y)}" r="1.7" stroke="${color}"/>${ghostText(X, Y, color, `seg ${prevSeg}`)}</g>`;
        }
      }

      // Numbered markers for this segment.
      const last = t.path.length - 1;
      const editingIds = new Set(this.target?.branch !== undefined ? t.branches[this.target.branch].path : this.target ? [] : t.path);
      for (const [id, us] of usedHere) {
        const mine = us.filter((u) => u.tendon.name === this.active);
        if (!mine.length) continue;
        const [X, Y] = toSvg(mine[0].info.xy);
        const isFree = mine[0].info.kind === "free";
        mine.forEach((u, j) => {
          const dx = j * 4.8;
          const isBranch = u.where === "branch";
          const isEnd = (u.where === "trunk" && u.index === last && last > 0) || (isBranch && u.index === t.branches[u.branch].path.length - 1);
          const isStart = u.where === "trunk" && u.index === 0;
          const cls = ["se-used", isBranch ? "is-branch" : "", isEnd ? "is-end" : "", isStart ? "is-start" : "", isFree ? "is-free" : "",
            editingIds.has(id) ? "is-editing" : "", this.lastAdded === id ? "is-new" : ""].filter(Boolean).join(" ");
          const text = u.label;
          const fs = text.length > 2 ? 1.75 : text.length > 1 ? 2.1 : 2.45;
          const ring = isEnd ? `<circle cx="${r2(X + dx)}" cy="${r2(Y)}" r="3.1" class="se-end-ring" stroke="${color}"/>` : "";
          const startRing = isStart ? `<circle cx="${r2(X + dx)}" cy="${r2(Y)}" r="3.1" class="se-start-ring" stroke="${color}"/>` : "";
          const ticks = isFree ? `<path class="se-free-ticks" stroke="${color}" d="M${r2(X + dx)} ${r2(Y - 3.5)}v1.1M${r2(X + dx)} ${r2(Y + 2.4)}v1.1M${r2(X + dx - 3.5)} ${r2(Y)}h1.1M${r2(X + dx + 2.4)} ${r2(Y)}h1.1"/>` : "";
          used += `<g class="${cls}" data-used="${esc(id)}" data-where="${u.where}" data-index="${u.index}"${isBranch ? ` data-branch="${u.branch}"` : ""} style="--cx:${r2(X + dx)}px;--cy:${r2(Y)}px">` +
            `${ring}${startRing}${ticks}<circle cx="${r2(X + dx)}" cy="${r2(Y)}" r="2.5" class="se-hit"/>` +
            `<circle cx="${r2(X + dx)}" cy="${r2(Y)}" r="2.25" class="se-used-dot" ${isBranch ? `fill="var(--se-bg)" stroke="${color}"` : `fill="${color}"`}/>` +
            `<text x="${r2(X + dx)}" y="${r2(Y + fs * 0.36)}" font-size="${fs}" fill="${isBranch ? color : ink}" class="se-used-num">${esc(text)}</text></g>`;
        });
      }
    }
    this.g.prev.innerHTML = prev;
    this.g.lines.innerHTML = lines;
    this.g.used.innerHTML = used;
    // Mark taps used by the active tendon so the plain marker hides beneath the number.
    this.g.taps.querySelectorAll(".se-tap.is-used").forEach((n) => n.classList.remove("is-used"));
    for (const [id, us] of usedHere) {
      if (us.some((u) => u.tendon.name === this.active)) this.g.taps.querySelector(`[data-tap="${CSS.escape(id)}"]`)?.classList.add("is-used");
    }
    this.svg.setAttribute("aria-label", `Cross-section of segment ${seg}. Dorsal up, ventral down, dextral right.`);
  }

  _renderHint(t) {
    let text = "";
    let tone = "info";
    if (this.flash) { text = this.flash.text; tone = this.flash.kind; }
    else if (!t) text = this.config.tendons.length ? "Pick a tendon to edit it here. Clicking a coloured point selects its tendon." : "Add a tendon to start routing.";
    else if (this.target?.draft !== undefined) text = "Click the next hole for the new branch.";
    else if (this.target?.branch !== undefined) text = `Click holes to extend branch ${BRANCH_LETTERS[this.target.branch]}. The last one is where the branch ends.`;
    else if (!t.path.length) text = "Click a hole to set where this tendon starts.";
    else if (t.path.length === 1) text = `Now add the next hole${this.autoAdvance ? "" : ", usually in the next segment (→)"}. The last point you add is where the tendon ends.`;
    else if (this.shift && this.tool !== "free") text = "Free point: click anywhere on a plate. Near a hole (1.5 mm) it snaps to the hole.";
    else if (this.tool === "free") text = "";
    else text = "Click a hole to extend the tendon · click a numbered point for options · Shift + click places a free point";
    this.hint.textContent = text;
    this.hint.dataset.tone = tone;
    this.hint.classList.toggle("is-on", !!text);
    this.hint.classList.toggle("is-quiet", !!t && t.path.length >= 2 && !this.flash);
  }

  _flash(kind, text, ms = 2600) {
    this.flash = { kind, text };
    clearTimeout(this._flashTimer);
    this._renderHint(this._tendon());
    this._flashTimer = setTimeout(() => { this.flash = null; this._renderHint(this._tendon()); }, ms);
  }

  _renderRoute(active) {
    const W = 460, padL = 68, padR = 18;
    const n = this.numSegments;
    const colX = (s) => padL + (s * (W - padL - padR)) / (n - 1);
    const lanes = [
      { key: "side", y0: 14, h: 64, label: "Side", top: "dorsal", bottom: "ventral", map: (xy) => xy[0] },
      { key: "top", y0: 98, h: 64, label: "Top", top: "sinistral", bottom: "dextral", map: (xy) => -xy[1] },
    ];
    const R = 0.046; // metres half-range shown per lane
    const laneY = (lane, v) => lane.y0 + lane.h / 2 + (v / R) * (lane.h / 2);
    let html = "";
    // Columns
    for (let s = 0; s < n; s++) {
      const x = colX(s);
      const cur = s === this.segment;
      html += `<g class="se-rcol${cur ? " is-current" : ""}" data-seg="${s}"><rect x="${x - 15}" y="4" width="30" height="164" rx="6" class="se-rcol-bg"/>` +
        `<line x1="${x}" y1="${lanes[0].y0}" x2="${x}" y2="${lanes[0].y0 + lanes[0].h}" class="se-rcol-line"/>` +
        `<line x1="${x}" y1="${lanes[1].y0}" x2="${x}" y2="${lanes[1].y0 + lanes[1].h}" class="se-rcol-line"/>` +
        `<text x="${x}" y="${lanes[1].y0 + lanes[1].h + 0}" dy="8" class="se-rcol-num">${s}</text></g>`;
    }
    for (const lane of lanes) {
      const mid = lane.y0 + lane.h / 2;
      html += `<text x="6" y="${mid - 3}" class="se-rlane">${lane.label}</text>` +
        `<text x="6" y="${mid + 9}" class="se-rlane-sub">${lane.top} ↑</text>` +
        `<line x1="${colX(0) - 8}" y1="${mid}" x2="${colX(n - 1) + 8}" y2="${mid}" class="se-rmid"/>`;
    }
    // Tendons: others faint first, active last.
    const tendons = [...this.config.tendons].sort((a, b) => (a.name === this.active) - (b.name === this.active));
    for (const t of tendons) {
      const isActive = t.name === this.active;
      const chains = [{ ids: t.path, branch: false }];
      (t.branches || []).forEach((b) => chains.push({ ids: [t.path[b.from], ...b.path], branch: true }));
      for (const lane of lanes) {
        for (const ch of chains) {
          const pts = ch.ids.map((id) => (id === undefined ? null : this._info(id))).filter(Boolean);
          if (pts.length < 1) continue;
          const d = pts.map((p, i) => `${i ? "L" : "M"}${r2(colX(p.segment))} ${r2(laneY(lane, lane.map(p.xy)))}`).join("");
          html += `<path d="${d}" stroke="${esc(t.color)}" class="se-rpath${isActive ? " is-active" : ""}${ch.branch ? " is-branch" : ""}"${isActive ? "" : ` data-tendon="${esc(t.name)}"`}/>`;
          if (isActive) {
            pts.forEach((p, i) => {
              if (ch.branch && i === 0) return;
              html += `<circle cx="${r2(colX(p.segment))}" cy="${r2(laneY(lane, lane.map(p.xy)))}" r="3.1" class="se-rnode${ch.branch ? " is-branch" : ""}" stroke="${esc(t.color)}" fill="${ch.branch ? "var(--se-bg)" : esc(t.color)}" data-seg="${p.segment}"/>`;
            });
          }
        }
      }
    }
    if (!this.config.tendons.length) {
      html += `<text x="${(colX(0) + colX(n - 1)) / 2}" y="${lanes[0].y0 + lanes[0].h + 12}" class="se-rempty">Routes appear here as you add points.</text>`;
    }
    this.routeSvg.innerHTML = html;
  }

  // -------------------------------------------------------------------------------------------
  // Pointer interaction
  // -------------------------------------------------------------------------------------------

  _svgPoint(e) {
    const pt = this.svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const m = this.svg.getScreenCTM();
    if (!m) return [0, 0];
    const p = pt.matrixTransform(m.inverse());
    return [p.x, p.y];
  }

  _stagePoint(e) {
    const r = this.stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _isFreeMode() { return this.tool === "free" || this.shift; }

  _onPointerDown(e) {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const used = e.target.closest(".se-used");
    const p = this._svgPoint(e);
    this.drag = { start: p, client: [e.clientX, e.clientY], moved: false, used, target: e.target, free: null, pan: null };
    if (used && this._info(used.dataset.used)?.kind === "free") {
      const fp = this.config.free_points.find((f) => f.id === used.dataset.used);
      if (fp) this.drag.free = { id: fp.id, plate: fp.plate, xy: fp.xy.slice(), snap: null };
    } else if (!e.target.closest(".se-tap, .se-other, .se-used, .se-corner-label") && this.view.w < BASE_VIEW.w - 0.01 && !this._isFreeMode()) {
      this.drag.pan = { view: { ...this.view } };
    }
    this.svg.setPointerCapture?.(e.pointerId);
  }

  _onPointerMove(e) {
    const p = this._svgPoint(e);
    const d = this.drag;
    if (d) {
      const dist = Math.hypot(e.clientX - d.client[0], e.clientY - d.client[1]);
      if (dist > 4) d.moved = true;
      if (d.free && d.moved) { this._dragFree(p); return; }
      if (d.pan && d.moved) {
        const rect = this.svg.getBoundingClientRect();
        const k = d.pan.view.w / rect.width;
        this.view = { ...d.pan.view, x: d.pan.view.x - (e.clientX - d.client[0]) * k, y: d.pan.view.y - (e.clientY - d.client[1]) * k };
        this._applyView();
        this.root.classList.add("is-panning");
        return;
      }
    }
    this._hover(e, p);
  }

  _onPointerUp(e) {
    const d = this.drag;
    this.drag = null;
    this.root.classList.remove("is-panning", "is-dragging");
    if (!d) return;
    if (d.free && d.moved) {
      this._clearPreview();
      if (d.free.snap) { this._snapFreeToTap(d.free.id, d.free.snap); this._flash("ok", "Snapped onto the hole."); }
      else this._moveFreePoint(d.free.id, d.free.xy);
      return;
    }
    if (d.moved) return;
    this._click(e, this._svgPoint(e), d.target);
  }

  _dragFree(svgP) {
    const d = this.drag.free;
    this.root.classList.add("is-dragging");
    const xy = fromSvg(svgP);
    const plate = getSegment(this.catalog, this.segment).plates.find((pl) => pl.index === d.plate);
    const snap = nearestTap(this.catalog, this.segment, xy, { maxDist: SNAP_DISTANCE });
    d.snap = snap ? snap.tap.id : null;
    if (snap) d.xy = snap.tap.xy.slice();
    else if (pointInPolygons(xy, plate.outline)) d.xy = xy;
    // Move the marker live.
    const node = this.g.used.querySelector(`[data-used="${CSS.escape(d.id)}"]`);
    if (node) {
      const [X, Y] = toSvg(d.xy);
      const [oX, oY] = toSvg(this.config.free_points.find((f) => f.id === d.id).xy);
      node.setAttribute("transform", `translate(${r2(X - oX)} ${r2(Y - oY)})`);
    }
    this._preview(snap ? { snap: snap.tap } : null);
    this._hideTip();
  }

  _hover(e, p) {
    const tap = e.target.closest(".se-tap");
    const used = e.target.closest(".se-used");
    const other = e.target.closest(".se-other");
    const label = e.target.closest(".se-corner-label");
    if (this._isFreeMode() && this._tendon() && !used && !other) {
      const xy = fromSvg(p);
      const snap = nearestTap(this.catalog, this.segment, xy, { maxDist: SNAP_DISTANCE });
      const plate = plateAt(this.catalog, this.segment, xy);
      this._preview({ xy, snap: snap?.tap || null, plate });
      if (snap) this._showTip(e, `Snap to ${KIND_TEXT[snap.tap.kind]}: ${snap.tap.label}`);
      else if (plate) this._showTip(e, `Free point on the ${CORNER_WORDS[plate.corner]} plate`, `${(xy[0] * 1000).toFixed(1)}, ${(xy[1] * 1000).toFixed(1)} mm`);
      else this._showTip(e, "Not on a plate", "Free points must sit on a plate.", "warn");
      return;
    }
    this._clearPreview();
    if (label) { this._showTip(e, "Zoom to this plate"); return; }
    const id = used?.dataset.used || tap?.dataset.tap || other?.dataset.point;
    if (!id) { this._hideTip(); return; }
    const info = this._info(id);
    if (!info) { this._hideTip(); return; }
    const us = this._usages(id);
    const lines = us.map((u) => `<span class="se-tip-use"><i style="background:${esc(u.tendon.color)}"></i>${esc(u.tendon.name)} · ${u.where === "trunk" ? `point ${u.label}` : `branch point ${u.label}`}</span>`).join("");
    const kindTxt = KIND_TEXT[info.kind] || info.kind;
    const head = info.kind === "free" ? "Free point" : info.label === kindTxt ? kindTxt : `${info.label}`;
    const plateTxt = `${CORNER_WORDS[info.corner] || "plate " + info.plate} plate`;
    this._showTip(e, head.charAt(0).toUpperCase() + head.slice(1), `${info.kind === "free" ? "drag to move · " : info.label.includes(kindTxt) ? "" : kindTxt + " · "}${plateTxt}`, "info", lines);
  }

  _click(e, p, target) {
    const t = this._tendon();
    const used = target.closest?.(".se-used");
    const other = target.closest?.(".se-other");
    const tap = target.closest?.(".se-tap");
    const label = target.closest?.(".se-corner-label");
    this._hideTip();
    if (label) { this._zoomToPlate(Number(label.dataset.zoomPlate)); return; }
    if (used) { this._openPopoverUsed(used, e); return; }
    if (other) {
      const name = other.dataset.other;
      if (!t) { this._selectTendon(name); return; }
      this._openPopoverOther(name, other.dataset.point, e);
      return;
    }
    if (!t) { this._flash("info", this.config.tendons.length ? "Pick a tendon first: use the tendon list, or click a coloured point." : "Add a tendon first, then click holes to route it."); return; }
    const xy = fromSvg(p);
    if (this._isFreeMode()) {
      const snap = nearestTap(this.catalog, this.segment, xy, { maxDist: SNAP_DISTANCE });
      if (snap) { this._appendTap(snap.tap.id); return; }
      const plate = plateAt(this.catalog, this.segment, xy);
      if (!plate) { this._flash("warn", "Free points must sit on a plate. Click on one of the four plates."); return; }
      this._createFreePoint(xy, plate);
      return;
    }
    if (tap) { this._appendTap(tap.dataset.tap); return; }
    // Forgiving click: nearest hole within 2 mm.
    const near = nearestTap(this.catalog, this.segment, xy, { maxDist: 0.002 });
    if (near) { this._appendTap(near.tap.id); return; }
    if (plateAt(this.catalog, this.segment, xy)) this._flash("info", "No hole here. Hold Shift (or use Free point) to place a point anywhere on a plate.");
  }

  _appendTap(id) {
    const us = this._usages(id).filter((u) => u.tendon.name === this.active);
    const editing = this.target?.branch !== undefined ? "branch" : this.target ? "draft" : "trunk";
    const inCurrent = us.find((u) => (editing === "trunk" && u.where === "trunk") || (editing === "branch" && u.where === "branch" && u.branch === this.target.branch));
    if (inCurrent) {
      const node = this.g.used.querySelector(`[data-used="${CSS.escape(id)}"]`);
      if (node) { this._openPopoverUsed(node); return; }
    }
    this._append(id);
  }

  _selectTendon(name) {
    this.setActiveTendon(name);
    this.onSelectTendon(name);
  }

  // -------------------------------------------------------------------------------------------
  // Preview / tooltip / popover
  // -------------------------------------------------------------------------------------------

  _preview(state) {
    if (!state) { this._clearPreview(); return; }
    let html = "";
    if (state.snap) {
      const [X, Y] = toSvg(state.snap.xy);
      html = `<circle cx="${r2(X)}" cy="${r2(Y)}" r="3" class="se-snap-ring"/>`;
    } else if (state.xy) {
      const [X, Y] = toSvg(state.xy);
      html = state.plate
        ? `<circle cx="${r2(X)}" cy="${r2(Y)}" r="1.5" class="se-free-preview"/><path class="se-free-cross" d="M${r2(X)} ${r2(Y - 3.4)}v1.6M${r2(X)} ${r2(Y + 1.8)}v1.6M${r2(X - 3.4)} ${r2(Y)}h1.6M${r2(X + 1.8)} ${r2(Y)}h1.6"/>`
        : `<path class="se-free-no" d="M${r2(X - 1.3)} ${r2(Y - 1.3)}l2.6 2.6M${r2(X + 1.3)} ${r2(Y - 1.3)}l-2.6 2.6"/>`;
    }
    this.g.preview.innerHTML = html;
    this.g.plates.querySelectorAll(".se-plate").forEach((n) => n.classList.toggle("is-target", !!state.plate && Number(n.dataset.plate) === state.plate.index));
  }

  _clearPreview() {
    if (this.g.preview.firstChild) this.g.preview.innerHTML = "";
    this.g.plates.querySelectorAll(".se-plate.is-target").forEach((n) => n.classList.remove("is-target"));
  }

  _showTip(e, title, sub = "", tone = "info", extra = "") {
    this.tip.innerHTML = `<b>${esc(title)}</b>${sub ? `<span>${esc(sub)}</span>` : ""}${extra}`;
    this.tip.dataset.tone = tone;
    const [x, y] = this._stagePoint(e);
    const w = this.stage.clientWidth;
    this.tip.classList.add("is-on");
    const tw = this.tip.offsetWidth;
    const left = Math.max(6, Math.min(w - tw - 6, x + 14));
    const top = y > 70 ? y - this.tip.offsetHeight - 12 : y + 18;
    this.tip.style.transform = `translate(${left}px, ${top}px)`;
  }

  _hideTip() { this.tip.classList.remove("is-on"); }

  _placePopover(anchorNode) {
    const sr = this.stage.getBoundingClientRect();
    const r = anchorNode.getBoundingClientRect();
    const cx = r.left + r.width / 2 - sr.left;
    const below = r.bottom - sr.top + 8;
    this.pop.classList.add("is-open");
    const pw = this.pop.offsetWidth, ph = this.pop.offsetHeight;
    let left = Math.max(8, Math.min(sr.width - pw - 8, cx - pw / 2));
    let top = below;
    let flip = false;
    if (top + ph > sr.height - 6) { top = r.top - sr.top - ph - 8; flip = true; }
    top = Math.max(6, top);
    this.pop.style.left = `${left}px`;
    this.pop.style.top = `${top}px`;
    this.pop.style.setProperty("--arrow-x", `${Math.max(14, Math.min(pw - 14, cx - left))}px`);
    this.pop.classList.toggle("is-above", flip);
  }

  _openPopoverUsed(node) {
    const t = this._tendon();
    if (!t) return;
    const id = node.dataset.used;
    const where = node.dataset.where;
    const index = Number(node.dataset.index);
    const branch = node.dataset.branch !== undefined ? Number(node.dataset.branch) : undefined;
    this._popRef = { id, where, index, branch };
    const info = this._info(id);
    const title = where === "trunk" ? `Point ${index + 1}` : `Branch ${BRANCH_LETTERS[branch]}, point ${index + 1}`;
    const pathLen = where === "trunk" ? t.path.length : t.branches[branch].path.length;
    const btn = (act, label, icon = "", disabled = false, cls = "") =>
      `<button type="button" role="menuitem" data-act="${act}" class="${cls}"${disabled ? " disabled" : ""}>${icon}<span>${label}</span></button>`;
    let items = "";
    items += btn("earlier", "Move earlier", ICON.up, index === 0);
    items += btn("later", "Move later", ICON.down, index >= pathLen - 1);
    if (where === "trunk") {
      const fromHere = (t.branches || []).map((b, bi) => (b.from === index ? bi : -1)).filter((x) => x >= 0);
      if (index < t.path.length - 1 || t.path.length === 1) items += btn("split", "Split here (start a branch)", ICON.branch, t.path.length < 2);
      for (const bi of fromHere) items += btn(`edit-branch:${bi}`, `Edit branch ${BRANCH_LETTERS[bi]}`, ICON.branch);
    } else {
      if (this.target?.branch !== branch) items += btn(`edit-branch:${branch}`, "Continue this branch", ICON.branch);
      items += btn(`del-branch:${branch}`, `Remove branch ${BRANCH_LETTERS[branch]}`, ICON.cut, false, "is-danger");
    }
    items += btn("remove", "Remove point", ICON.trash, false, "is-danger");
    const sub = info ? `${info.kind === "free" ? "free point" : info.label} · ${CORNER_WORDS[info.corner] || ""}` : id;
    this.pop.innerHTML = `<div class="se-pop-head"><i style="background:${esc(t.color)}"></i><b>${title}</b><span>${esc(sub)}</span></div><div class="se-pop-items">${items}</div>`;
    this._placePopover(node);
    this.pop.querySelector("button:not([disabled])")?.focus({ preventScroll: true });
  }

  _openPopoverOther(name, pointId, e) {
    const other = this.config.tendons.find((t) => t.name === name);
    const t = this._tendon();
    this._popRef = { other: name, id: pointId };
    this.pop.innerHTML = `<div class="se-pop-head"><i style="background:${esc(other?.color)}"></i><b>${esc(name)}</b><span>uses this hole</span></div><div class="se-pop-items">` +
      `<button type="button" role="menuitem" data-act="select-other">${ICON.edit}<span>Edit ${esc(name)}</span></button>` +
      (t ? `<button type="button" role="menuitem" data-act="share">${ICON.plus}<span>Also route ${esc(t.name)} through here</span></button>` : "") + "</div>";
    const node = e.target.closest(".se-other") || e.target;
    this._placePopover(node);
    this.pop.querySelector("button")?.focus({ preventScroll: true });
  }

  _closePopover() {
    this.pop?.classList.remove("is-open");
    this._popRef = null;
  }

  _popAction(act) {
    const ref = this._popRef;
    this._closePopover();
    if (!ref) return;
    if (act === "select-other") { this._selectTendon(ref.other); return; }
    if (act === "share") { this._append(ref.id); return; }
    if (act === "remove") { this._removePoint(ref); return; }
    if (act === "earlier") { this._movePoint(ref, -1); return; }
    if (act === "later") { this._movePoint(ref, +1); return; }
    if (act === "split") { this._startBranch(ref.index); return; }
    if (act.startsWith("edit-branch:")) { this.target = { branch: Number(act.split(":")[1]) }; this._render(); return; }
    if (act.startsWith("del-branch:")) { this._removeBranch(Number(act.split(":")[1])); }
  }

  // -------------------------------------------------------------------------------------------
  // View (zoom / pan) and animation
  // -------------------------------------------------------------------------------------------

  _applyView() {
    const v = this.view;
    this.svg.setAttribute("viewBox", `${r2(v.x)} ${r2(v.y)} ${r2(v.w)} ${r2(v.h)}`);
    this.root.classList.toggle("is-zoomed", v.w < BASE_VIEW.w - 0.01);
    this.root.style.setProperty("--se-zoom", String(BASE_VIEW.w / v.w));
  }

  _zoomBy(f, at = null) {
    const v = this.view;
    const w = Math.max(BASE_VIEW.w / 6, Math.min(BASE_VIEW.w, v.w / f));
    const k = w / v.w;
    const c = at || [v.x + v.w / 2, v.y + v.h / 2];
    let x = c[0] - (c[0] - v.x) * k;
    let y = c[1] - (c[1] - v.y) * k;
    if (w >= BASE_VIEW.w - 0.01) { x = BASE_VIEW.x; y = BASE_VIEW.y; }
    this.view = { x, y, w, h: (w * BASE_VIEW.h) / BASE_VIEW.w };
    this._applyView();
  }

  _zoomToPlate(index) {
    const plate = getSegment(this.catalog, this.segment)?.plates.find((p) => p.index === index);
    if (!plate) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const pt of plate.outline[0]) {
      const [X, Y] = toSvg(pt);
      x0 = Math.min(x0, X); x1 = Math.max(x1, X); y0 = Math.min(y0, Y); y1 = Math.max(y1, Y);
    }
    const w = Math.max(x1 - x0, ((y1 - y0) * BASE_VIEW.w) / BASE_VIEW.h) * 1.18;
    const h = (w * BASE_VIEW.h) / BASE_VIEW.w;
    this.view = { x: (x0 + x1) / 2 - w / 2, y: (y0 + y1) / 2 - h / 2, w, h };
    this.svg.classList.add("is-zooming");
    this._applyView();
    setTimeout(() => this.svg.classList.remove("is-zooming"), 400);
  }

  _animateSwap(dir) {
    const w = this.g.world;
    w.classList.remove("se-swap-l", "se-swap-r");
    void w.getBoundingClientRect();
    w.classList.add(dir > 0 ? "se-swap-r" : "se-swap-l");
  }
}

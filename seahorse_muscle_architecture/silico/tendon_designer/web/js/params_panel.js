// "Body" parameters panel: sliders + number inputs for every `params` entry of DESIGN.md
// (except orientation/gravity, which live in the simulation controls). Values are stored in
// SI units in the config; some are shown in friendlier units (N·mm, ms) via `scale`.
//
//   const p = new ParamsPanel(container, catalog, {onChange: (params, key) => ...});
//   p.setParams(config.params);

import { h, clear, icon, fmtParam, deepCopy, parseNum } from "./ui.js";

const JOINT_TIPS = {
  pitch: "Pitch = bending towards ventral or dorsal (curling the tail forward or backward).",
  roll: "Roll = bending sideways, towards sinistral (left) or dextral (right).",
  yaw: "Yaw = twisting one segment around the long axis of the tail.",
};
const STIFF_TIP = "How strongly the joint springs back to straight. Higher = you need more torque to bend it the same angle.\nUnit: N·mm per radian (1 radian ≈ 57°).";
const DAMP_TIP = "How much the joint resists fast movement, like moving through honey. It slows the motion down but does not change the final shape.\nUnit: N·mm·s per radian.";
const RANGE_TIP = "The largest angle this joint can bend in each direction before it hits its stop.\nUnit: degrees.";

/** Parameter definitions. path = location inside config.params. */
export function paramDefs(catalog) {
  const d = catalog?.defaults || {};
  const v = d.vertebra || {};
  const defs = [];
  for (const ax of ["pitch", "roll", "yaw"]) {
    const title = ax[0].toUpperCase() + ax.slice(1);
    defs.push({ section: "Vertebral joints", sub: `${title} (${ax === "pitch" ? "ventral / dorsal" : ax === "roll" ? "sideways" : "twist"})`, subTip: JOINT_TIPS[ax] });
    defs.push({ path: ["vertebra", `${ax}_stiffness`], label: "Stiffness", unit: "N·mm/rad", scale: 1000, log: true, def: v[`${ax}_stiffness`], tip: `${JOINT_TIPS[ax]}\n${STIFF_TIP}` });
    defs.push({ path: ["vertebra", `${ax}_damping`], label: "Damping", unit: "N·mm·s/rad", scale: 1000, log: true, def: v[`${ax}_damping`], tip: `${JOINT_TIPS[ax]}\n${DAMP_TIP}` });
    defs.push({ path: ["vertebra", `${ax}_range_deg`], label: "Range", unit: "± °", scale: 1, min: 0.5, max: 60, step: 0.5, def: v[`${ax}_range_deg`], tip: `${JOINT_TIPS[ax]}\n${RANGE_TIP}` });
  }
  defs.push({ path: ["stiffness_taper"], label: "Stiffness taper", unit: "× at tip", scale: 1, min: 0.1, max: 3, step: 0.05, def: 1,
    tip: "Makes the joints stiffer or softer towards the tip. 1 = every joint the same. 0.5 = the last joint is half as stiff as the first one, 2 = twice as stiff. The change is gradual (a straight line) along the tail." });
  defs.push({ section: "Bony plates", sub: "Plate gliding", subTip: "The bony plates can slide a little over each other. A spring pulls each plate back to its rest place." });
  defs.push({ path: ["plate_glide", "stiffness"], label: "Stiffness", unit: "N/m", scale: 1, log: true, def: d.plate_glide?.stiffness,
    tip: "Spring that pulls a sliding plate back to its rest place. Higher = plates slide less.\nUnit: newton per metre of sliding." });
  defs.push({ path: ["plate_glide", "damping"], label: "Damping", unit: "N·s/m", scale: 1, log: true, def: d.plate_glide?.damping,
    tip: "Resistance of the plates to fast sliding (like friction in a fluid). Slows motion, does not change the final shape.\nUnit: N·s per metre." });
  defs.push({ section: "Vertebral struts", sub: "Ligament-like struts", subTip: "Elastic struts connect neighbouring vertebrae on four sides, a bit like ligaments." });
  defs.push({ path: ["strut", "stiffness"], label: "Stiffness", unit: "N/m", scale: 1, log: true, def: d.strut?.stiffness,
    tip: "How strongly the struts resist being stretched. Higher = the whole tail is stiffer.\nUnit: newton per metre of stretch." });
  defs.push({ path: ["strut", "damping"], label: "Damping", unit: "N·s/m", scale: 1, log: true, def: d.strut?.damping,
    tip: "Resistance of the struts to fast stretching. Slows motion, does not change the final shape.\nUnit: N·s per metre." });
  defs.push({ section: "Simulation", sub: "Numerical settings", subTip: "Settings of the computer simulation itself, not of the animal or robot." });
  defs.push({ path: ["timestep"], label: "Time step", unit: "ms", scale: 1000, min: 0.25, max: 5, step: 0.05, def: d.timestep ?? 0.002,
    tip: "How far the simulation jumps forward in time per calculation. Smaller = more accurate but slower. If the tail shakes or flies apart, make this smaller." });
  return defs;
}

export function getPath(obj, path) { let o = obj; for (const k of path) { if (o == null) return undefined; o = o[k]; } return o; }
export function setPath(obj, path, value) {
  let o = obj;
  for (const k of path.slice(0, -1)) { if (!o[k] || typeof o[k] !== "object") o[k] = {}; o = o[k]; }
  o[path[path.length - 1]] = value;
}
export function deletePath(obj, path) {
  const parents = [];
  let o = obj;
  for (const k of path.slice(0, -1)) { if (!o[k]) return; parents.push([o, k]); o = o[k]; }
  delete o[path[path.length - 1]];
  for (let i = parents.length - 1; i >= 0; i--) {
    const [p, k] = parents[i];
    if (p[k] && typeof p[k] === "object" && !Object.keys(p[k]).length) delete p[k];
  }
}

export class ParamsPanel {
  constructor(container, catalog, { onChange } = {}) {
    this.container = container;
    this.catalog = catalog;
    this.onChange = onChange || (() => {});
    this.params = {};
    this.defs = paramDefs(catalog);
    this.rows = [];
    this.build();
  }
  build() {
    clear(this.container);
    const root = h("div", { class: "params" });
    root.append(h("p", { class: "panel-intro" },
      "Change how stiff and how mobile the tail is. Hover a label for an explanation. A dot marks values that differ from the model default."));
    const resetAll = h("button", { class: "btn btn-ghost btn-small", onClick: () => {
      const keep = {};
      for (const k of ["orientation", "gravity"]) if (this.params[k] !== undefined) keep[k] = this.params[k];
      this.params = keep; this.refresh(); this.onChange(this._clean(), "params:all"); } }, icon("reset", 14), "Reset all to default");
    let sectionEl = null;
    for (const d of this.defs) {
      if (d.section) {
        if (!sectionEl || sectionEl.dataset.name !== d.section) {
          sectionEl = h("section", { class: "param-section", dataset: { name: d.section } }, h("h3", { class: "param-section-title" }, d.section));
          root.append(sectionEl);
        }
        sectionEl.append(h("div", { class: "param-sub", tip: d.subTip }, d.sub));
        continue;
      }
      const row = this._row(d);
      (sectionEl || root).append(row.el);
      this.rows.push(row);
    }
    root.append(h("div", { class: "param-footer" }, resetAll));
    this.container.append(root);
  }
  _range(d) {
    if (d.log) {
      const dv = (d.def || 1) * d.scale;
      return { log: true, min: Math.log10(dv) - 2, max: Math.log10(dv) + 2 };
    }
    return { log: false, min: d.min, max: d.max };
  }
  _row(d) {
    const id = `param-${d.path.join("-")}`;
    const r = this._range(d);
    const slider = h("input", { type: "range", id, min: r.min, max: r.max, step: r.log ? 0.01 : d.step, "aria-label": d.label });
    const num = h("input", { type: "text", inputmode: "decimal", class: "num", spellcheck: false, "aria-label": `${d.label} value` });
    const dot = h("span", { class: "changed-dot", "aria-hidden": "true" });
    const reset = h("button", { class: "icon-btn reset-btn", tip: `Reset to default (${fmtParam((d.def ?? 0) * d.scale)} ${d.unit})`, "aria-label": `Reset ${d.label} to default` }, icon("reset", 14));
    const el = h("div", { class: "param-row" },
      h("label", { class: "param-label", for: id, tip: d.tip }, dot, d.label, h("span", { class: "unit" }, d.unit)),
      h("div", { class: "param-inputs" }, slider, num, reset));
    const toDisplay = (sv) => (r.log ? Math.pow(10, sv) : sv);
    const commit = (display, key) => {
      if (!Number.isFinite(display)) return;
      const si = display / d.scale;
      setPath(this.params, d.path, si);
      this._sync(row);
      this.onChange(this._clean(), key);
    };
    slider.addEventListener("input", () => { const v = toDisplay(Number(slider.value)); num.value = fmtParam(Number(v.toPrecision(3))); commit(Number(v.toPrecision(3)), `param:${d.path.join(".")}`); });
    num.addEventListener("change", () => {
      const v = parseNum(num.value);
      if (!Number.isFinite(v) || v < 0) { num.classList.add("invalid"); return; }
      num.classList.remove("invalid");
      commit(v, `param:${d.path.join(".")}:typed`);
    });
    reset.addEventListener("click", () => { deletePath(this.params, d.path); this._sync(row); this.onChange(this._clean(), `param:${d.path.join(".")}:reset`); });
    const row = { d, el, slider, num, reset, dot, r };
    return row;
  }
  _sync(row) {
    const { d, slider, num, el, r } = row;
    const cur = getPath(this.params, d.path);
    const has = cur !== undefined && cur !== null;
    const si = has ? cur : d.def;
    const display = (si ?? 0) * d.scale;
    if (document.activeElement !== num) num.value = fmtParam(display);
    if (document.activeElement !== slider) slider.value = r.log ? Math.log10(Math.max(1e-12, display)) : display;
    const changed = has && Math.abs(cur - (d.def ?? 0)) > 1e-12 * Math.max(1, Math.abs(d.def ?? 0));
    el.classList.toggle("changed", changed);
    row.reset.disabled = !has;
  }
  _clean() { return deepCopy(this.params); }
  setParams(params) {
    this.params = deepCopy(params || {});
    this.refresh();
  }
  refresh() { for (const row of this.rows) this._sync(row); }
}

// Tendon list (add / duplicate / delete / select) and the editor for the selected tendon
// (name, colour, group, actuator + parameters, passive properties, path & splits).
// All edits go through `edit(fn, key)`, which applies fn to a copy of the config, records
// undo history and rebuilds the simulation (see app.js).

import { h, clear, icon, segmented, fmtParam, parseNum } from "./ui.js";
import { isActuated, ACTUATOR_DEFAULTS, NAME_RE } from "./config_tools.js";

const ACT_INFO = {
  motor: "Motor: pulls with a force. 100 % activation = the maximum force below.",
  position: "Position: shortens the tendon towards a target length. 100 % activation = shortened by the maximum shortening; the force is limited by the maximum force.",
  none: "Passive: not driven. It only acts like an elastic band if you give it a stiffness.",
};
const GROUP_SUGGESTIONS = ["ventral", "dorsal", "sinistral", "dextral", "ventral_sinistral", "ventral_dextral"];

export class TendonPanel {
  /**
   * helpers (see app.js makeHelpers): newTendon(config, overrides), uniqueName(config, base),
   * pointInfo(id, config), describePoint(id, config), removeTrunkPoint(tendon, k),
   * mirror(config, name) -> {config, name, notes} | null, replaceConfig(config, key)
   */
  constructor(listEl, detailEl, { catalog, edit, select, helpers }) {
    this.listEl = listEl; this.detailEl = detailEl;
    this.catalog = catalog; this.edit = edit; this.select = select; this.H = helpers;
    this.config = null; this.selected = null; this.issues = [];
    this._detailFor = null;
  }
  setCatalog(catalog) { this.catalog = catalog; }

  render(config, selected, issues = []) {
    this.config = config; this.selected = selected; this.issues = issues;
    this._renderList();
    this._renderDetail();
  }

  _issuesFor(name) { return this.issues.filter((i) => i.tendon === name); }

  _renderList() {
    const cfg = this.config;
    clear(this.listEl);
    const H = this.H;
    const head = h("div", { class: "list-head" },
      h("span", { class: "count" }, cfg.tendons.length === 1 ? "1 tendon" : `${cfg.tendons.length} tendons`),
      h("button", { class: "btn btn-primary btn-small", onClick: () => this.add(), tip: "Add a new, empty tendon. Then click holes in the slice view below to give it a path." }, icon("plus", 14), "Add tendon"));
    this.listEl.append(head);
    if (!cfg.tendons.length) {
      this.listEl.append(h("div", { class: "empty-state" },
        h("div", { class: "empty-title" }, "No tendons yet"),
        h("p", {}, "Add a tendon, then click a start hole, any via holes and an end hole in the slice view. Or load a preset from the top bar.")));
      return;
    }
    const ul = h("ul", { class: "tendon-list", role: "listbox", "aria-label": "Tendons" });
    for (const t of cfg.tendons) {
      const iss = this._issuesFor(t.name);
      const nErr = iss.filter((i) => i.level === "error").length;
      const segs = [...t.path, ...(t.branches || []).flatMap((b) => b.path || [])].map((id) => H.pointInfo(id, cfg)?.segment).filter((x) => x !== undefined);
      const span = segs.length ? [Math.min(...segs), Math.max(...segs)] : null;
      const spanText = span ? (span[0] === span[1] ? `seg ${span[0]}` : `seg ${span[0]}–${span[1]}`) : "no path yet";
      const act = t.actuator?.type === "motor" ? `${fmtParam(t.actuator.max_force)} N` : t.actuator?.type === "position" ? `−${Math.round((t.actuator.max_strain ?? 0.26) * 100)} %` : "passive";
      const li = h("li", { class: `tendon-item${t.name === this.selected ? " selected" : ""}${nErr ? " has-error" : ""}`, role: "option",
        "aria-selected": String(t.name === this.selected), tabindex: 0,
        onClick: () => this.select(t.name),
        onKeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.select(t.name); } } },
        h("span", { class: "swatch", style: { background: t.color } }),
        h("span", { class: "tendon-main" },
          h("span", { class: "tendon-name" }, t.name),
          h("span", { class: "tendon-meta" }, [t.group ? h("span", { class: "chip" }, t.group) : null, `${act}, ${spanText}`])),
        nErr ? h("span", { class: "issue-badge", tip: iss.map((i) => i.message).join("\n") }, "!") : null,
        h("span", { class: "row-actions" },
          H.mirror ? h("button", { class: "icon-btn", tip: "Add a mirrored copy (sinistral ↔ dextral)", "aria-label": `Mirror ${t.name}`, onClick: (e) => { e.stopPropagation(); this.mirror(t.name); } }, icon("mirror", 14)) : null,
          h("button", { class: "icon-btn", tip: "Duplicate", "aria-label": `Duplicate ${t.name}`, onClick: (e) => { e.stopPropagation(); this.duplicate(t.name); } }, icon("copy", 14)),
          h("button", { class: "icon-btn danger", tip: "Delete", "aria-label": `Delete ${t.name}`, onClick: (e) => { e.stopPropagation(); this.remove(t.name); } }, icon("trash", 14))));
      ul.append(li);
    }
    this.listEl.append(ul);
  }

  add() {
    let name = null;
    this.edit((cfg) => {
      const base = `tendon_${cfg.tendons.length + 1}`;
      const t = this.H.newTendon(cfg, { name: base });
      t.name = this.H.uniqueName(cfg, base);
      if (!t.group) t.group = t.name;
      name = t.name;
      cfg.tendons.push(t);
    }, "tendon:add");
    this.select(name);
  }
  duplicate(src) {
    let name = null;
    this.edit((cfg) => {
      const t = cfg.tendons.find((x) => x.name === src);
      if (!t) return;
      const c = JSON.parse(JSON.stringify(t));
      c.name = this.H.uniqueName(cfg, `${t.name}_copy`);
      name = c.name;
      cfg.tendons.splice(cfg.tendons.indexOf(t) + 1, 0, c);
    }, "tendon:duplicate");
    if (name) this.select(name);
  }
  mirror(name) {
    const r = this.H.mirror?.(this.config, name);
    if (!r) return;
    this.H.replaceConfig(r.config, "tendon:mirror");
    this.select(r.name);
    if (r.notes?.length) this.H.notify?.(r.notes.join(" "), "info");
  }
  remove(name) {
    const i = this.config.tendons.findIndex((t) => t.name === name);
    this.edit((cfg) => { cfg.tendons = cfg.tendons.filter((t) => t.name !== name); }, "tendon:delete");
    const rest = this.config.tendons;
    this.select(rest.length ? rest[Math.max(0, Math.min(i, rest.length - 1))].name : null);
  }

  _renderDetail() {
    const cfg = this.config;
    const t = cfg.tendons.find((x) => x.name === this.selected);
    // Keep focus while typing: only rebuild when the selected tendon or its structure changed.
    const active = document.activeElement;
    const focusKey = active && this.detailEl.contains(active) ? active.dataset.field : null;
    clear(this.detailEl);
    this.detailEl.classList.toggle("hidden", !t);
    if (!t) return;
    const upd = (fn, key) => this.edit((c) => { const x = c.tendons.find((y) => y.name === t.name); if (x) fn(x, c); }, key);

    const nameIn = h("input", { class: "text-input", value: t.name, spellcheck: false, dataset: { field: "name" }, "aria-label": "Tendon name" });
    const nameErr = h("div", { class: "field-error" });
    nameIn.addEventListener("input", () => {
      const v = nameIn.value.trim();
      let msg = "";
      if (!v) msg = "A name is needed.";
      else if (!NAME_RE.test(v)) msg = "Use only letters, digits, _ and - (no spaces).";
      else if (v.startsWith("segment_")) msg = "Names cannot start with “segment_”.";
      else if (v !== t.name && cfg.tendons.some((x) => x.name === v)) msg = "Another tendon already has this name.";
      nameErr.textContent = msg;
      nameIn.classList.toggle("invalid", !!msg);
    });
    nameIn.addEventListener("change", () => {
      const v = nameIn.value.trim();
      if (!v || !NAME_RE.test(v) || v.startsWith("segment_") || (v !== t.name && cfg.tendons.some((x) => x.name === v))) { nameIn.value = t.name; nameErr.textContent = ""; nameIn.classList.remove("invalid"); return; }
      if (v === t.name) return;
      const old = t.name;
      this.edit((c) => { const x = c.tendons.find((y) => y.name === old); if (x) x.name = v; }, "tendon:rename");
      this.select(v, old);
    });
    const color = h("input", { type: "color", class: "color-input", value: t.color || "#2a78d6", "aria-label": "Tendon colour", dataset: { field: "color" } });
    color.addEventListener("input", () => upd((x) => { x.color = color.value; }, `tendon:color:${t.name}`));

    const groups = [...new Set([...cfg.tendons.map((x) => x.group).filter(Boolean), ...GROUP_SUGGESTIONS])];
    const dl = h("datalist", { id: "group-suggestions" }, groups.map((g) => h("option", { value: g })));
    const group = h("input", { class: "text-input", value: t.group || "", list: "group-suggestions", spellcheck: false, dataset: { field: "group" }, "aria-label": "Group" });
    group.addEventListener("change", () => upd((x) => { x.group = group.value.trim(); }, "tendon:group"));

    const type = t.actuator?.type || "none";
    const actSeg = segmented([
      { value: "motor", label: "Motor", tip: ACT_INFO.motor },
      { value: "position", label: "Position", tip: ACT_INFO.position },
      { value: "none", label: "Passive", tip: ACT_INFO.none },
    ], type, (v) => upd((x) => {
      const prev = x.actuator || {};
      x.actuator = { ...JSON.parse(JSON.stringify(ACTUATOR_DEFAULTS[v])) };
      if (v !== "none" && prev.max_force && v === "motor") x.actuator.max_force = prev.max_force;
    }, "tendon:actuator"), { label: "Actuator type", small: true });

    const numField = (label, unit, value, onSet, tip, { scale = 1, field, min = 0 } = {}) => {
      const inp = h("input", { type: "text", inputmode: "decimal", class: "num", value: fmtParam(value * scale), spellcheck: false, dataset: { field }, "aria-label": label });
      inp.addEventListener("change", () => {
        const v = parseNum(inp.value);
        if (!Number.isFinite(v) || v < min) { inp.classList.add("invalid"); return; }
        inp.classList.remove("invalid");
        onSet(v / scale);
      });
      return h("label", { class: "field", tip }, h("span", { class: "field-label" }, label, unit ? h("span", { class: "unit" }, unit) : null), inp);
    };

    const actFields = h("div", { class: "field-grid" });
    if (type === "motor") {
      actFields.append(numField("Max force", "N", t.actuator.max_force ?? 10, (v) => upd((x) => { x.actuator.max_force = v; }, "tendon:maxf"), "Pulling force at 100 % activation.", { field: "max_force" }));
    } else if (type === "position") {
      actFields.append(
        numField("Max shortening", "%", t.actuator.max_strain ?? 0.26, (v) => upd((x) => { x.actuator.max_strain = v; }, "tendon:strain"), "How much shorter the tendon tries to become at 100 % activation, as a percentage of its rest length.", { scale: 100, field: "max_strain" }),
        numField("Max force", "N", t.actuator.max_force ?? 1000, (v) => upd((x) => { x.actuator.max_force = v; }, "tendon:maxf"), "The actuator never pulls harder than this.", { field: "max_force" }),
        numField("Gain (kp)", "N/m", t.actuator.kp ?? 1000, (v) => upd((x) => { x.actuator.kp = v; }, "tendon:kp"), "How hard the actuator pulls per metre that the tendon is still too long. Higher = it reaches its target length faster and more exactly.", { field: "kp" }));
    }
    const passiveFields = h("div", { class: "field-grid" },
      numField("Stiffness", "N/m", t.stiffness ?? 0, (v) => upd((x) => { x.stiffness = v; }, "tendon:stiff"), "Elastic resistance of the tendon itself when it is stretched beyond its rest length (newton per metre of stretch). 0 = a slack string.", { field: "stiffness" }),
      numField("Damping", "N·s/m", t.damping ?? 0.01, (v) => upd((x) => { x.damping = v; }, "tendon:damp"), "Resistance of the tendon to fast changes in length.", { field: "damping" }),
      numField("Width", "mm", t.width ?? 0.001, (v) => upd((x) => { x.width = v; }, "tendon:width"), "Only changes how thick the tendon is drawn in the 3D view.", { scale: 1000, field: "width" }));

    // ---- path
    const H = this.H;
    const pathList = h("ol", { class: "path-list" });
    (t.path || []).forEach((id, i) => {
      const info = H.pointInfo(id, cfg);
      const role = i === 0 ? "start" : i === t.path.length - 1 ? "end" : "via";
      const splits = (t.branches || []).filter((b) => b.from === i).length;
      pathList.append(h("li", { class: `path-pt ${role}${info ? "" : " missing"}` },
        h("span", { class: "pt-role" }, role),
        h("span", { class: "pt-desc", title: id }, info ? H.describePoint(id, cfg) : `missing point ${id}`),
        splits ? h("span", { class: "chip chip-split", tip: "A split starts here" }, icon("split", 12), splits) : null,
        h("button", { class: "icon-btn", tip: "Remove this point", "aria-label": `Remove point ${i + 1}`, onClick: () => upd((x) => {
          const y = H.removeTrunkPoint(x, i);
          x.path = y.path; x.branches = y.branches;
        }, "tendon:path") }, icon("close", 12))));
    });
    const branchList = h("div", { class: "branch-list" });
    (t.branches || []).forEach((b, bi) => {
      const ends = (b.path || []).map((id) => H.pointInfo(id, cfg)).filter(Boolean);
      const last = ends[ends.length - 1];
      branchList.append(h("div", { class: "branch-item" },
        icon("split", 14),
        h("span", { class: "pt-desc" }, `Split ${bi + 1}: from point ${b.from + 1} via ${b.path.length} point${b.path.length === 1 ? "" : "s"}${last ? ` to segment ${last.segment}` : ""}`),
        h("button", { class: "icon-btn", tip: "Remove this split", "aria-label": `Remove split ${bi + 1}`, onClick: () => upd((x) => { x.branches.splice(bi, 1); }, "tendon:branch") }, icon("close", 12))));
    });
    const iss = this._issuesFor(t.name);

    const detail = h("div", { class: "tendon-detail" },
      dl,
      h("div", { class: "detail-row name-row" }, color, h("div", { class: "grow" }, nameIn, nameErr)),
      iss.length ? h("ul", { class: "issue-list" }, iss.map((i) => h("li", { class: `issue ${i.level}` }, i.message))) : null,
      h("label", { class: "field", tip: "Tendons with the same group move together with one group slider, and experiments compare configurations group by group." }, h("span", { class: "field-label" }, "Group"), group),
      h("div", { class: "field" }, h("span", { class: "field-label", tip: ACT_INFO[type] }, "Actuator"), actSeg),
      h("p", { class: "hint" }, ACT_INFO[type]),
      actFields.childElementCount ? actFields : null,
      h("details", { class: "fold" }, h("summary", {}, "Passive properties"), passiveFields),
      h("details", { class: "fold path-fold", open: this._pathOpen ?? false, onToggle: (e) => { this._pathOpen = e.target.open; } },
        h("summary", {}, "Path", h("span", { class: "unit" }, `${t.path.length} point${t.path.length === 1 ? "" : "s"}${t.branches?.length ? `, ${t.branches.length} split${t.branches.length === 1 ? "" : "s"}` : ""}`)),
        t.path.length ? pathList : h("p", { class: "hint" }, "Click a hole in the slice view to set the start point."),
        branchList.childElementCount ? branchList : null,
        t.path.length ? h("div", { class: "path-actions" }, h("button", { class: "btn btn-ghost btn-small", onClick: () => upd((x) => { x.path = []; x.branches = []; }, "tendon:clearpath") }, icon("trash", 13), "Clear path")) : null));
    this.detailEl.append(detail);
    if (focusKey) this.detailEl.querySelector(`[data-field="${focusKey}"]`)?.focus();
  }
}

export { isActuated };

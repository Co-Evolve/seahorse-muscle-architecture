// Experiments tab: build an activation protocol per tendon group, run it headless (no rendering)
// on the current configuration (A) and optionally on a second one (B), overlay the results,
// summarise the difference in words, export every sample as CSV, and a live side-by-side view.
//
// Protocols drive tendons by GROUP name, so A and B can have different tendons as long as
// they share group names.

import { h, clear, icon, segmented, toast, downloadText, pickFile, safeFileName, fmt, parseNum } from "./ui.js";
import { LineChart, BarChart, SERIES } from "./plots.js";
import { actuatedGroups, isActuated } from "./config_tools.js";
import { library } from "./store.js";

const NMM = 1000, MM = 1000;
const PATTERNS = [
  { value: "off", label: "Off", tip: "This group stays relaxed (0 %)." },
  { value: "hold", label: "Hold", tip: "Constant activation from the start." },
  { value: "step", label: "Step", tip: "Switches from 0 to the amplitude at the start time." },
  { value: "ramp", label: "Ramp", tip: "Rises evenly from 0 to the amplitude over the ramp time, then holds." },
  { value: "sine", label: "Sine", tip: "Smoothly goes up and down between 0 and the amplitude, with the given period." },
];
const RUN_COLORS = { A: SERIES[0], B: SERIES[1] };

/** Activation (0..1) of one group pattern at time t. */
export function patternValue(p, t) {
  if (!p) return 0;
  const amp = Math.max(0, Math.min(1, p.amplitude ?? 1));
  const t0 = p.start ?? 0;
  switch (p.pattern) {
    case "hold": return amp;
    case "step": return t >= t0 ? amp : 0;
    case "ramp": { const T = Math.max(1e-6, p.ramp ?? 1); return amp * Math.max(0, Math.min(1, (t - t0) / T)); }
    case "sine": { const T = Math.max(1e-3, p.period ?? 2); return t < t0 ? 0 : amp * 0.5 * (1 - Math.cos((2 * Math.PI * (t - t0)) / T)); }
    default: return 0;
  }
}

function defaultGroupProtocol(i) {
  return i === 0 ? { pattern: "ramp", amplitude: 1, start: 0.25, ramp: 2, period: 2 } : { pattern: "off", amplitude: 1, start: 0.25, ramp: 2, period: 2 };
}

export class ExperimentsView {
  constructor(el, app) {
    this.el = el;
    this.app = app;
    this.protocol = { duration: 4, sampleMs: 20, groups: {} };
    this.mode = "single";
    this.B = null;        // {config, source}
    this.runs = [];       // [{key, label, config, samples}]
    this.running = false;
    this.axis = "pitch";
    this.driver = null;
    this.live = null;
    this._build();
  }

  // ------------------------------------------------------------------ layout
  _build() {
    clear(this.el);
    this.left = h("div", { class: "exp-left" });
    this.right = h("div", { class: "exp-right" });
    this.el.append(this.left, this.right);

    // protocol card
    this.protoCard = h("section", { class: "card" });
    this.left.append(this.protoCard);
    // compare card
    this.cmpCard = h("section", { class: "card" });
    this.left.append(this.cmpCard);

    // results area with sub tabs
    this.resTabs = segmented([{ value: "results", label: "Results" }, { value: "live", label: "Live side by side" }], "results",
      (v) => this._showRes(v), { label: "Result view" });
    this.resHead = h("div", { class: "res-head" }, this.resTabs, h("div", { class: "spacer" }));
    this.resultsPane = h("div", { class: "res-pane" });
    this.livePane = h("div", { class: "res-pane", hidden: true });
    this.right.append(this.resHead, this.resultsPane, this.livePane);
    this._buildResults();
    this._buildLive();
  }

  /** Called when the tab becomes visible or the current config changed. */
  refresh() {
    const cfg = this.app.getConfig();
    const groups = this._groups();
    groups.forEach((g, i) => { if (!this.protocol.groups[g]) this.protocol.groups[g] = defaultGroupProtocol(i); });
    this._renderProtocol();
    this._renderCompare();
    this._renderLiveControls();
    this.cfgName = cfg.name;
  }
  onHide() { this._stopLive(); }

  _groups() {
    const g = actuatedGroups(this.app.getConfig());
    if (this.mode === "compare" && this.B) for (const x of actuatedGroups(this.B.config)) if (!g.includes(x)) g.push(x);
    return g;
  }

  // ------------------------------------------------------------------ protocol
  _renderProtocol() {
    const card = this.protoCard;
    clear(card);
    const groups = this._groups();
    const cfg = this.app.getConfig();
    card.append(h("header", { class: "card-head" },
      h("h2", { class: "card-title" }, "Protocol"),
      h("span", { class: "card-sub" }, "How hard each group of tendons pulls over time")));
    if (!groups.length) {
      card.append(h("div", { class: "empty-state" }, h("div", { class: "empty-title" }, "No driven tendons"),
        h("p", {}, "Add a tendon with a motor or position actuator in the Design view, then come back here.")));
      return;
    }
    const P = this.protocol;
    const durIn = h("input", { type: "text", inputmode: "decimal", class: "num", value: P.duration, "aria-label": "Duration in seconds" });
    durIn.addEventListener("change", () => { P.duration = Math.max(0.1, Math.min(120, parseNum(durIn.value) || 4)); durIn.value = P.duration; this._preview(); });
    const smpIn = h("input", { type: "text", inputmode: "numeric", class: "num", value: P.sampleMs, "aria-label": "Sample interval in milliseconds" });
    smpIn.addEventListener("change", () => { P.sampleMs = Math.max(2, Math.min(500, Math.round(parseNum(smpIn.value)) || 20)); smpIn.value = P.sampleMs; });
    card.append(h("div", { class: "field-grid two" },
      h("label", { class: "field", tip: "Total simulated time of the experiment." }, h("span", { class: "field-label" }, "Duration", h("span", { class: "unit" }, "s")), durIn),
      h("label", { class: "field", tip: "How often a measurement is stored (one CSV row per sample)." }, h("span", { class: "field-label" }, "Sample every", h("span", { class: "unit" }, "ms")), smpIn)));

    const list = h("div", { class: "group-protos" });
    for (const g of groups) {
      const p = P.groups[g];
      const tendonsA = cfg.tendons.filter((t) => isActuated(t) && (t.group || "(no group)") === g);
      const tendonsB = this.mode === "compare" && this.B ? this.B.config.tendons.filter((t) => isActuated(t) && (t.group || "(no group)") === g) : null;
      const swatches = h("span", { class: "swatches" }, tendonsA.map((t) => h("span", { class: "swatch small", style: { background: t.color }, title: t.name })));
      const missing = [];
      if (!tendonsA.length) missing.push("A");
      if (tendonsB && !tendonsB.length) missing.push("B");
      const fields = h("div", { class: "proto-fields" });
      const renderFields = () => {
        clear(fields);
        if (p.pattern === "off") return;
        const amp = h("input", { type: "range", min: 0, max: 100, step: 1, value: Math.round(p.amplitude * 100), "aria-label": `${g} amplitude` });
        const ampOut = h("output", { class: "num-out" }, `${Math.round(p.amplitude * 100)} %`);
        amp.addEventListener("input", () => { p.amplitude = Number(amp.value) / 100; ampOut.textContent = `${amp.value} %`; this._preview(); });
        fields.append(h("label", { class: "field slider-field" }, h("span", { class: "field-label" }, "Amplitude"), h("span", { class: "slider-line" }, amp, ampOut)));
        const num = (label, key, unit, tip) => {
          const inp = h("input", { type: "text", inputmode: "decimal", class: "num", value: p[key], "aria-label": `${g} ${label}` });
          inp.addEventListener("change", () => { const v = parseNum(inp.value); if (Number.isFinite(v) && v >= 0) { p[key] = v; this._preview(); } else inp.value = p[key]; });
          return h("label", { class: "field", tip }, h("span", { class: "field-label" }, label, h("span", { class: "unit" }, unit)), inp);
        };
        const grid = h("div", { class: "field-grid three" });
        if (p.pattern !== "hold") grid.append(num("Start", "start", "s", "Time at which this group starts to pull."));
        if (p.pattern === "ramp") grid.append(num("Ramp time", "ramp", "s", "Time to rise from 0 to the amplitude."));
        if (p.pattern === "sine") grid.append(num("Period", "period", "s", "Time for one full up-and-down cycle."));
        if (grid.childElementCount) fields.append(grid);
      };
      const pat = h("select", { class: "select select-small", "aria-label": `${g} pattern` }, PATTERNS.map((o) => h("option", { value: o.value, selected: o.value === p.pattern }, o.label)));
      pat.addEventListener("change", () => { p.pattern = pat.value; renderFields(); this._preview(); });
      renderFields();
      list.append(h("div", { class: "group-proto" },
        h("div", { class: "gp-head" }, h("span", { class: "gp-name" }, g), swatches,
          missing.length ? h("span", { class: "chip chip-warn", tip: `Configuration ${missing.join(" and ")} has no driven tendon in this group, so this row does nothing there.` }, `not in ${missing.join(", ")}`) : null,
          h("span", { class: "spacer" }), pat),
        fields));
    }
    card.append(list);
    const prevBox = h("div", { class: "proto-preview" });
    card.append(prevBox);
    this.previewChart = new LineChart(prevBox, { title: "Activation over time", height: 120, margin: { top: 8, right: 8, bottom: 26, left: 36 },
      xLabel: "s", yLabel: "%", yDomain: [0, 100], valueFmt: (v) => `${v.toFixed(0)} %`, hoverXFmt: (v) => `${v.toFixed(2)} s` });
    this._preview();
  }
  _preview() {
    if (!this.previewChart) return;
    const P = this.protocol;
    const groups = this._groups();
    const N = 160;
    const cfg = this.app.getConfig();
    const colorOf = (g, i) => {
      const t = [...cfg.tendons, ...(this.B?.config.tendons || [])].find((x) => isActuated(x) && (x.group || "(no group)") === g);
      return t?.color || SERIES[i % 8];
    };
    this.previewChart.setSeries(groups.filter((g) => P.groups[g]?.pattern !== "off").map((g, i) => ({
      id: g, label: g, color: colorOf(g, i),
      points: Array.from({ length: N + 1 }, (_, k) => { const t = (k / N) * P.duration; return [t, patternValue(P.groups[g], t) * 100]; }),
    })));
  }

  // ------------------------------------------------------------------ compare picker
  _renderCompare() {
    const card = this.cmpCard;
    clear(card);
    card.append(h("header", { class: "card-head" }, h("h2", { class: "card-title" }, "Run"),
      h("span", { class: "card-sub" }, "Test the current design alone, or against a second one")));
    const modeSeg = segmented([
      { value: "single", label: "Current only" },
      { value: "compare", label: "Compare A vs B" },
    ], this.mode, (v) => { this.mode = v; this.refresh(); }, { label: "Run mode" });
    card.append(modeSeg);
    const cfg = this.app.getConfig();
    const rowA = h("div", { class: "cfg-slot" }, h("span", { class: "slot-tag a" }, "A"), h("div", { class: "slot-main" },
      h("div", { class: "slot-name" }, cfg.name || "Untitled"), h("div", { class: "slot-meta" }, `current design, ${cfg.tendons.length} tendon${cfg.tendons.length === 1 ? "" : "s"}`)));
    card.append(rowA);
    if (this.mode === "compare") {
      const bName = this.B ? this.B.config.name : "Pick a configuration";
      const bMeta = this.B ? `${this.B.source}, ${this.B.config.tendons.length} tendons` : "from presets, your library or a file";
      const pick = h("button", { class: "btn btn-secondary btn-small" }, icon("open", 14), this.B ? "Change" : "Choose");
      this.app.attachMenu(pick, async () => {
        const items = [];
        const presets = await this.app.listPresets();
        items.push({ heading: "Presets" });
        if (!presets.length) items.push({ label: "No presets found", disabled: true });
        for (const p of presets) items.push({ label: p.name, hint: p.description, onClick: async () => this._setB(await this.app.fetchPreset(p), "preset") });
        items.push({ separator: true }, { heading: "My library" });
        const lib = library.list();
        if (!lib.length) items.push({ label: "Empty: use Save → Save to my library", disabled: true });
        for (const l of lib) items.push({ label: l.name, hint: new Date(l.saved).toLocaleString(), onClick: () => this._setB(l.config, "my library") });
        items.push({ separator: true }, { label: "Upload a JSON file…", onClick: async () => {
          const f = await pickFile();
          if (!f) return;
          try { this._setB(this.app.parseConfigText(f.text), `file ${f.name}`); } catch (e) { toast(e.message, "error", 6000); }
        } });
        items.push({ label: "Copy of the current design", hint: "then edit A and compare", onClick: () => this._setB(JSON.parse(JSON.stringify(this.app.getConfig())), "snapshot of A") });
        return items;
      });
      card.append(h("div", { class: `cfg-slot${this.B ? "" : " empty"}` }, h("span", { class: "slot-tag b" }, "B"),
        h("div", { class: "slot-main" }, h("div", { class: "slot-name" }, bName), h("div", { class: "slot-meta" }, bMeta)), pick));
    }
    const issuesA = this.app.validate(cfg).filter((i) => i.level === "error");
    const issuesB = this.mode === "compare" && this.B ? this.app.validate(this.B.config).filter((i) => i.level === "error") : [];
    const blocked = issuesA.length || issuesB.length || (this.mode === "compare" && !this.B) || !this._groups().length;
    this.runBtn = h("button", { class: "btn btn-primary btn-wide", disabled: !!blocked || this.running, onClick: () => this.run() }, icon("play", 14), this.mode === "compare" ? "Run A and B" : "Run experiment");
    this.cancelBtn = h("button", { class: "btn btn-ghost", hidden: !this.running, onClick: () => { this._cancel = true; } }, "Cancel");
    this.progress = h("div", { class: "progress", hidden: !this.running }, h("div", { class: "progress-bar" }));
    this.progressText = h("div", { class: "hint run-status" }, this.running ? "" : (this.status || ""));
    card.append(h("div", { class: "run-row" }, this.runBtn, this.cancelBtn), this.progress, this.progressText);
    if (issuesA.length) card.append(h("p", { class: "issue error" }, `A cannot run yet: ${issuesA[0].message}`));
    if (issuesB.length) card.append(h("p", { class: "issue error" }, `B cannot run: ${issuesB[0].message}`));
    card.append(h("p", { class: "hint" }, "Runs as fast as possible without drawing. Both runs start from the straight rest pose with the same protocol."));
  }
  _setB(config, source) {
    if (!config) return;
    try { config = this.app.normalise(config); } catch (e) { toast(e.message, "error", 6000); return; }
    this.B = { config, source };
    this.refresh();
    if (this.live?.running) this._startLive();
  }

  // ------------------------------------------------------------------ headless runs
  async run() {
    if (this.running) return;
    this.running = true; this._cancel = false;
    this._renderCompare();
    const jobs = [{ key: "A", config: this.app.getConfig() }];
    if (this.mode === "compare" && this.B) jobs.push({ key: "B", config: this.B.config });
    const P = JSON.parse(JSON.stringify(this.protocol));
    const runs = [];
    const bar = this.progress.firstChild;
    const tStart = performance.now();
    this.status = "";
    try {
      for (let j = 0; j < jobs.length; j++) {
        const job = jobs[j];
        this.progressText.textContent = `Building ${job.key}: ${job.config.name}…`;
        await new Promise((r) => setTimeout(r, 0));
        const sim = await this.app.createSim(job.config);
        try {
          const run = await this._runOne(sim, job, P, (f) => {
            bar.style.width = `${(((j + f) / jobs.length) * 100).toFixed(1)}%`;
            this.progressText.textContent = `Running ${job.key}: ${Math.round(f * 100)} %`;
          });
          if (!run) { toast("Experiment cancelled", "info"); return; }
          runs.push(run);
        } finally { sim.dispose(); }
      }
      this.runs = runs;
      this.lastProtocol = P;
      this._renderResults();
      this.status = `Done: ${runs.map((r) => `${r.key} ${r.samples.length} samples`).join(", ")}, in ${((performance.now() - tStart) / 1000).toFixed(1)} s.`;
    } catch (e) {
      console.error(e);
      toast(`The experiment could not run: ${e.message}`, "error", 8000);
      this.progressText.textContent = "";
    } finally {
      this.running = false;
      this._renderCompare();
    }
  }

  async _runOne(sim, job, P, onProgress) {
    const dt = P.sampleMs / 1000;
    const nSamples = Math.max(1, Math.round(P.duration / dt));
    const groupsOf = {};
    for (const t of sim.tendons) if (t.actuatorId !== null) (groupsOf[t.group || "(no group)"] ||= []).push(t.name);
    const zero = {};
    for (const t of sim.tendons) zero[t.name] = 0;
    sim.setActivations(zero);
    sim.reset();
    const samples = [];
    const record = (t) => {
      const m = sim.measure();
      const act = {};
      for (const g of Object.keys(P.groups)) act[g] = patternValue(P.groups[g], t);
      samples.push({ t: m.time, act, m });
    };
    const apply = (t) => {
      const a = {};
      for (const [g, names] of Object.entries(groupsOf)) { const v = patternValue(P.groups[g], t); for (const n of names) a[n] = v; }
      sim.setActivations(a);
    };
    apply(0);
    record(0);
    let last = performance.now();
    for (let k = 1; k <= nSamples; k++) {
      const tPrev = (k - 1) * dt;
      apply(tPrev + dt * 0.5);
      sim.advance(dt);
      const tNow = k * dt;
      apply(tNow);
      record(tNow);
      if (performance.now() - last > 30) {
        onProgress(k / nSamples);
        await new Promise((r) => setTimeout(r, 0));
        last = performance.now();
        if (this._cancel) return null;
      }
    }
    onProgress(1);
    return { key: job.key, label: `${job.key}: ${job.config.name}`, config: job.config, tendons: sim.tendons.map((t) => ({ name: t.name, group: t.group, color: t.color, actuated: t.actuatorId !== null })), samples };
  }

  // ------------------------------------------------------------------ results
  _buildResults() {
    const pane = this.resultsPane;
    this.summaryCard = h("section", { class: "card summary-card" });
    this.chartGrid = h("div", { class: "result-grid" });
    pane.append(this.summaryCard, this.chartGrid);
    const box = () => { const b = h("div", { class: "card chart-card" }); this.chartGrid.append(b); return b; };
    const H = 190;
    this.cTime = new LineChart(box(), { title: "Tip ventral angle over time", height: H, xLabel: "time (s)", yLabel: "°", includeZero: true, minYSpan: 2,
      valueFmt: (v) => `${v.toFixed(1)}°`, hoverXFmt: (v) => `${v.toFixed(2)} s` });
    this.cAct = new LineChart(box(), { title: "Tip ventral angle vs activation", height: H, xLabel: "activation (%)", yLabel: "°", includeZero: true, minYSpan: 2,
      xDomain: [0, 100], hover: "nearest", valueFmt: (v) => `${v.toFixed(1)}°`, hoverXFmt: (v) => `${v.toFixed(0)} %`, xName: "activation" });
    this.cCurv = new BarChart(box(), { title: "Final curvature profile", subtitle: "Ventral bend per joint at the end of the run", height: H, xLabel: "segment", yLabel: "°",
      categoryName: "Segment ", valueFmt: (v) => `${v.toFixed(1)}°`, minYSpan: 2 });
    this.cFE = new LineChart(box(), { title: "Force vs excursion", subtitle: "Each driven tendon; excursion = shortening", height: H, xLabel: "excursion (mm)", yLabel: "N",
      hover: "nearest", includeZero: true, xIncludeZero: true, valueFmt: (v, se) => `${v.toFixed(2)} N`, hoverXFmt: (v) => `${v.toFixed(2)} mm` });
    this.cTq = new BarChart(box(), { title: "Joint torque at peak activation", subtitle: "Total from all tendons", height: H, xLabel: "segment", yLabel: "N·mm",
      categoryName: "Joint at segment ", valueFmt: (v) => `${v.toFixed(1)} N·mm`, minYSpan: 0.2 });
    this.cTqT = new LineChart(box(), { title: "Peak joint torque over time", subtitle: "Largest torque on any joint", height: H, xLabel: "time (s)", yLabel: "N·mm",
      includeZero: true, minYSpan: 0.2, valueFmt: (v) => `${v.toFixed(2)} N·mm`, hoverXFmt: (v) => `${v.toFixed(2)} s` });
    this.axisSeg = segmented(["pitch", "roll", "yaw"].map((a) => ({ value: a, label: a[0].toUpperCase() + a.slice(1) })), this.axis,
      (v) => { this.axis = v; this._renderResults(); }, { label: "Torque axis", small: true });
    this.driverSel = h("select", { class: "select select-small", "aria-label": "Activation of group" });
    this.driverSel.addEventListener("change", () => { this.driver = this.driverSel.value; this._renderResults(); });
    this.csvBtn = h("button", { class: "btn btn-secondary btn-small", disabled: true, onClick: () => this.exportCsv() }, icon("download", 14), "Download CSV");
    this.resTools = [h("span", { class: "tool-label" }, "Torque axis"), this.axisSeg, h("span", { class: "tool-label", tip: "x axis of “tip angle vs activation”; also defines the moment of peak activation" }, "Activation of"), this.driverSel, this.csvBtn];
    this.resHead.append(...this.resTools);
    this._renderResults();
  }

  _renderResults() {
    const runs = this.runs;
    const card = this.summaryCard;
    clear(card);
    this.csvBtn.disabled = !runs.length;
    this.chartGrid.hidden = !runs.length;
    for (const el of this.resTools || []) el.hidden = !runs.length || this.resView === "live";
    const groups = runs.length ? Object.keys(runs[0].samples[0].act) : [];
    clear(this.driverSel);
    if (!this.driver || !groups.includes(this.driver)) {
      const P = this.lastProtocol;
      this.driver = groups.slice().sort((a, b) => (P?.groups[b]?.pattern !== "off" ? P?.groups[b]?.amplitude || 0 : -1) - (P?.groups[a]?.pattern !== "off" ? P?.groups[a]?.amplitude || 0 : -1))[0] || null;
    }
    for (const g of groups) this.driverSel.append(h("option", { value: g, selected: g === this.driver }, g));
    if (!runs.length) {
      card.append(h("div", { class: "empty-state big" }, h("div", { class: "empty-title" }, "No results yet"),
        h("p", {}, "Set up a protocol on the left and press Run. The charts below then show how the tail bent, which forces the tendons needed and which torques reached each joint.")));
      for (const c of [this.cTime, this.cAct, this.cFE, this.cTqT]) c.setSeries([]);
      for (const c of [this.cCurv, this.cTq]) c.setData({ categories: [], series: [] });
      return;
    }
    const ax = this.axis;
    const S = (r) => r.samples;
    const color = (r) => RUN_COLORS[r.key];
    const lbl = (r) => (runs.length > 1 ? `${r.key}: ${r.config.name}` : r.config.name);
    this.cTime.setSeries(runs.map((r) => ({ id: r.key, label: lbl(r), color: color(r), points: S(r).map((s) => [s.t, s.m.tip.ventral]) })));
    this.cAct.setSeries(runs.map((r) => ({ id: r.key, label: lbl(r), color: color(r), points: S(r).map((s) => [(s.act[this.driver] ?? 0) * 100, s.m.tip.ventral]) })));
    const last = (r) => S(r)[S(r).length - 1].m;
    const cats = last(runs[0]).segments.map((s) => String(s.index));
    this.cCurv.setData({ categories: cats, series: runs.map((r) => ({ id: r.key, label: lbl(r), color: color(r), values: last(r).segments.map((s) => s.ventral) })) });
    const fe = [];
    for (const r of runs) {
      r.tendons.forEach((t, i) => {
        if (!t.actuated) return;
        fe.push({ id: `${r.key}-${t.name}`, label: `${r.key} · ${t.name}`, legendLabel: lbl(r), color: color(r),
          points: S(r).map((s) => [s.m.tendons[i].excursion * MM, s.m.tendons[i].force]) });
      });
    }
    this.cFE.setSeries(fe);
    const peak = runs.map((r) => this._peakSample(r));
    this.cTq.opts.title = `Joint ${ax} torque at peak activation`;
    const th = this.cTq.root.querySelector(".chart-title"); if (th) th.textContent = `Joint ${ax} torque at peak activation`;
    this.cTq.setData({ categories: cats, series: runs.map((r, i) => ({ id: r.key, label: lbl(r), color: color(r), values: peak[i].m.segments.map((s) => (s.torque?.[ax] ?? 0) * NMM) })) });
    const t2 = this.cTqT.root.querySelector(".chart-title"); if (t2) t2.textContent = `Peak ${ax} torque over time`;
    this.cTqT.setSeries(runs.map((r) => ({ id: r.key, label: lbl(r), color: color(r),
      points: S(r).map((s) => [s.t, s.m.segments.reduce((a, x) => (Math.abs(x.torque?.[ax] ?? 0) > Math.abs(a) ? x.torque[ax] : a), 0) * NMM]) })));
    this._renderSummary(runs, peak);
  }

  /** Sample with the highest driving activation (the last one, so the tail had time to settle). */
  _peakSample(r) {
    let best = r.samples[0], bv = -1;
    for (const s of r.samples) {
      const v = this.driver ? (s.act[this.driver] ?? 0) : 0;
      if (v >= bv - 1e-9) { bv = v; best = s; }
    }
    return best;
  }

  _metrics(r, s) {
    const m = s.m;
    const ax = this.axis;
    const driven = r.tendons.map((t, i) => (t.actuated ? m.tendons[i] : null)).filter(Boolean);
    const exc = driven.reduce((a, t) => a + Math.max(0, t.excursion), 0) * MM;
    const force = driven.reduce((a, t) => a + t.force, 0);
    let peakTq = 0, peakSeg = null, sumTq = 0;
    for (const sg of m.segments) {
      const v = (sg.torque?.[ax] ?? 0) * NMM;
      sumTq += Math.abs(v);
      if (Math.abs(v) > Math.abs(peakTq)) { peakTq = v; peakSeg = sg.index; }
    }
    const work = r.tendons.reduce((a, t, i) => a + (t.actuated ? m.tendons[i].work : 0), 0) * 1000;
    return { act: this.driver ? (s.act[this.driver] ?? 0) * 100 : 0, ventral: m.tip.ventral, lateral: m.tip.lateral, twist: m.tip.twist, distance: m.tip.distance * MM,
      exc, force, peakTq, peakSeg, perN: force > 1e-9 ? sumTq / force : NaN, work, time: s.t };
  }

  _renderSummary(runs, peak) {
    const card = this.summaryCard;
    const ms = runs.map((r, i) => this._metrics(r, peak[i]));
    const ax = this.axis;
    const A = ms[0], B = ms[1];
    const lines = [];
    const actTxt = A.act >= 99.5 ? "full activation" : `${A.act.toFixed(0)} % activation of “${this.driver}”`;
    if (B) {
      const dv = B.ventral - A.ventral;
      lines.push(Math.abs(dv) < 0.5
        ? `At ${actTxt}, A and B bend about equally far ventrally (${fmt(A.ventral)}° vs ${fmt(B.ventral)}°).`
        : `At ${actTxt}, B reaches ${fmt(Math.abs(dv))}° ${dv > 0 ? "more" : "less"} ventral bending than A (${fmt(B.ventral)}° vs ${fmt(A.ventral)}°).`);
      if (A.exc > 0.05 || B.exc > 0.05) {
        const pc = A.exc > 1e-6 ? ((B.exc - A.exc) / A.exc) * 100 : NaN;
        lines.push(Number.isFinite(pc) && Math.abs(pc) >= 3
          ? `B needs ${fmt(Math.abs(pc), 0)} % ${pc < 0 ? "less" : "more"} tendon excursion (${fmt(B.exc, 1)} mm vs ${fmt(A.exc, 1)} mm in total).`
          : `Both need a similar total tendon excursion (${fmt(A.exc, 1)} mm vs ${fmt(B.exc, 1)} mm).`);
      }
      const lat = Math.max(Math.abs(A.lateral), Math.abs(B.lateral));
      if (lat >= 1) lines.push(`Sideways bending: A ${fmt(A.lateral)}°, B ${fmt(B.lateral)}° (positive = dextral).`);
      lines.push(`Largest ${ax} torque: A ${fmt(A.peakTq, 2)} N·mm at segment ${A.peakSeg ?? "–"}, B ${fmt(B.peakTq, 2)} N·mm at segment ${B.peakSeg ?? "–"}.`);
      if (Number.isFinite(A.perN) && Number.isFinite(B.perN)) {
        lines.push(`Per newton of tendon force, B produces ${fmt(B.perN, 1)} N·mm of ${ax} torque summed over all joints, A ${fmt(A.perN, 1)} N·mm (${B.perN >= A.perN ? "B uses force more effectively" : "A uses force more effectively"}).`);
      }
    } else {
      lines.push(`At ${actTxt}, the tip bends ${fmt(A.ventral)}° ventrally and ${fmt(A.lateral)}° sideways (positive = dextral), and moves ${fmt(A.distance, 1)} mm.`);
      lines.push(`The driven tendons shorten ${fmt(A.exc, 1)} mm in total and pull with ${fmt(A.force, 2)} N in total.`);
      lines.push(`Largest ${ax} torque: ${fmt(A.peakTq, 2)} N·mm at the joint of segment ${A.peakSeg ?? "–"}. Per newton of tendon force that is ${fmt(A.perN, 1)} N·mm summed over all joints.`);
    }
    card.append(h("header", { class: "card-head" }, h("h2", { class: "card-title" }, "Summary"),
      h("span", { class: "card-sub" }, `Measured at the moment of peak activation (t = ${fmt(A.time, 2)} s)`)));
    card.append(h("ul", { class: "summary-list" }, lines.map((l) => h("li", {}, l))));
    // metric table
    const rows = [
      ["Tip ventral angle", "°", (m) => fmt(m.ventral, 1)],
      ["Tip lateral angle (dextral +)", "°", (m) => fmt(m.lateral, 1)],
      ["Tip displacement", "mm", (m) => fmt(m.distance, 1)],
      ["Total tendon excursion", "mm", (m) => fmt(m.exc, 2)],
      ["Total tendon force", "N", (m) => fmt(m.force, 2)],
      [`Peak ${ax} torque`, "N·mm", (m) => `${fmt(m.peakTq, 2)}${m.peakSeg ? ` (seg ${m.peakSeg})` : ""}`],
      [`${ax[0].toUpperCase() + ax.slice(1)} torque per N of force`, "N·mm/N", (m) => fmt(m.perN, 1)],
      ["Tendon work", "mJ", (m) => fmt(m.work, 2)],
    ];
    const table = h("table", { class: "metric-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Measure"), h("th", {}, "Unit"), ...runs.map((r) => h("th", {}, h("span", { class: "key key-line", style: { borderColor: RUN_COLORS[r.key] } }), r.key)), B ? h("th", {}, "B − A") : null)),
      h("tbody", {}, rows.map(([name, unit, f]) => h("tr", {}, h("td", {}, name), h("td", { class: "unit" }, unit), ms.map((m) => h("td", { class: "num" }, f(m))),
        B ? h("td", { class: "num delta" }, (() => { const a = Number(f(A).split(" ")[0]), b = Number(f(B).split(" ")[0]); return Number.isFinite(a) && Number.isFinite(b) ? `${b - a >= 0 ? "+" : "−"}${fmt(Math.abs(b - a), 2)}` : "–"; })()) : null))));
    card.append(table);
  }

  // ------------------------------------------------------------------ CSV
  exportCsv() {
    if (!this.runs.length) return;
    const runs = this.runs;
    const groups = [...new Set(runs.flatMap((r) => Object.keys(r.samples[0].act)))];
    const nSeg = Math.max(...runs.map((r) => r.samples[0].m.segments.length));
    const segIdx = runs[0].samples[0].m.segments.map((s) => s.index);
    const tendonNames = [...new Set(runs.flatMap((r) => r.tendons.map((t) => t.name)))];
    const AX = ["pitch", "roll", "yaw"];
    const cols = ["run", "config", "time_s", ...groups.map((g) => `act_${g}`),
      "tip_ventral_deg", "tip_lateral_deg", "tip_bend_deg", "tip_twist_deg", "tip_dx_mm", "tip_dy_mm", "tip_dz_mm", "tip_distance_mm"];
    for (const i of segIdx) {
      cols.push(`seg${i}_ventral_deg`, `seg${i}_lateral_deg`, `seg${i}_twist_deg`);
      for (const a of AX) cols.push(`seg${i}_torque_${a}_Nmm`);
      for (const a of AX) cols.push(`seg${i}_passive_torque_${a}_Nmm`);
      for (const a of AX) cols.push(`seg${i}_limit_torque_${a}_Nmm`);
    }
    for (const n of tendonNames) {
      cols.push(`${n}_activation`, `${n}_force_N`, `${n}_length_mm`, `${n}_excursion_mm`, `${n}_strain`, `${n}_work_mJ`);
      for (const i of segIdx) for (const a of AX) cols.push(`${n}_seg${i}_torque_${a}_Nmm`);
      for (const i of segIdx) for (const a of AX) cols.push(`${n}_seg${i}_moment_arm_${a}_mm`);
    }
    const esc = (v) => { const s = String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const num = (v, d = 6) => (v === undefined || v === null || !Number.isFinite(v) ? "" : Number(v.toPrecision(d)));
    const lines = [cols.join(",")];
    for (const r of runs) {
      const tIndex = new Map(r.samples[0].m.tendons.map((t, i) => [t.name, i]));
      for (const s of r.samples) {
        const m = s.m;
        const row = [r.key, r.config.name, num(s.t), ...groups.map((g) => num(s.act[g] ?? 0)),
          num(m.tip.ventral), num(m.tip.lateral), num(m.tip.bend), num(m.tip.twist),
          num(m.tip.displacement[0] * MM), num(m.tip.displacement[1] * MM), num(m.tip.displacement[2] * MM), num(m.tip.distance * MM)];
        for (const i of segIdx) {
          const sg = m.segments.find((x) => x.index === i) || {};
          row.push(num(sg.ventral), num(sg.lateral), num(sg.twist));
          for (const key of ["torque", "passive_torque", "limit_torque"]) for (const a of AX) row.push(num(sg[key] ? sg[key][a] * NMM : undefined));
        }
        for (const n of tendonNames) {
          const ti = tIndex.get(n);
          const t = ti === undefined ? null : m.tendons[ti];
          if (!t) { row.push(...Array(6 + nSeg * 6).fill("")); continue; }
          row.push(num(t.activation), num(t.force), num(t.length * MM), num(t.excursion * MM), num(t.strain), num(t.work * 1000));
          for (const i of segIdx) { const q = t.torque?.find((x) => x.segment === i); for (const a of AX) row.push(num(q ? q[a] * NMM : undefined)); }
          for (const i of segIdx) { const q = t.moment_arms?.find((x) => x.segment === i); for (const a of AX) row.push(num(q ? q[a] * MM : undefined)); }
        }
        lines.push(row.map(esc).join(","));
      }
    }
    const name = runs.length > 1 ? `experiment_${runs[0].config.name}_vs_${runs[1].config.name}` : `experiment_${runs[0].config.name}`;
    downloadText(safeFileName(name, "csv"), lines.join("\n") + "\n", "text/csv");
    toast(`Downloaded ${lines.length - 1} rows`, "success");
  }

  // ------------------------------------------------------------------ live side by side
  _showRes(v) {
    this.resultsPane.hidden = v !== "results";
    this.livePane.hidden = v !== "live";
    this.resView = v;
    for (const el of this.resTools || []) el.hidden = v !== "results" || !this.runs.length;
    if (v === "live") this._startLive(); else this._stopLive();
  }
  _buildLive() {
    const pane = this.livePane;
    this.liveControls = h("section", { class: "card live-controls" });
    this.liveViews = h("div", { class: "live-views" });
    pane.append(this.liveControls, this.liveViews);
    this.live = { running: false, playing: true, sims: [], viewers: [], acts: {} };
  }
  _renderLiveControls() {
    const L = this.live;
    clear(this.liveControls);
    const groups = this._groups();
    const play = h("button", { class: "btn btn-secondary btn-small", onClick: () => { L.playing = !L.playing; this._renderLiveControls(); } }, icon(L.playing ? "pause" : "play", 14), L.playing ? "Pause" : "Play");
    const reset = h("button", { class: "btn btn-ghost btn-small", onClick: () => { for (const s of L.sims) s?.sim?.reset(); } }, icon("reset", 14), "Reset");
    this.liveControls.append(h("div", { class: "live-row" }, play, reset,
      h("span", { class: "hint" }, this.mode === "compare" && this.B ? "Both models get exactly the same group activations." : "Choose “Compare A vs B” and a configuration B on the left to see two tails side by side.")));
    const sliders = h("div", { class: "live-sliders" });
    for (const g of groups) {
      const v = L.acts[g] ?? 0;
      const inp = h("input", { type: "range", min: 0, max: 100, step: 1, value: Math.round(v * 100), "aria-label": `${g} activation` });
      const out = h("output", { class: "num-out" }, `${Math.round(v * 100)} %`);
      inp.addEventListener("input", () => { L.acts[g] = Number(inp.value) / 100; out.textContent = `${inp.value} %`; this._applyLiveActs(); });
      sliders.append(h("label", { class: "field slider-field" }, h("span", { class: "field-label" }, g), h("span", { class: "slider-line" }, inp, out)));
    }
    this.liveControls.append(sliders);
  }
  _applyLiveActs() {
    const L = this.live;
    for (const e of L.sims) {
      if (!e?.sim) continue;
      const a = {};
      for (const t of e.sim.tendons) if (t.actuatorId !== null) a[t.name] = L.acts[t.group || "(no group)"] ?? 0;
      e.sim.setActivations(a);
    }
  }
  async _startLive() {
    this._stopLive();
    const L = this.live;
    L.running = true;
    const token = (L.token = Symbol("live"));
    clear(this.liveViews);
    const jobs = [{ key: "A", config: this.app.getConfig() }];
    if (this.mode === "compare" && this.B) jobs.push({ key: "B", config: this.B.config });
    L.sims = [];
    for (const job of jobs) {
      const view = h("div", { class: "live-view" });
      const readout = h("div", { class: "live-readout" });
      const card = h("div", { class: "card live-card" }, h("div", { class: "live-title" }, h("span", { class: `slot-tag ${job.key.toLowerCase()}` }, job.key), job.config.name), view, readout);
      this.liveViews.append(card);
      try {
        const sim = await this.app.createSim(job.config);
        if (L.token !== token) { sim.dispose(); return; }
        const viewer = this.app.createViewer(view);
        viewer.setSimulation(sim);
        L.sims.push({ key: job.key, sim, viewer, readout });
      } catch (e) {
        view.append(h("div", { class: "issue error" }, e.message));
      }
    }
    this._applyLiveActs();
    let lastT = performance.now();
    let lastRead = 0;
    const loop = (now) => {
      if (L.token !== token) return;
      const dtReal = Math.min(0.05, (now - lastT) / 1000);
      lastT = now;
      for (const e of L.sims) {
        try {
          if (L.playing) e.sim.advance(dtReal);
          e.viewer.update();
        } catch (err) { console.error(err); }
      }
      if (now - lastRead > 150) {
        lastRead = now;
        for (const e of L.sims) {
          const m = e.sim.measure({ torques: false });
          clear(e.readout);
          e.readout.append(
            h("span", { class: "stat" }, h("span", { class: "stat-label" }, "tip ventral"), h("strong", {}, `${fmt(m.tip.ventral)}°`)),
            h("span", { class: "stat" }, h("span", { class: "stat-label" }, "tip lateral"), h("strong", {}, `${fmt(m.tip.lateral)}°`)),
            h("span", { class: "stat" }, h("span", { class: "stat-label" }, "force"), h("strong", {}, `${fmt(m.tendons.reduce((a, t) => a + t.force, 0), 2)} N`)));
        }
      }
      L.raf = requestAnimationFrame(loop);
    };
    L.raf = requestAnimationFrame(loop);
  }
  _stopLive() {
    const L = this.live;
    if (!L) return;
    L.token = null;
    L.running = false;
    cancelAnimationFrame(L.raf);
    for (const e of L.sims) { try { e.viewer.dispose(); } catch { /* ignore */ } try { e.sim.dispose(); } catch { /* ignore */ } }
    L.sims = [];
  }
}

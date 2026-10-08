// Live analysis dock under the 3D view: three tabs of charts fed by Simulation.measure().
//   Bending:  curvature profile (per-joint ventral & lateral angle) + tail shape (side & front view)
//   Torques:  joint torque per segment, stacked per tendon (pitch / roll / yaw) + moment-arm table
//   Over time: rolling tip angle, tendon forces, and the torques at one chosen joint

import { h, clear, segmented } from "./ui.js";
import { LineChart, BarChart, HeatTable, SERIES } from "./plots.js";

export const NMM = 1000; // N·m -> N·mm
export const MM = 1000;  // m -> mm
const AXIS_LABEL = { pitch: "Pitch (ventral / dorsal)", roll: "Roll (sideways)", yaw: "Yaw (twist)" };
const AXIS_SHORT = { pitch: "pitch", roll: "roll", yaw: "yaw" };

export class AnalysisDock {
  constructor(el, { onTab } = {}) {
    this.el = el;
    this.tab = "bending";
    this.axis = "pitch";
    this.joint = null; // segment index for the "over time" torque chart (null = auto)
    this.window = 10;  // seconds of history
    this.buf = [];
    this.sim = null;
    this.last = null;
    this.onTab = onTab;
    this._build();
  }
  _build() {
    clear(this.el);
    const tabs = segmented([
      { value: "bending", label: "Bending" },
      { value: "torques", label: "Torques" },
      { value: "time", label: "Over time" },
    ], this.tab, (v) => { this.tab = v; this._showTab(); this.render(this.last, true); this.onTab?.(v); }, { label: "Analysis view" });
    this.axisSeg = segmented(["pitch", "roll", "yaw"].map((a) => ({ value: a, label: a[0].toUpperCase() + a.slice(1), tip: AXIS_LABEL[a] })),
      this.axis, (v) => { this.axis = v; this.render(this.last, true); }, { label: "Joint axis", small: true });
    this.jointSel = h("select", { class: "select select-small", "aria-label": "Joint for torque over time" });
    this.jointSel.addEventListener("change", () => { this.joint = this.jointSel.value === "auto" ? null : Number(this.jointSel.value); this.render(this.last, true); });
    this.axisWrap = h("div", { class: "dock-tools" }, h("span", { class: "tool-label" }, "Joint axis"), this.axisSeg);
    this.jointWrap = h("div", { class: "dock-tools" }, h("span", { class: "tool-label" }, "Joint"), this.jointSel);
    this.head = h("div", { class: "dock-head" }, tabs, h("div", { class: "spacer" }), this.axisWrap, this.jointWrap);

    // --- bending
    this.pBend = h("div", { class: "dock-pane pane-bending" });
    const curvBox = h("div", { class: "dock-chart grow-2" });
    const sideBox = h("div", { class: "dock-chart shape" });
    const frontBox = h("div", { class: "dock-chart shape" });
    this.pBend.append(curvBox, sideBox, frontBox);
    const H = 196;
    this.curv = new BarChart(curvBox, { title: "Curvature profile", subtitle: "Bend of each joint relative to the segment before it", height: H,
      yLabel: "angle (°)", xLabel: "segment", categoryName: "Segment ", valueFmt: (v) => `${v.toFixed(1)}°`, minYSpan: 2 });
    this.side = new LineChart(sideBox, { title: "Side view", subtitle: "ventral ← → dorsal, as in 3D", height: H, equalAspect: true, hover: "nearest",
      margin: { top: 18, right: 8, bottom: 30, left: 34 }, xLabel: "dorsal (mm)", yLabel: "z (mm)", valueFmt: (v) => `z ${v.toFixed(0)} mm`, hoverXFmt: (v) => `${(-v).toFixed(1)} mm ventral`, legend: false });
    this.front = new LineChart(frontBox, { title: "Front view", subtitle: "sinistral ← → dextral", height: H, equalAspect: true, hover: "nearest",
      margin: { top: 18, right: 8, bottom: 30, left: 34 }, xLabel: "dextral (mm)", yLabel: "z (mm)", valueFmt: (v) => `z ${v.toFixed(0)} mm`, hoverXFmt: (v) => `${v.toFixed(1)} mm dextral`, legend: false });

    // --- torques
    this.pTorque = h("div", { class: "dock-pane pane-torques" });
    const tqBox = h("div", { class: "dock-chart grow-2" });
    const maBox = h("div", { class: "dock-chart grow-15 heat-box" });
    this.pTorque.append(tqBox, maBox);
    this.tq = new BarChart(tqBox, { title: "Joint torque from the tendons", subtitle: "Each tendon's share, stacked. Line = total on that joint.", height: H,
      stacked: true, yLabel: "torque (N·mm)", xLabel: "segment", categoryName: "Joint at segment ", valueFmt: (v) => `${v.toFixed(1)} N·mm`, minYSpan: 0.2 });
    this.ma = new HeatTable(maBox, { title: "Moment arms", subtitle: "mm; length change of the tendon per radian of joint rotation", rowHeader: "tendon", colPrefix: "", valueFmt: (v) => (Math.abs(v) >= 9.95 ? v.toFixed(0) : v.toFixed(1)), unit: "mm",
      invertColor: true, key: { pos: "pulling bends ventral", neg: "pulling bends dorsal" } });

    // --- time
    this.pTime = h("div", { class: "dock-pane pane-time" });
    const t1 = h("div", { class: "dock-chart" }), t2 = h("div", { class: "dock-chart" }), t3 = h("div", { class: "dock-chart" });
    this.pTime.append(t1, t2, t3);
    const tm = { top: 18, right: 10, bottom: 30, left: 40 };
    this.tTip = new LineChart(t1, { title: "Tip angle", subtitle: "Tip relative to the base", height: H, margin: tm, xLabel: "time (s)", yLabel: "°",
      valueFmt: (v) => `${v.toFixed(1)}°`, hoverXFmt: (v) => `${v.toFixed(2)} s`, symmetric: false, includeZero: true, minYSpan: 4 });
    this.tForce = new LineChart(t2, { title: "Tendon force", subtitle: "Pulling force of each tendon", height: H, margin: tm, xLabel: "time (s)", yLabel: "N",
      valueFmt: (v) => `${v.toFixed(2)} N`, hoverXFmt: (v) => `${v.toFixed(2)} s`, includeZero: true, minYSpan: 1 });
    this.tTorque = new LineChart(t3, { title: "Joint torque", subtitle: "", height: H, margin: tm, xLabel: "time (s)", yLabel: "N·mm",
      valueFmt: (v) => `${v.toFixed(2)} N·mm`, hoverXFmt: (v) => `${v.toFixed(2)} s`, includeZero: true, minYSpan: 0.2 });
    this.el.append(this.head, this.pBend, this.pTorque, this.pTime);
    this._showTab();
  }
  _showTab() {
    this.pBend.hidden = this.tab !== "bending";
    this.pTorque.hidden = this.tab !== "torques";
    this.pTime.hidden = this.tab !== "time";
    this.axisWrap.hidden = this.tab !== "torques";
    this.jointWrap.hidden = this.tab !== "time";
  }

  /** New simulation (rebuild or reset): clear history, remember rest pose. */
  setSimulation(sim) {
    this.sim = sim;
    this.buf = [];
    this.rest = null;
    try {
      const n = sim.numSegments || sim.catalog?.num_segments || 11;
      const sp = sim.catalog?.segment_spacing || 0.032;
      this.rest = Array.from({ length: n }, (_, i) => [0, 0, i * sp]);
    } catch { this.rest = null; }
    const n = sim?.numSegments || 11;
    clear(this.jointSel);
    this.jointSel.append(h("option", { value: "auto" }, "Most loaded"));
    for (let i = 1; i < n; i++) this.jointSel.append(h("option", { value: i }, `Segment ${i}`));
    this.jointSel.value = this.joint === null ? "auto" : String(this.joint);
  }
  resetHistory() { this.buf = []; }

  push(m) {
    if (!m) return;
    const b = this.buf;
    if (b.length && m.time < b[b.length - 1].time) b.length = 0; // sim reset
    b.push({
      time: m.time, ventral: m.tip.ventral, lateral: m.tip.lateral,
      forces: m.tendons.map((t) => t.force),
      torque: m.segments.map((s) => s.torque ? [s.torque.pitch, s.torque.roll, s.torque.yaw] : [0, 0, 0]),
    });
    const t0 = m.time - this.window;
    let k = 0;
    while (k < b.length && b[k].time < t0) k++;
    if (k) b.splice(0, k);
  }

  render(m, force = false) {
    if (!m) return;
    this.last = m;
    if (this.tab === "bending") this._renderBending(m);
    else if (this.tab === "torques") this._renderTorques(m);
    else this._renderTime(m);
  }

  _renderBending(m) {
    const cats = m.segments.map((s) => String(s.index));
    this.curv.setData({ categories: cats, series: [
      { id: "ventral", label: "Ventral (+) / dorsal (−)", color: SERIES[0], values: m.segments.map((s) => s.ventral) },
      { id: "lateral", label: "Dextral (+) / sinistral (−)", color: SERIES[1], values: m.segments.map((s) => s.lateral) },
    ] });
    const pts = [[0, 0, 0], ...m.segments.map((s) => s.pos)];
    const hanging = this.sim?.config?.params?.orientation === "hanging";
    const zf = hanging ? -1 : 1;
    const toXZ = (p) => [-p[0] * MM, zf * p[2] * MM]; // ventral (+x) to the left, like the 3D side view
    const toYZ = (p) => [-p[1] * MM, zf * p[2] * MM]; // dextral (−y) to the right
    const rest = this.rest || [];
    this.side.setSeries([
      { id: "rest", label: "Rest pose", color: "var(--chart-muted)", points: rest.map(toXZ), width: 1.5, noHover: true },
      { id: "now", label: "Now", color: SERIES[0], points: pts.map(toXZ), markers: true },
    ]);
    this.front.setSeries([
      { id: "rest", label: "Rest pose", color: "var(--chart-muted)", points: rest.map(toYZ), width: 1.5, noHover: true },
      { id: "now", label: "Now", color: SERIES[1], points: pts.map(toYZ), markers: true },
    ]);
  }

  _renderTorques(m) {
    const ax = this.axis;
    const cats = m.segments.map((s) => String(s.index));
    const tend = (this.sim?.tendons || []);
    const series = m.tendons.filter((t) => t.torque).map((t, i) => ({
      id: t.name, label: t.name, color: tend.find((x) => x.name === t.name)?.color || SERIES[i % 8],
      values: t.torque.map((q) => q[ax] * NMM),
    })).filter((s) => s.values.some((v) => Math.abs(v) > 1e-9));
    const totals = m.segments.map((s) => (s.torque ? s.torque[ax] * NMM : NaN));
    this.tq.opts.subtitle = AXIS_LABEL[ax];
    const sub = this.tq.root.querySelector(".chart-sub");
    if (sub) sub.textContent = `${AXIS_LABEL[ax]}: each tendon's share, stacked; bar = total on that joint`;
    this.tq.opts.emptyText = "No torque: activate a tendon with the sliders on the right";
    this.tq.setData({ categories: cats, series, totals: series.length ? totals : null, totalLabel: "Total (all tendons)" });
    const rows = m.tendons.filter((t) => t.moment_arms).map((t, i) => ({
      label: t.name, color: tend.find((x) => x.name === t.name)?.color || SERIES[i % 8],
      values: t.moment_arms.map((a) => a[ax] * MM),
    }));
    this.ma.setKey(ax === "pitch" ? { pos: "pulling bends ventral", neg: "pulling bends dorsal" }
      : ax === "roll" ? { pos: "pulling bends dextral", neg: "pulling bends sinistral" }
        : { pos: "pulling twists +", neg: "pulling twists −" });
    this.ma.setData({ columns: cats, rows, emptyText: "Add a tendon to see its moment arms." });
  }

  _renderTime(m) {
    const b = this.buf;
    const tend = this.sim?.tendons || [];
    this.tTip.setSeries([
      { id: "v", label: "Ventral", color: SERIES[0], points: b.map((r) => [r.time, r.ventral]) },
      { id: "l", label: "Lateral (dextral +)", color: SERIES[1], points: b.map((r) => [r.time, r.lateral]) },
    ]);
    const fs = [];
    tend.forEach((t, i) => {
      if (t.actuatorId === null && !t.stiffness) return;
      fs.push({ id: t.name, label: t.name, color: t.color || SERIES[i % 8], points: b.map((r) => [r.time, r.forces[i] ?? 0]) });
    });
    this.tForce.opts.emptyText = "No actuated tendons";
    this.tForce.setSeries(fs);
    let j = this.joint;
    if (j === null) {
      let best = 1, bv = -1;
      for (const s of m.segments) { const v = s.torque ? Math.abs(s.torque.pitch) + Math.abs(s.torque.roll) : 0; if (v > bv) { bv = v; best = s.index; } }
      j = best;
    }
    const k = m.segments.findIndex((s) => s.index === j);
    const sub = this.tTorque.root.querySelector(".chart-sub");
    if (sub) sub.textContent = `Joint at segment ${j}${this.joint === null ? " (most loaded now)" : ""}, all tendons`;
    this.tTorque.setSeries(k < 0 ? [] : ["pitch", "roll", "yaw"].map((ax, a) => ({
      id: ax, label: AXIS_SHORT[ax], color: SERIES[a], points: b.map((r) => [r.time, (r.torque[k]?.[a] ?? 0) * NMM]),
    })));
  }
}

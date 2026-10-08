// Stand-ins used ONLY when a real module (sim.js, render3d.js, slice_editor.js) fails to load.
// app.js shows a "stand-in" badge whenever any of these is in use, so it is never silent.
//   MockSimulation: a crude quasi-static bending model with the same API + Measurement shape.
//   MockViewer3D:   a 2D canvas side/front view of the tail.
//   MockSliceEditor: a minimal clickable SVG cross-section (append taps to the active tendon).

import { h, s, clear, deepCopy } from "./ui.js";
import { pointIndex } from "./config_tools.js";

const AX = ["pitch", "roll", "yaw"];
const D2R = Math.PI / 180, R2D = 180 / Math.PI;

export class MockSimulation {
  static async create(_mujoco, assets, config) {
    return new MockSimulation(assets, config);
  }
  constructor(assets, config) {
    this.catalog = assets.catalog;
    this.config = config;
    this.isMock = true;
    this.xml = `<!-- stand-in simulation: the real MuJoCo model was not available -->\n<mujoco model="mock"/>`;
    this.numSegments = this.catalog.num_segments;
    this.timestep = config.params?.timestep || this.catalog.defaults?.timestep || 0.002;
    const idx = pointIndex(this.catalog, config.free_points);
    const spacing = this.catalog.segment_spacing || 0.032;
    this.tendons = (config.tendons || []).map((t, i) => {
      const pts = (t.path || []).map((id) => idx.get(id)).filter(Boolean);
      const segs = pts.map((p) => p.segment);
      const lo = segs.length ? Math.min(...segs) : 0, hi = segs.length ? Math.max(...segs) : 0;
      const xy = pts.length ? [pts.reduce((a, p) => a + p.xy[0], 0) / pts.length, pts.reduce((a, p) => a + p.xy[1], 0) / pts.length] : [0, 0];
      return { name: t.name, group: t.group ?? "", color: t.color, type: t.actuator?.type || "none",
        tendonId: i, actuatorId: t.actuator?.type && t.actuator.type !== "none" ? i : null,
        length0: Math.max(0.01, (hi - lo) * spacing + 0.004), lo, hi, xy, maxForce: t.actuator?.max_force || 10,
        maxStrain: t.actuator?.max_strain ?? 0.26 };
    });
    this._act = new Float64Array(this.tendons.length);
    this.reset();
  }
  get time() { return this._time; }
  setActivation(name, a) { const i = this.tendons.findIndex((t) => t.name === name); if (i < 0) throw new Error(`Unknown tendon "${name}"`); this._act[i] = Math.max(0, Math.min(1, +a || 0)); }
  getActivation(name) { const i = this.tendons.findIndex((t) => t.name === name); return this._act[i]; }
  setActivations(map) { for (const [k, v] of Object.entries(map)) this.setActivation(k, v); }
  reset() {
    this._time = 0;
    this._q = new Float64Array(this.numSegments * 3);
    this._work = new Float64Array(this.tendons.length);
  }
  _params() {
    const d = this.catalog.defaults.vertebra, p = this.config.params?.vertebra || {};
    const g = (k) => (p[k] ?? d[k]);
    return { k: [g("pitch_stiffness"), g("roll_stiffness"), g("yaw_stiffness")], r: [g("pitch_range_deg"), g("roll_range_deg"), g("yaw_range_deg")].map((x) => x * D2R) };
  }
  _arms(t, i) {
    // moment arm (m): tendon at segment-frame (x, y) crossing joint i shortens with ventral pitch if x > 0
    if (i <= t.lo || i > t.hi) return [0, 0, 0];
    return [-t.xy[0] * 0.8, t.xy[1] * 0.8, 0];
  }
  _force(ti) { const t = this.tendons[ti]; return t.actuatorId === null ? 0 : this._act[ti] * (t.type === "position" ? 40 : t.maxForce); }
  step(n = 1) {
    const { k, r } = this._params();
    const dt = this.timestep;
    for (let s = 0; s < n; s++) {
      for (let i = 1; i < this.numSegments; i++) {
        for (let a = 0; a < 2; a++) {
          let tau = 0;
          this.tendons.forEach((t, ti) => { tau += -this._arms(t, i)[a] * this._force(ti); });
          const target = Math.max(-r[a], Math.min(r[a], tau / (k[a] * 400 + 1e-9)));
          const j = i * 3 + a;
          this._q[j] += (target - this._q[j]) * Math.min(1, dt / 0.08);
        }
      }
      this._time += dt;
    }
  }
  advance(seconds) { const n = Math.floor(seconds / this.timestep + 1e-9); if (n > 0) this.step(n); return n; }
  measure() {
    const n = this.numSegments, spacing = this.catalog.segment_spacing || 0.032;
    let R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let pos = [0, 0, 0];
    const segments = [], cum = [0];
    for (let i = 1; i < n; i++) {
      const dz = [R[0][2] * spacing, R[1][2] * spacing, R[2][2] * spacing];
      pos = [pos[0] + dz[0], pos[1] + dz[1], pos[2] + dz[2]];
      const p = this._q[i * 3], rl = this._q[i * 3 + 1];
      const Ry = [[Math.cos(p), 0, Math.sin(p)], [0, 1, 0], [-Math.sin(p), 0, Math.cos(p)]];
      const Rx = [[1, 0, 0], [0, Math.cos(rl), -Math.sin(rl)], [0, Math.sin(rl), Math.cos(rl)]];
      R = mul(R, mul(Ry, Rx));
      const torque = { pitch: 0, roll: 0, yaw: 0 };
      this.tendons.forEach((t, ti) => { const a = this._arms(t, i); AX.forEach((ax, k) => { torque[ax] += -a[k] * this._force(ti); }); });
      const ventral = p * R2D, lateral = rl * R2D;
      segments.push({ index: i, pos: [...pos], ventral, lateral, bend: Math.hypot(ventral, lateral), twist: 0,
        joint_angle: { pitch: ventral, roll: lateral, yaw: 0 }, torque,
        passive_torque: { pitch: -torque.pitch * 0.9, roll: -torque.roll * 0.9, yaw: 0 }, limit_torque: { pitch: 0, roll: 0, yaw: 0 } });
      cum.push(cum[i - 1] + ventral);
    }
    const tipPos = pos;
    const tip0 = [0, 0, spacing * (n - 1)];
    const disp = [tipPos[0] - tip0[0], tipPos[1] - tip0[1], tipPos[2] - tip0[2]];
    const tv = Math.atan2(R[0][2], R[2][2]) * R2D, tl = Math.atan2(-R[1][2], R[2][2]) * R2D;
    const tendons = this.tendons.map((t, ti) => {
      let exc = 0;
      const arms = [], tq = [];
      const force = this._force(ti);
      for (let i = 1; i < n; i++) {
        const a = this._arms(t, i);
        exc += -(a[0] * this._q[i * 3] + a[1] * this._q[i * 3 + 1]);
        arms.push({ segment: i, pitch: a[0], roll: a[1], yaw: 0 });
        tq.push({ segment: i, pitch: -a[0] * force, roll: -a[1] * force, yaw: 0 });
      }
      this._work[ti] = force * exc;
      return { name: t.name, group: t.group, activation: this._act[ti], length: t.length0 - exc, length0: t.length0,
        excursion: exc, strain: exc / t.length0, force, passive_force: 0, work: this._work[ti], moment_arms: arms, torque: tq };
    });
    return { time: this._time, tip: { pos: tipPos, displacement: disp, distance: Math.hypot(...disp), ventral: tv, lateral: tl, bend: Math.acos(Math.max(-1, Math.min(1, R[2][2]))) * R2D, twist: 0 },
      segments, cumulative_ventral: cum, tendons };
  }
  dispose() {}
}
function mul(A, B) { return A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j])); }

export class MockViewer3D {
  constructor(container, options = {}) {
    this.container = container; this.options = options;
    this.canvas = h("canvas", { class: "mock-viewer" });
    container.append(this.canvas);
    this.sim = null; this.highlight = null;
    this._ro = new ResizeObserver(() => this.resize()); this._ro.observe(container);
  }
  setSimulation(sim) { this.sim = sim; this.update(); }
  setTendonHighlight(name) { this.highlight = name; }
  setOptions(o) { Object.assign(this.options, o); }
  resize() { const r = this.container.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; this.canvas.width = r.width * dpr; this.canvas.height = r.height * dpr; this.canvas.style.width = r.width + "px"; this.canvas.style.height = r.height + "px"; this.update(); }
  update() {
    const c = this.canvas.getContext("2d"); if (!c || !this.sim) return;
    const W = this.canvas.width, H = this.canvas.height;
    c.clearRect(0, 0, W, H);
    const m = this.sim.measure({ torques: false });
    const pts = [[0, 0, 0], ...m.segments.map((x) => x.pos)];
    const scale = Math.min(W, H) / 0.42;
    const hanging = this.sim.config.params?.orientation === "hanging";
    const toPx = (p, ox) => [ox + p[0] * scale, hanging ? H * 0.12 + p[2] * scale : H * 0.88 - p[2] * scale];
    for (const [ox, ix] of [[W * 0.3, 0], [W * 0.7, 1]]) {
      c.strokeStyle = "rgba(180,220,230,0.25)"; c.lineWidth = 2 * (window.devicePixelRatio || 1);
      c.beginPath(); pts.forEach((p, i) => { const q = toPx([ix ? p[1] : p[0], 0, p[2]], ox); i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]); }); c.stroke();
      pts.forEach((p) => { const q = toPx([ix ? p[1] : p[0], 0, p[2]], ox); c.fillStyle = "rgba(220,240,245,0.9)"; c.beginPath(); c.arc(q[0], q[1], 9 * (window.devicePixelRatio || 1), 0, 7); c.fill(); });
      c.fillStyle = "rgba(220,240,245,0.6)"; c.font = `${12 * (window.devicePixelRatio || 1)}px system-ui`;
      c.fillText(ix ? "front view (stand-in)" : "side view (stand-in)", ox - 60, H - 12);
    }
  }
  dispose() { this._ro.disconnect(); this.canvas.remove(); }
}

export class MockSliceEditor {
  constructor(container, catalog, { onChange, onSelectTendon } = {}) {
    this.container = container; this.catalog = catalog; this.onChange = onChange; this.onSelectTendon = onSelectTendon;
    this.segment = 0; this.active = null; this.config = null;
    this.svg = s("svg", { class: "mock-slice", viewBox: "-0.07 -0.07 0.14 0.14" });
    container.append(this.svg);
  }
  setConfig(config) { this.config = deepCopy(config); this.render(); }
  setActiveTendon(name) { this.active = name; this.render(); }
  setSegment(i) { this.segment = i; this.render(); }
  render() {
    clear(this.svg);
    const seg = this.catalog.segments[this.segment]; if (!seg) return;
    // screen: +x (ventral) down, +y (sinistral) left? keep simple: screen x = -y, screen y = x
    const P = (xy) => [-xy[1], xy[0]];
    for (const poly of seg.vertebra_outline || []) this.svg.append(s("polygon", { points: poly.map((p) => P(p).join(",")).join(" "), class: "mock-vert" }));
    for (const plate of seg.plates) {
      for (const poly of plate.outline || []) this.svg.append(s("polygon", { points: poly.map((p) => P(p).join(",")).join(" "), class: "mock-plate" }));
      for (const tap of plate.taps) {
        const used = this.config?.tendons.find((t) => t.name === this.active)?.path.includes(tap.id);
        const [cx, cy] = P(tap.xy);
        const c = s("circle", { cx, cy, r: 0.0016, class: `mock-tap ${tap.kind}${used ? " used" : ""}` });
        c.append(s("title", {}, `${tap.label} (${tap.kind})`));
        c.addEventListener("click", () => {
          const t = this.config?.tendons.find((x) => x.name === this.active);
          if (!t) return;
          t.path.push(tap.id);
          this.render();
          this.onChange && this.onChange(deepCopy(this.config));
        });
        this.svg.append(c);
      }
    }
  }
}

// Lightweight SVG charts (no chart library). Line/XY charts with a crosshair tooltip,
// grouped or stacked (diverging) bar charts with per-bar tooltips, equal-aspect shape plots,
// and a diverging heat table. Chrome colours come from CSS custom properties (see app.css,
// ".chart"); series colours are passed in by the caller (tendon colour or --series-N).

import { h, s, clear } from "./ui.js";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
  "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)"];

// ------------------------------------------------------------------ scales & ticks
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1, ticks: [0, 0.5, 1], step: 0.5 };
  if (min === max) { const d = Math.abs(min) || 1; min -= d * 0.5; max += d * 0.5; }
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const lo = Math.floor(min / step + 1e-9) * step;
  const hi = Math.ceil(max / step - 1e-9) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { min: lo, max: hi, ticks, step };
}

function tickFormat(step) {
  const d = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
  return (v) => {
    const r = v.toFixed(d);
    return r === "-0" || /^-0\.0*$/.test(r) ? r.slice(1) : r;
  };
}

function extent(values, { includeZero = false, symmetric = false, minSpan = 0 } = {}) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === Infinity) { lo = 0; hi = 1; }
  if (includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  if (hi - lo < minSpan) {
    const c = symmetric ? 0 : (includeZero && lo >= 0 ? minSpan / 2 : (lo + hi) / 2);
    lo = c - minSpan / 2; hi = c + minSpan / 2;
    if (includeZero && !symmetric && (lo + hi) / 2 === minSpan / 2) { lo = Math.min(lo, 0); }
  }
  return [lo, hi];
}

function textWidth(str, px = 11) { return String(str).length * px * 0.58; }

// ------------------------------------------------------------------ base
class ChartBase {
  constructor(container, opts = {}) {
    this.opts = { height: 180, ...opts, margin: { top: 18, right: 12, bottom: 30, left: 44, ...(opts.margin || {}) } };
    this.root = h("div", { class: `chart ${opts.className || ""}` });
    this.legendEl = h("div", { class: "chart-legend" });
    this.plot = h("div", { class: "chart-plot" });
    this.tip = h("div", { class: "chart-tip", role: "status" });
    this.empty = h("div", { class: "chart-empty" });
    this.plot.append(this.tip, this.empty);
    if (opts.title) {
      this.root.append(h("div", { class: "chart-head" },
        h("div", { class: "chart-title" }, opts.title),
        opts.subtitle ? h("div", { class: "chart-sub" }, opts.subtitle) : null));
    }
    this.root.append(this.legendEl, this.plot);
    container.append(this.root);
    this.width = 0;
    this._ro = new ResizeObserver(() => {
      const w = Math.floor(this.plot.clientWidth);
      if (w && w !== this.width) { this.width = w; this.render(); }
    });
    this._ro.observe(this.plot);
    this.hover = null;
  }
  get height() { return typeof this.opts.height === "function" ? this.opts.height(this.width) : this.opts.height; }
  setEmpty(msg) {
    this.empty.textContent = msg || "";
    this.empty.classList.toggle("show", !!msg);
  }
  setLegend(items) {
    clear(this.legendEl);
    const show = items && items.length >= 2 && this.opts.legend !== false;
    this.legendEl.classList.toggle("show", !!show);
    if (!show) return;
    for (const it of items) {
      const key = it.mark === "bar"
        ? h("span", { class: "key key-bar", style: { background: it.color } })
        : h("span", { class: `key key-line${it.dash ? " dashed" : ""}`, style: { borderColor: it.color } });
      this.legendEl.append(h("span", { class: "legend-item" }, key, it.label));
    }
  }
  _svg() {
    const W = this.width, H = this.height;
    let svg = this.plot.querySelector("svg.chart-svg");
    if (!svg) {
      svg = s("svg", { class: "chart-svg" });
      this.plot.prepend(svg);
    }
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    clear(svg);
    return svg;
  }
  _axes(svg, x, y, xt, yt, { xLabel, yLabel, xFmt, yFmt, zero = true }) {
    const m = this.opts.margin;
    const W = this.width, H = this.height;
    const g = s("g", { class: "axes" });
    for (const v of yt.ticks) {
      const py = y(v);
      g.append(s("line", { class: v === 0 && zero ? "zero" : "grid", x1: m.left, x2: W - m.right, y1: py, y2: py }));
      g.append(s("text", { class: "tick", x: m.left - 6, y: py, "text-anchor": "end", "dominant-baseline": "middle" }, yFmt(v)));
    }
    let lastRight = -Infinity;
    for (const v of xt.ticks) {
      const px = x(v);
      const label = xFmt(v);
      const w = textWidth(label);
      if (px - w / 2 < lastRight + 6) continue;
      lastRight = px + w / 2;
      g.append(s("line", { class: "tickmark", x1: px, x2: px, y1: H - m.bottom, y2: H - m.bottom + 4 }));
      g.append(s("text", { class: "tick", x: px, y: H - m.bottom + 15, "text-anchor": "middle" }, label));
    }
    g.append(s("line", { class: "baseline", x1: m.left, x2: W - m.right, y1: H - m.bottom, y2: H - m.bottom }));
    if (xLabel) g.append(s("text", { class: "axis-label", x: W - m.right, y: H - 3, "text-anchor": "end" }, xLabel));
    if (yLabel) g.append(s("text", { class: "axis-label", x: 2, y: 9, "text-anchor": "start" }, yLabel));
    svg.append(g);
  }
  _showTip(px, py, rows, header) {
    clear(this.tip);
    if (header) this.tip.append(h("div", { class: "tip-head" }, header));
    for (const r of rows) {
      this.tip.append(h("div", { class: "tip-row" },
        h("span", { class: `key key-line${r.dash ? " dashed" : ""}`, style: { borderColor: r.color } }),
        h("strong", {}, r.value), h("span", { class: "tip-label" }, r.label)));
    }
    this.tip.classList.add("show");
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    const W = this.width;
    let x = px + 14;
    if (x + tw > W - 4) x = px - tw - 14;
    x = Math.max(2, x);
    let y = Math.max(2, Math.min(this.height - th - 2, py - th / 2));
    this.tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }
  _hideTip() { this.tip.classList.remove("show"); }
  dispose() { this._ro.disconnect(); this.root.remove(); }
}

// ------------------------------------------------------------------ line / XY chart
/**
 * new LineChart(container, {title, xLabel, yLabel, xFmt, yFmt, valueFmt, hover: "x"|"nearest",
 *   includeZero, symmetric, minYSpan, xDomain:[a,b], yDomain:[a,b], equalAspect, height, markers})
 * setSeries([{id, label, color, points: [[x,y],...], dash, width, opacity, markers, endDot}])
 */
export class LineChart extends ChartBase {
  constructor(container, opts = {}) {
    super(container, { hover: "x", ...opts });
    this.series = [];
    this.plot.tabIndex = 0;
    this.plot.addEventListener("pointermove", (e) => {
      const r = this.plot.getBoundingClientRect();
      this.hover = { px: e.clientX - r.left, py: e.clientY - r.top };
      this._renderHover();
    });
    this.plot.addEventListener("pointerleave", () => { this.hover = null; this._renderHover(); });
    this.plot.addEventListener("keydown", (e) => {
      if (!this._scales || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      const m = this.opts.margin;
      const cur = this.hover?.px ?? m.left;
      const px = Math.max(m.left, Math.min(this.width - m.right, cur + (e.key === "ArrowLeft" ? -12 : 12)));
      this.hover = { px, py: this.height / 2 };
      this._renderHover();
      e.preventDefault();
    });
    this.plot.addEventListener("blur", () => { this.hover = null; this._renderHover(); });
  }
  setSeries(series) {
    this.series = series || [];
    const seen = new Set(), items = [];
    for (const x of this.series) {
      if (x.hideLegend) continue;
      const label = x.legendLabel || x.label;
      if (seen.has(label)) continue;
      seen.add(label);
      items.push({ label, color: x.color, dash: x.dash });
    }
    this.setLegend(items);
    this.render();
  }
  render() {
    if (!this.width) return;
    const o = this.opts, m = o.margin;
    const W = this.width, H = this.height;
    const svg = this._svg();
    const xs = [], ys = [];
    for (const se of this.series) for (const p of se.points || []) { xs.push(p[0]); ys.push(p[1]); }
    const has = xs.length > 0;
    this.setEmpty(has ? "" : (o.emptyText || "No data yet"));
    let [x0, x1] = o.xDomain || extent(xs, { includeZero: o.xIncludeZero, symmetric: o.xSymmetric, minSpan: o.minXSpan || 0 });
    let [y0, y1] = o.yDomain || extent(ys, { includeZero: o.includeZero, symmetric: o.symmetric, minSpan: o.minYSpan || 0 });
    const pw = W - m.left - m.right, ph = H - m.top - m.bottom;
    if (o.equalAspect) {
      // same metres per pixel on both axes, centred on the data
      const sx = (x1 - x0) / pw, sy = (y1 - y0) / ph;
      const sc = Math.max(sx, sy) * 1.08 || 1;
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - sc * pw / 2; x1 = cx + sc * pw / 2;
      y0 = cy - sc * ph / 2; y1 = cy + sc * ph / 2;
    }
    const xt = o.equalAspect ? { ...niceTicks(x0, x1, Math.max(2, Math.round(pw / 60))), min: x0, max: x1 } : niceTicks(x0, x1, Math.max(2, Math.round(pw / 70)));
    const yt = o.equalAspect ? { ...niceTicks(y0, y1, Math.max(2, Math.round(ph / 40))), min: y0, max: y1 } : niceTicks(y0, y1, Math.max(2, Math.round(ph / 34)));
    if (o.equalAspect) { xt.ticks = xt.ticks.filter((v) => v >= x0 && v <= x1); yt.ticks = yt.ticks.filter((v) => v >= y0 && v <= y1); }
    const xmin = o.xDomain ? o.xDomain[0] : xt.min, xmax = o.xDomain ? o.xDomain[1] : xt.max;
    const ymin = yt.min, ymax = yt.max;
    const x = (v) => m.left + ((v - xmin) / (xmax - xmin || 1)) * pw;
    const y = (v) => m.top + (1 - (v - ymin) / (ymax - ymin || 1)) * ph;
    this._scales = { x, y, xmin, xmax, ymin, ymax, xinv: (px) => xmin + ((px - m.left) / pw) * (xmax - xmin) };
    this._axes(svg, x, y, o.xDomain ? { ticks: niceTicks(xmin, xmax, Math.max(2, Math.round(pw / 70))).ticks.filter((v) => v >= xmin - 1e-9 && v <= xmax + 1e-9) } : xt, yt, {
      xLabel: o.xLabel, yLabel: o.yLabel,
      xFmt: o.xFmt || tickFormat(xt.step || 1), yFmt: o.yFmt || tickFormat(yt.step || 1),
    });
    const clipId = `clip-${Math.random().toString(36).slice(2, 8)}`;
    svg.append(s("defs", {}, s("clipPath", { id: clipId }, s("rect", { x: m.left, y: m.top - 4, width: pw, height: ph + 8 }))));
    const g = s("g", { "clip-path": `url(#${clipId})` });
    for (const se of this.series) {
      const pts = (se.points || []).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      if (!pts.length) continue;
      if (se.area) {
        const d = `M${x(pts[0][0])},${y(0)}` + pts.map((p) => `L${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("") + `L${x(pts[pts.length - 1][0])},${y(0)}Z`;
        g.append(s("path", { d, class: "area", style: `fill:${se.color}` }));
      }
      const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
      g.append(s("path", { d, class: `line${se.dash ? " dashed" : ""}`,
        style: `stroke:${se.color};stroke-width:${se.width || 2}px;opacity:${se.opacity ?? 1}` }));
      if (se.markers) {
        for (const p of pts) g.append(s("circle", { class: "dot", cx: x(p[0]), cy: y(p[1]), r: se.markerR || 3.5, style: `fill:${se.color};opacity:${se.opacity ?? 1}` }));
      }
      if (se.endDot) {
        const p = pts[pts.length - 1];
        g.append(s("circle", { class: "dot", cx: x(p[0]), cy: y(p[1]), r: 4.5, style: `fill:${se.color}` }));
      }
    }
    svg.append(g);
    this.hoverLayer = s("g", { class: "hover" });
    svg.append(this.hoverLayer);
    this._renderHover();
  }
  _renderHover() {
    if (!this.hoverLayer || !this._scales) return;
    clear(this.hoverLayer);
    const o = this.opts, m = o.margin, sc = this._scales;
    if (!this.hover || this.hover.px < m.left - 4 || this.hover.px > this.width - m.right + 4) { this._hideTip(); return; }
    const vf = o.valueFmt || ((v) => v.toFixed(2));
    const xf = o.hoverXFmt || o.xFmt || ((v) => v.toFixed(2));
    const live = this.series.filter((se) => se.points && se.points.length && !se.noHover);
    if (!live.length) { this._hideTip(); return; }
    if (o.hover === "nearest") {
      let best = null;
      for (const se of live) for (const p of se.points) {
        const dx = sc.x(p[0]) - this.hover.px, dy = sc.y(p[1]) - this.hover.py;
        const d = dx * dx + dy * dy;
        if (!best || d < best.d) best = { d, se, p };
      }
      if (!best || best.d > 60 * 60) { this._hideTip(); return; }
      const cx = sc.x(best.p[0]), cy = sc.y(best.p[1]);
      this.hoverLayer.append(s("circle", { class: "hover-dot", cx, cy, r: 5, style: `fill:${best.se.color}` }));
      const rows = [{ color: best.se.color, dash: best.se.dash, value: vf(best.p[1], best.se), label: best.se.label }];
      this._showTip(cx, cy, rows, `${o.xName ? o.xName + " " : ""}${xf(best.p[0])}`);
      return;
    }
    const xv = sc.xinv(this.hover.px);
    // nearest x across series (series are assumed sorted by x)
    let snapX = null, bestD = Infinity;
    for (const se of live) {
      const p = nearestByX(se.points, xv);
      if (p && Math.abs(p[0] - xv) < bestD) { bestD = Math.abs(p[0] - xv); snapX = p[0]; }
    }
    if (snapX === null) { this._hideTip(); return; }
    const px = sc.x(snapX);
    this.hoverLayer.append(s("line", { class: "crosshair", x1: px, x2: px, y1: m.top, y2: this.height - m.bottom }));
    const rows = [];
    let sumY = 0, nY = 0;
    for (const se of live) {
      const p = nearestByX(se.points, snapX);
      if (!p || Math.abs(p[0] - snapX) > (sc.xmax - sc.xmin) * 0.03 + 1e-12) continue;
      const cy = sc.y(p[1]);
      this.hoverLayer.append(s("circle", { class: "hover-dot", cx: sc.x(p[0]), cy, r: 4.5, style: `fill:${se.color}` }));
      rows.push({ color: se.color, dash: se.dash, value: vf(p[1], se), label: se.label });
      sumY += cy; nY++;
    }
    this._showTip(px, nY ? sumY / nY : this.height / 2, rows, `${o.xName ? o.xName + " " : ""}${xf(snapX)}`);
  }
}

function nearestByX(points, xv) {
  if (!points.length) return null;
  let lo = 0, hi = points.length - 1;
  if (points[hi][0] < points[lo][0]) { // not sorted ascending: linear scan
    let best = points[0];
    for (const p of points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
    return best;
  }
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (points[mid][0] < xv) lo = mid; else hi = mid; }
  return Math.abs(points[lo][0] - xv) <= Math.abs(points[hi][0] - xv) ? points[lo] : points[hi];
}

// ------------------------------------------------------------------ bar chart
/**
 * new BarChart(container, {title, yLabel, xLabel, valueFmt, stacked, symmetric, minYSpan})
 * setData({categories: [label,...], series: [{id, label, color, values: [...]}], totals: [...]|null, totalLabel})
 * Grouped bars (side by side) or stacked diverging bars (positive up, negative down from 0).
 */
export class BarChart extends ChartBase {
  constructor(container, opts = {}) {
    super(container, { stacked: false, ...opts });
    this.data = { categories: [], series: [] };
  }
  setData(data) {
    this.data = data;
    const items = data.series.map((x) => ({ label: x.label, color: x.color, mark: "bar" }));
    if (data.totals) items.push({ label: data.totalLabel || "Total", color: "var(--ink)", mark: "line" });
    this.setLegend(items);
    this.render();
  }
  render() {
    if (!this.width) return;
    const o = this.opts, m = o.margin;
    const W = this.width, H = this.height;
    const svg = this._svg();
    const { categories, series, totals } = this.data;
    const n = categories.length;
    const vals = [];
    if (o.stacked) {
      for (let i = 0; i < n; i++) {
        let pos = 0, neg = 0;
        for (const se of series) { const v = se.values[i] || 0; if (v > 0) pos += v; else neg += v; }
        vals.push(pos, neg);
      }
    } else for (const se of series) vals.push(...se.values);
    if (totals) vals.push(...totals);
    const hasData = n > 0 && series.length > 0;
    this.setEmpty(hasData ? "" : (o.emptyText || "No data yet"));
    const [y0, y1] = extent(vals, { includeZero: true, symmetric: o.symmetric, minSpan: o.minYSpan || 0 });
    const pw = W - m.left - m.right, ph = H - m.top - m.bottom;
    const yt = niceTicks(y0, y1, Math.max(2, Math.round(ph / 34)));
    const y = (v) => m.top + (1 - (v - yt.min) / (yt.max - yt.min || 1)) * ph;
    const band = pw / Math.max(1, n);
    const xc = (i) => m.left + band * (i + 0.5);
    // axes
    const g0 = s("g", { class: "axes" });
    const yFmt = o.yFmt || tickFormat(yt.step);
    for (const v of yt.ticks) {
      g0.append(s("line", { class: v === 0 ? "zero" : "grid", x1: m.left, x2: W - m.right, y1: y(v), y2: y(v) }));
      g0.append(s("text", { class: "tick", x: m.left - 6, y: y(v), "text-anchor": "end", "dominant-baseline": "middle" }, yFmt(v)));
    }
    const every = Math.ceil((textWidth("00") + 6) / band);
    categories.forEach((c, i) => {
      if (i % every) return;
      g0.append(s("text", { class: "tick", x: xc(i), y: H - m.bottom + 15, "text-anchor": "middle" }, c));
    });
    if (o.xLabel) g0.append(s("text", { class: "axis-label", x: W - m.right, y: H - 3, "text-anchor": "end" }, o.xLabel));
    if (o.yLabel) g0.append(s("text", { class: "axis-label", x: 2, y: 9 }, o.yLabel));
    svg.append(g0);
    const g = s("g", { class: "bars" });
    const vf = o.valueFmt || ((v) => v.toFixed(2));
    const k = series.length || 1;
    const gap = 2;
    const groupW = Math.min(band * 0.78, o.stacked ? 24 : 24 * k + gap * (k - 1));
    const barW = o.stacked ? groupW : Math.max(2, Math.min(24, (groupW - gap * (k - 1)) / k));
    const radius = Math.min(4, barW / 2);
    for (let i = 0; i < n; i++) {
      let pos = 0, neg = 0;
      const tipRows = [];
      const hit = s("rect", { class: "bar-hit", x: xc(i) - band / 2, y: m.top, width: band, height: ph, tabindex: 0 });
      const marks = [];
      series.forEach((se, j) => {
        const v = se.values[i] || 0;
        tipRows.push({ color: se.color, value: vf(v), label: se.label });
        if (!v) return;
        let a, b, bx;
        if (o.stacked) {
          bx = xc(i) - barW / 2;
          if (v > 0) { a = pos; b = pos + v; pos = b; } else { a = neg; b = neg + v; neg = b; }
        } else {
          bx = xc(i) - groupW / 2 + j * (barW + gap);
          a = 0; b = v;
        }
        const ya = y(a), yb = y(b);
        // 2px surface gap between stacked segments
        const top = Math.min(ya, yb), hgt = Math.max(0.5, Math.abs(yb - ya) - (o.stacked && a !== 0 ? gap : 0));
        const yy = o.stacked && a !== 0 && v > 0 ? top : (o.stacked && a !== 0 && v < 0 ? top + gap : top);
        marks.push(s("path", { class: "bar", d: barPath(bx, yy, barW, hgt, v > 0 ? "top" : "bottom", radius), style: `fill:${se.color}` }));
      });
      if (totals && Number.isFinite(totals[i])) {
        const ty = y(totals[i]);
        marks.push(s("line", { class: "total-tick", x1: xc(i) - groupW / 2 - 3, x2: xc(i) + groupW / 2 + 3, y1: ty, y2: ty }));
        tipRows.push({ color: "var(--ink)", value: vf(totals[i]), label: o.totalLabel || this.data.totalLabel || "Total" });
      }
      const grp = s("g", { class: "bar-group" }, ...marks, hit);
      const show = () => {
        grp.classList.add("hovered");
        const cy = y(Math.max(pos, totals ? totals[i] || 0 : 0));
        this._showTip(xc(i), Math.max(m.top + 20, cy), tipRows.slice().reverse(), `${o.categoryName || ""}${categories[i]}`);
      };
      const hide = () => { grp.classList.remove("hovered"); this._hideTip(); };
      hit.addEventListener("pointerenter", show);
      hit.addEventListener("pointerleave", hide);
      hit.addEventListener("focus", show);
      hit.addEventListener("blur", hide);
      g.append(grp);
    }
    svg.append(g);
  }
}

function barPath(x, y, w, hgt, roundEnd, r) {
  r = Math.min(r, hgt / 2, w / 2);
  if (r <= 0.2) return `M${x},${y}h${w}v${hgt}h${-w}Z`;
  if (roundEnd === "top") {
    return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
  }
  return `M${x},${y}H${x + w}V${y + hgt - r}Q${x + w},${y + hgt} ${x + w - r},${y + hgt}H${x + r}Q${x},${y + hgt} ${x},${y + hgt - r}Z`;
}

// ------------------------------------------------------------------ heat table
/**
 * Diverging heat table (rows x columns), used for moment arms. Values are signed; colour
 * intensity encodes magnitude (blue = positive, red = negative, grey = ~0). Text stays ink.
 */
export class HeatTable {
  constructor(container, { title, subtitle, rowHeader = "", colPrefix = "", valueFmt, unit = "", invertColor = false, key = null } = {}) {
    this.root = h("div", { class: "heat" });
    if (title) this.root.append(h("div", { class: "chart-head" }, h("div", { class: "chart-title" }, title), subtitle ? h("div", { class: "chart-sub" }, subtitle) : null));
    this.tableWrap = h("div", { class: "heat-wrap" });
    this.root.append(this.tableWrap);
    container.append(this.root);
    this.o = { rowHeader, colPrefix, valueFmt: valueFmt || ((v) => v.toFixed(1)), unit, invertColor };
    if (key) {
      this.keyEl = h("div", { class: "heat-key" },
        h("span", { class: "heat-swatch pos" }), h("span", { class: "heat-key-text" }, key.pos),
        h("span", { class: "heat-swatch neg" }), h("span", { class: "heat-key-text" }, key.neg));
      this.root.append(this.keyEl);
    }
    this._key = "";
  }
  setKey(key) {
    if (!this.keyEl || !key) return;
    const t = this.keyEl.querySelectorAll(".heat-key-text");
    t[0].textContent = key.pos; t[1].textContent = key.neg;
  }
  setData({ columns, rows, emptyText }) {
    const o = this.o;
    clear(this.tableWrap);
    if (!rows.length) { this.tableWrap.append(h("div", { class: "chart-empty show static" }, emptyText || "No data yet")); return; }
    let max = 0;
    for (const r of rows) for (const v of r.values) if (Number.isFinite(v)) max = Math.max(max, Math.abs(v));
    max = max || 1;
    const table = h("table", { class: "heat-table" });
    table.append(h("thead", {}, h("tr", {}, h("th", { class: "rowhead" }, o.rowHeader), ...columns.map((c) => h("th", {}, `${o.colPrefix}${c}`)))));
    const tb = h("tbody");
    for (const r of rows) {
      const tr = h("tr", {}, h("th", { class: "rowhead" }, h("span", { class: "key key-line", style: { borderColor: r.color } }), r.label));
      r.values.forEach((v, i) => {
        const t = Number.isFinite(v) ? Math.min(1, Math.abs(v) / max) : 0;
        const sgn = o.invertColor ? -v : v;
        const pol = sgn > 0 ? "pos" : sgn < 0 ? "neg" : "zero";
        const step = Math.round(t * 5);
        tr.append(h("td", { class: `heat-cell ${pol} s${step}`, title: `${r.label}, ${o.colPrefix}${columns[i]}: ${Number.isFinite(v) ? o.valueFmt(v) : "–"} ${o.unit}` },
          Number.isFinite(v) && Math.abs(v) >= max * 0.005 ? o.valueFmt(v) : "·"));
      });
      tb.append(tr);
    }
    table.append(tb);
    this.tableWrap.append(table);
  }
}

// ------------------------------------------------------------------ sparkline-ish meter
export function meter(fraction, color) {
  const f = Math.max(0, Math.min(1, fraction || 0));
  return h("span", { class: "meter" }, h("span", { class: "meter-fill", style: { width: `${(f * 100).toFixed(1)}%`, background: color } }));
}

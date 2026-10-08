// Tendon configuration helpers (pure functions, no DOM).
//
// Everything here follows DESIGN.md "Configuration JSON" and "Site expansion rules".
// Functions never mutate their inputs; they return fresh objects.
//
// Point ids are catalog tap ids (e.g. "segment_6_plate_1_ghost_hm_tap_b") or
// free point ids ("free_1", stored in config.free_points).

export const CONFIG_FORMAT = "seahorse-tendon-config";
export const CONFIG_VERSION = 1;

/** Tendon colours, in the order new tendons get them (same order as the app shell). */
export const TENDON_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];

export const NAME_RE = /^[A-Za-z0-9_-]+$/;

export const ACTUATOR_DEFAULTS = {
  motor: { type: "motor", max_force: 10 },
  position: { type: "position", kp: 1000, max_force: 1000, max_strain: 0.26 },
  none: { type: "none" },
};

export const TENDON_DEFAULTS = { group: "", width: 0.001, damping: 0.01, stiffness: 0 };

/** Free points closer than this (metres) to a catalog tap snap onto the tap. */
export const SNAP_DISTANCE = 0.0015;

const CORNER_WORDS = {
  ventral_dextral: "ventral-dextral",
  ventral_sinistral: "ventral-sinistral",
  dorsal_sinistral: "dorsal-sinistral",
  dorsal_dextral: "dorsal-dextral",
};

export const deepCopy = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));
const isObj = (x) => x !== null && typeof x === "object" && !Array.isArray(x);
const sign = (x) => (x < 0 ? -1 : 1);

// ---------------------------------------------------------------------------
// New configs / tendons
// ---------------------------------------------------------------------------

export function newConfig(name = "My configuration") {
  return {
    format: CONFIG_FORMAT,
    version: CONFIG_VERSION,
    name,
    description: "",
    params: {},
    free_points: [],
    tendons: [],
  };
}

/** `base` if unused by the config's tendons, else base_2, base_3, ... */
export function uniqueTendonName(config, base = "tendon") {
  const names = new Set((config?.tendons || []).map((t) => t.name));
  const clean = sanitiseName(base) || "tendon";
  if (!names.has(clean)) return clean;
  const m = clean.match(/^(.*?)(?:_(\d+))?$/);
  const stem = m[1] || clean;
  let i = m[2] ? Number(m[2]) + 1 : 2;
  while (names.has(`${stem}_${i}`)) i++;
  return `${stem}_${i}`;
}

/** Make any text a valid tendon name ([A-Za-z0-9_-]+, not starting with "segment_"). */
export function sanitiseName(text) {
  let n = String(text ?? "").trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_-]/g, "");
  if (n.startsWith("segment_")) n = "t_" + n;
  return n;
}

/** First palette colour not used by any tendon yet. */
export function nextTendonColor(config) {
  const used = new Set((config?.tendons || []).map((t) => String(t.color || "").toLowerCase()));
  return TENDON_COLORS.find((c) => !used.has(c)) ||
    TENDON_COLORS[(config?.tendons || []).length % TENDON_COLORS.length];
}

/**
 * A new tendon object (not added to the config). Unique name, next palette colour,
 * motor actuator with 10 N. `overrides` replace any field.
 */
export function newTendon(config, overrides = {}) {
  const t = {
    name: uniqueTendonName(config, overrides.name || "tendon_1"),
    group: TENDON_DEFAULTS.group,
    color: nextTendonColor(config),
    width: TENDON_DEFAULTS.width,
    damping: TENDON_DEFAULTS.damping,
    stiffness: TENDON_DEFAULTS.stiffness,
    actuator: deepCopy(ACTUATOR_DEFAULTS.motor),
    path: [],
    branches: [],
  };
  const o = deepCopy(overrides) || {};
  delete o.name;
  Object.assign(t, o);
  if (o.actuator) t.actuator = normaliseActuator(o.actuator);
  return t;
}

/** Return a new config with `tendon` appended (name made unique). */
export function addTendon(config, tendon) {
  const c = deepCopy(config);
  const t = deepCopy(tendon);
  t.name = uniqueTendonName(c, t.name);
  c.tendons.push(t);
  return c;
}

// ---------------------------------------------------------------------------
// Normalisation / migration
// ---------------------------------------------------------------------------

function normaliseActuator(a) {
  if (typeof a === "string") a = { type: a };
  if (!isObj(a) || !a.type) return deepCopy(ACTUATOR_DEFAULTS.motor);
  const type = String(a.type).toLowerCase();
  if (type === "motor" || type === "force") {
    return { ...a, type: "motor", max_force: num(a.max_force ?? a.gear, ACTUATOR_DEFAULTS.motor.max_force) };
  }
  if (type === "position" || type === "p_control") {
    const d = ACTUATOR_DEFAULTS.position;
    return { ...a, type: "position", kp: num(a.kp, d.kp), max_force: num(a.max_force, d.max_force),
      max_strain: num(a.max_strain ?? a.strain, d.max_strain) };
  }
  if (type === "none" || type === "passive") return { type: "none" };
  return { ...a, type };
}

function num(v, fallback) {
  const n = typeof v === "string" ? Number(v) : v;
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Fill defaults and migrate older/looser shapes. Returns a fresh config; never throws for
 * mere missing fields. Throws a readable Error only if `raw` is not a config at all.
 * Migrations: tendon.points -> path; actuator as string; branch.from given as point id;
 * free point xy as {x, y}.
 */
export function normaliseConfig(raw) {
  if (!isObj(raw)) throw new Error("This file is not a tendon configuration (it is not a JSON object).");
  if (raw.format && raw.format !== CONFIG_FORMAT) {
    throw new Error(`This file has format "${raw.format}", but a tendon configuration ("${CONFIG_FORMAT}") was expected.`);
  }
  const c = deepCopy(raw);
  c.format = CONFIG_FORMAT;
  c.version = Number.isFinite(c.version) ? c.version : CONFIG_VERSION;
  c.name = typeof c.name === "string" && c.name.trim() ? c.name : "Untitled";
  c.description = typeof c.description === "string" ? c.description : "";
  c.params = isObj(c.params) ? c.params : {};
  c.free_points = (Array.isArray(c.free_points) ? c.free_points : []).filter(isObj).map((fp) => {
    const xy = Array.isArray(fp.xy) ? fp.xy : isObj(fp.xy) ? [fp.xy.x, fp.xy.y] : [0, 0];
    return { ...fp, id: String(fp.id), segment: num(fp.segment, 0), plate: num(fp.plate, 0), xy: [num(xy[0], 0), num(xy[1], 0)] };
  });
  const tendons = Array.isArray(c.tendons) ? c.tendons.filter(isObj) : [];
  const out = [];
  tendons.forEach((t0, i) => {
    const t = { ...t0 };
    if (!t.name) t.name = `tendon_${i + 1}`;
    t.group = typeof t.group === "string" ? t.group : TENDON_DEFAULTS.group;
    t.color = typeof t.color === "string" && t.color ? t.color : TENDON_COLORS[i % TENDON_COLORS.length];
    t.width = num(t.width, TENDON_DEFAULTS.width);
    t.damping = num(t.damping, TENDON_DEFAULTS.damping);
    t.stiffness = num(t.stiffness, TENDON_DEFAULTS.stiffness);
    t.actuator = normaliseActuator(t.actuator);
    if (!Array.isArray(t.path) && Array.isArray(t.points)) t.path = t.points;
    delete t.points;
    t.path = (Array.isArray(t.path) ? t.path : []).map(String);
    t.branches = (Array.isArray(t.branches) ? t.branches : []).filter(isObj).map((b) => {
      let from = b.from;
      if (typeof from === "string") {
        const k = t.path.indexOf(from);
        from = k >= 0 ? k : Number.isFinite(Number(from)) ? Number(from) : -1;
      }
      return { ...b, from, path: (Array.isArray(b.path) ? b.path : []).map(String) };
    });
    out.push(t);
  });
  c.tendons = out;
  return c;
}

// ---------------------------------------------------------------------------
// Catalog lookup and geometry
// ---------------------------------------------------------------------------

const tapIndexCache = new WeakMap();

/** id -> {id, segment, plate, corner, kind, label, xy, sites, body} for every catalog tap (cached). */
export function catalogIndex(catalog) {
  let idx = tapIndexCache.get(catalog);
  if (idx) return idx;
  idx = new Map();
  for (const seg of catalog?.segments || []) {
    for (const plate of seg.plates || []) {
      for (const tap of plate.taps || []) {
        idx.set(tap.id, {
          id: tap.id, segment: seg.index, plate: plate.index, corner: plate.corner,
          kind: tap.kind, label: tap.label, xy: tap.xy, sites: tap.sites, body: plate.body,
        });
      }
    }
  }
  tapIndexCache.set(catalog, idx);
  return idx;
}

export function getSegment(catalog, index) {
  return (catalog?.segments || []).find((s) => s.index === index) || null;
}

export function getPlate(catalog, segment, plate) {
  return getSegment(catalog, segment)?.plates.find((p) => p.index === plate) || null;
}

/** Point-in-polygon with the even-odd rule over all polygons (outer boundaries and holes). */
export function pointInPolygons(xy, polygons) {
  const [px, py] = xy;
  let inside = false;
  for (const poly of polygons || []) {
    const n = poly.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** Plate object of `segment` whose silhouette contains xy (segment frame, metres), or null. */
export function plateAt(catalog, segment, xy) {
  const seg = getSegment(catalog, segment);
  if (!seg) return null;
  return seg.plates.find((p) => pointInPolygons(xy, p.outline)) || null;
}

/** Nearest catalog tap in `segment` (optionally restricted to a plate / kind) within maxDist metres. */
export function nearestTap(catalog, segment, xy, { maxDist = Infinity, plate = null, kind = null } = {}) {
  const seg = getSegment(catalog, segment);
  if (!seg) return null;
  let best = null;
  let bestD = maxDist;
  for (const p of seg.plates) {
    if (plate !== null && p.index !== plate) continue;
    for (const t of p.taps) {
      if (kind && t.kind !== kind) continue;
      const d = Math.hypot(t.xy[0] - xy[0], t.xy[1] - xy[1]);
      if (d <= bestD) { bestD = d; best = { tap: t, plate: p.index, distance: d }; }
    }
  }
  return best;
}

function cornerWords(corner) { return CORNER_WORDS[corner] || corner || "plate"; }

/**
 * Info about a point id: {id, segment, plate, corner, kind, label, xy, free} or null if unknown.
 * kind is "intermediate" | "ghost" | "end" | "mvm" | "free".
 */
export function pointInfo(pointId, catalog, config) {
  const tap = catalogIndex(catalog).get(pointId);
  if (tap) {
    return { id: tap.id, segment: tap.segment, plate: tap.plate, corner: tap.corner, kind: tap.kind,
      label: tap.label, xy: tap.xy.slice(), free: false };
  }
  const fp = (config?.free_points || []).find((f) => f.id === pointId);
  if (fp) {
    const corner = catalog?.corners?.[fp.plate] ?? getPlate(catalog, fp.segment, fp.plate)?.corner;
    return { id: fp.id, segment: fp.segment, plate: fp.plate, corner, kind: "free", label: "free point",
      xy: fp.xy.slice(), free: true };
  }
  return null;
}

/** "segment 6, ventral-sinistral plate, midline hole b" */
export function describePoint(pointId, catalog, config) {
  const p = pointInfo(pointId, catalog, config);
  if (!p) return `unknown point "${pointId}"`;
  return `segment ${p.segment}, ${cornerWords(p.corner)} plate, ${p.label}`;
}

/** Short label used in summaries: "seg 6 (midline hole b)". */
function shortPoint(p) {
  if (!p) return "an unknown point";
  const label = p.kind === "end" ? "corner" : String(p.label).replace(/\s*\(([^)]*)\)/, ", $1");
  return `seg ${p.segment} (${label})`;
}

/** Segments visited by a tendon: {trunk: [seg...], branches: [[seg...]...], all: Set}. */
export function tendonSegments(tendon, catalog, config) {
  const segOf = (id) => pointInfo(id, catalog, config)?.segment;
  const trunk = (tendon?.path || []).map(segOf).filter((s) => s !== undefined);
  const branches = (tendon?.branches || []).map((b) => (b.path || []).map(segOf).filter((s) => s !== undefined));
  const all = new Set([...trunk, ...branches.flat()]);
  return { trunk, branches, all };
}

// ---------------------------------------------------------------------------
// Human summary
// ---------------------------------------------------------------------------

function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }

/**
 * One-line summary, e.g.
 * "starts seg 5 (midline hole b) → 3 via points → ends seg 9 (corner); branch at seg 8 → 1 via point → ends seg 10 (corner)"
 */
export function describeTendon(tendon, catalog, config) {
  const path = tendon?.path || [];
  if (!path.length) return "no points yet";
  const pts = path.map((id) => pointInfo(id, catalog, config));
  if (pts.length === 1) return `starts ${shortPoint(pts[0])}; no end point yet`;
  const parts = [`starts ${shortPoint(pts[0])}`];
  const via = pts.length - 2;
  if (via > 0) parts.push(plural(via, "via point"));
  parts.push(`ends ${shortPoint(pts[pts.length - 1])}`);
  let text = parts.join(" → ");
  for (const b of tendon.branches || []) {
    const bp = (b.path || []).map((id) => pointInfo(id, catalog, config));
    const at = pts[b.from];
    const head = `branch at ${at ? `seg ${at.segment}` : `point ${Number(b.from) + 1}`}`;
    if (!bp.length) { text += `; ${head} (empty)`; continue; }
    const bparts = [head];
    if (bp.length > 1) bparts.push(plural(bp.length - 1, "via point"));
    bparts.push(`ends ${shortPoint(bp[bp.length - 1])}`);
    text += "; " + bparts.join(" → ");
  }
  return text;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Plain-language validation. Returns [{level: "error"|"warning", tendon: name|null, message}].
 * Errors stop the model from being built; warnings are advice.
 */
export function validateConfig(config, catalog) {
  const out = [];
  const err = (tendon, message) => out.push({ level: "error", tendon, message });
  const warn = (tendon, message) => out.push({ level: "warning", tendon, message });
  if (!isObj(config)) return [{ level: "error", tendon: null, message: "There is no configuration loaded." }];
  if (config.format && config.format !== CONFIG_FORMAT) {
    err(null, `This is not a tendon configuration (format "${config.format}").`);
  }
  const n = catalog?.num_segments ?? (catalog?.segments || []).length;
  const tapIdx = catalogIndex(catalog);

  // Free points
  const freeIds = new Map();
  for (const fp of config.free_points || []) {
    const name = `Free point "${fp.id}"`;
    if (!fp.id || !NAME_RE.test(String(fp.id))) err(null, `${name}: ids can only use letters, digits, _ and -.`);
    if (freeIds.has(fp.id)) err(null, `Two free points share the id "${fp.id}". Each free point needs its own id.`);
    if (tapIdx.has(fp.id)) err(null, `${name} has the same id as a hole in the model. Rename the free point.`);
    freeIds.set(fp.id, fp);
    const plate = getPlate(catalog, fp.segment, fp.plate);
    if (!(Number.isInteger(fp.segment) && fp.segment >= 0 && fp.segment < n)) {
      err(null, `${name} is on segment ${fp.segment}, but the tail only has segments 0 to ${n - 1}.`);
    } else if (!plate) {
      err(null, `${name} is on plate ${fp.plate}; plates are numbered 0 to 3.`);
    } else if (!Array.isArray(fp.xy) || fp.xy.length !== 2 || !fp.xy.every(Number.isFinite)) {
      err(null, `${name} has no valid position.`);
    } else if (!pointInPolygons(fp.xy, plate.outline)) {
      warn(null, `${name} lies outside its plate (segment ${fp.segment}, ${cornerWords(plate.corner)}). Drag it back onto the plate.`);
    }
  }

  const known = (id) => tapIdx.has(id) || freeIds.has(id);
  const info = (id) => pointInfo(id, catalog, config);
  const counts = new Map();
  for (const t of config.tendons || []) counts.set(t.name, (counts.get(t.name) || 0) + 1);
  const usedFree = new Set();

  for (const t of config.tendons || []) {
    const label = `Tendon "${t.name || "(no name)"}"`;
    if (!t.name) err(t.name ?? null, "A tendon has no name. Give it a short name such as hm_dextral.");
    else if (!NAME_RE.test(t.name)) err(t.name, `${label}: names can only use letters, digits, _ and - (no spaces).`);
    else if (t.name.startsWith("segment_")) err(t.name, `${label}: names cannot start with "segment_" (the robot model uses that prefix).`);
    if (counts.get(t.name) > 1) err(t.name, `${label}: another tendon has the same name. Names must be unique.`);

    const path = Array.isArray(t.path) ? t.path : [];
    if (path.length === 0) err(t.name, `${label} has no points yet. Click a hole in the slice view to set where it starts.`);
    else if (path.length === 1) err(t.name, `${label} has a start but no end yet. Click a hole in another segment to add the end point.`);

    const checkPath = (ids, where) => {
      ids.forEach((id, i) => {
        if (freeIds.has(id)) usedFree.add(id);
        if (!known(id)) err(t.name, `${label}: ${where} point ${i + 1} ("${id}") does not exist on this model.`);
        else if (i > 0 && ids[i - 1] === id) warn(t.name, `${label}: ${where} point ${i + 1} is the same hole as the point before it.`);
      });
    };
    checkPath(path, "");

    // Direction / skipped segments along the trunk.
    const segs = path.map((id) => info(id)?.segment).filter((s) => s !== undefined);
    if (segs.length >= 2) {
      const D = sign(segs[segs.length - 1] - segs[0]);
      if (segs[0] === segs[segs.length - 1]) {
        warn(t.name, `${label} starts and ends in the same segment, so it crosses no joint and cannot bend the tail.`);
      }
      for (let i = 1; i < segs.length; i++) {
        const step = segs[i] - segs[i - 1];
        if (step * D < 0) {
          warn(t.name, `${label} turns back: point ${i + 1} (segment ${segs[i]}) is closer to the ${D > 0 ? "base" : "tip"} than point ${i} (segment ${segs[i - 1]}).`);
        } else if (Math.abs(step) > 1) {
          warn(t.name, `${label} jumps from segment ${segs[i - 1]} to ${segs[i]}. On the real robot it would have to pass through the plates in between; add a hole in each skipped segment.`);
        }
      }
    }

    (Array.isArray(t.branches) ? t.branches : []).forEach((b, bi) => {
      const bl = `branch ${bi + 1}`;
      if (!(Number.isInteger(b.from) && b.from >= 0 && b.from < path.length)) {
        err(t.name, `${label}: ${bl} splits at point ${Number(b.from) + 1}, but the main path only has ${path.length} points.`);
      }
      const bp = Array.isArray(b.path) ? b.path : [];
      if (bp.length < 1) err(t.name, `${label}: ${bl} has no points. Add at least an end point or remove the branch.`);
      checkPath(bp, `${bl}, `);
      if (Number.isInteger(b.from) && b.from === path.length - 1 && path.length >= 2) {
        warn(t.name, `${label}: ${bl} splits at the very end of the main path; it simply extends the tendon.`);
      }
    });

    const a = t.actuator || {};
    if (!["motor", "position", "none"].includes(a.type)) err(t.name, `${label}: unknown actuator type "${a.type}". Use force (motor), position or none.`);
    if (a.type === "motor" && !(a.max_force > 0)) err(t.name, `${label}: the maximum force must be larger than 0 N.`);
    if (a.type === "position") {
      if (!(a.kp > 0)) err(t.name, `${label}: the controller gain (kp) must be larger than 0.`);
      if (!(a.max_force > 0)) err(t.name, `${label}: the maximum force must be larger than 0 N.`);
      if (!(a.max_strain > 0 && a.max_strain < 1)) err(t.name, `${label}: the maximum shortening must be between 0 and 100 %.`);
    }
    if (!(t.stiffness >= 0)) err(t.name, `${label}: stiffness cannot be negative.`);
    if (!(t.damping >= 0)) err(t.name, `${label}: damping cannot be negative.`);
    if (!(t.width > 0)) warn(t.name, `${label}: the drawing width should be larger than 0.`);
    if (t.color && !/^#[0-9a-fA-F]{6}$/.test(t.color)) warn(t.name, `${label}: colour "${t.color}" is not a #rrggbb colour.`);
  }

  for (const id of freeIds.keys()) {
    if (!usedFree.has(id)) warn(null, `Free point "${id}" is not used by any tendon.`);
  }
  return out;
}

export const hasErrors = (issues) => issues.some((i) => i.level === "error");

// ---------------------------------------------------------------------------
// Free points
// ---------------------------------------------------------------------------

export function uniqueFreePointId(config, base = "free") {
  const ids = new Set((config?.free_points || []).map((f) => f.id));
  let i = 1;
  while (ids.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

/** Ids of free points not referenced by any tendon path or branch. */
export function unusedFreePoints(config) {
  const used = new Set();
  for (const t of config?.tendons || []) {
    for (const id of t.path || []) used.add(id);
    for (const b of t.branches || []) for (const id of b.path || []) used.add(id);
  }
  return (config?.free_points || []).map((f) => f.id).filter((id) => !used.has(id));
}

/** New config without free points that no tendon uses. */
export function pruneFreePoints(config) {
  const c = deepCopy(config);
  const unused = new Set(unusedFreePoints(c));
  c.free_points = (c.free_points || []).filter((f) => !unused.has(f.id));
  return c;
}

// ---------------------------------------------------------------------------
// Mirroring (sinistral <-> dextral)
// ---------------------------------------------------------------------------

const SIDE_SWAP = { dextral: "sinistral", sinistral: "dextral", Dextral: "Sinistral", Sinistral: "Dextral" };
function swapSides(text) {
  return String(text ?? "").replace(/dextral|sinistral|Dextral|Sinistral/g, (m) => SIDE_SWAP[m]);
}

/** Corner of the mirrored plate: ventral_dextral <-> ventral_sinistral, dorsal_* likewise. */
export function mirrorPlateIndex(catalog, plateIndex) {
  const corners = catalog?.corners || ["ventral_dextral", "ventral_sinistral", "dorsal_sinistral", "dorsal_dextral"];
  const m = corners.indexOf(swapSides(corners[plateIndex]));
  return m >= 0 ? m : plateIndex;
}

/**
 * Mirror one point (y -> -y). Taps map to the nearest tap of the same kind at the mirrored
 * position (within 3 mm, preferring the mirrored plate); free points become new free points.
 * Returns {id, freePoint|null, exact: bool}.
 */
export function mirrorPoint(pointId, catalog, config, takenIds = new Set()) {
  const p = pointInfo(pointId, catalog, config);
  if (!p) return { id: pointId, freePoint: null, exact: false };
  const target = [p.xy[0], -p.xy[1]];
  const mPlate = mirrorPlateIndex(catalog, p.plate);
  if (!p.free) {
    const tol = 0.003;
    const hit = nearestTap(catalog, p.segment, target, { maxDist: tol, plate: mPlate, kind: p.kind }) ||
      nearestTap(catalog, p.segment, target, { maxDist: tol, kind: p.kind });
    if (hit) return { id: hit.tap.id, freePoint: null, exact: hit.distance < 0.0012 };
  }
  // Free point (or a tap without a mirrored partner): create a free point at the mirrored spot.
  const plate = plateAt(catalog, p.segment, target);
  const ids = new Set([...(config?.free_points || []).map((f) => f.id), ...takenIds]);
  let i = 1;
  while (ids.has(`free_${i}`)) i++;
  const fp = { id: `free_${i}`, segment: p.segment, plate: plate ? plate.index : mPlate, xy: target };
  return { id: fp.id, freePoint: fp, exact: !!plate };
}

/**
 * Mirror a tendon sinistral <-> dextral.
 * Returns {tendon, free_points: [new free points to add], notes: [plain-language strings]}.
 * The mirrored tendon's name/group swap "dextral"/"sinistral" (or get a "_mirrored" suffix),
 * and its name is unique in `config`.
 */
export function mirrorTendon(tendon, catalog, config) {
  const t = deepCopy(tendon);
  const newFree = [];
  const notes = [];
  const taken = new Set();
  const cache = new Map();
  const map = (id) => {
    if (cache.has(id)) return cache.get(id);
    const r = mirrorPoint(id, catalog, { ...config, free_points: [...(config?.free_points || []), ...newFree] }, taken);
    if (r.freePoint) {
      newFree.push(r.freePoint);
      taken.add(r.freePoint.id);
      if (!pointInfo(id, catalog, config)?.free) notes.push(`No mirrored hole for ${describePoint(id, catalog, config)}; a free point was placed instead.`);
    }
    cache.set(id, r.id);
    return r.id;
  };
  t.path = (t.path || []).map(map);
  t.branches = (t.branches || []).map((b) => ({ ...b, path: (b.path || []).map(map) }));
  const swapped = swapSides(tendon.name);
  t.name = uniqueTendonName(config, swapped !== tendon.name ? swapped : `${tendon.name}_mirrored`);
  t.group = swapSides(tendon.group || "");
  const used = new Set((config?.tendons || []).map((x) => String(x.color).toLowerCase()));
  if (used.has(String(t.color).toLowerCase())) t.color = nextTendonColor(config);
  return { tendon: t, free_points: newFree, notes };
}

/** New config with the mirror image of tendon `name` added. Returns {config, name, notes}. */
export function addMirroredTendon(config, name, catalog) {
  const src = (config.tendons || []).find((t) => t.name === name);
  if (!src) throw new Error(`There is no tendon called "${name}".`);
  const r = mirrorTendon(src, catalog, config);
  const c = deepCopy(config);
  c.free_points = [...(c.free_points || []), ...r.free_points];
  c.tendons.push(r.tendon);
  return { config: c, name: r.tendon.name, notes: r.notes };
}

// ---------------------------------------------------------------------------
// Path editing helpers (used by the slice editor; pure)
// ---------------------------------------------------------------------------

/**
 * Remove trunk point k from a tendon; branch indices are shifted, branches that split at
 * the removed point move to the previous point (or are dropped if it was the first).
 */
export function removeTrunkPoint(tendon, k) {
  const t = deepCopy(tendon);
  t.path.splice(k, 1);
  t.branches = (t.branches || []).flatMap((b) => {
    if (b.from === k) return k > 0 ? [{ ...b, from: k - 1 }] : t.path.length ? [{ ...b, from: 0 }] : [];
    return [{ ...b, from: b.from > k ? b.from - 1 : b.from }];
  });
  return t;
}

/** Move trunk point k by delta (-1 earlier, +1 later); branch split indices follow their point. */
export function moveTrunkPoint(tendon, k, delta) {
  const t = deepCopy(tendon);
  const j = k + delta;
  if (j < 0 || j >= t.path.length) return t;
  [t.path[k], t.path[j]] = [t.path[j], t.path[k]];
  t.branches = (t.branches || []).map((b) => ({ ...b, from: b.from === k ? j : b.from === j ? k : b.from }));
  return t;
}

// ---------------------------------------------------------------------------
// Presets (routing generators reproducing the paper's routing)
// ---------------------------------------------------------------------------

/** Python's round(): round half to even. */
function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}

/** Port of experiments/utils.py select_symmetric_points(n): hole indices 0..9, base -> tip. */
export function selectSymmetricPoints(n) {
  const lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const mid = Math.floor(lst.length / 2);
  if (n === 1) return [lst[mid]];
  const out = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.max(0, Math.min(lst.length - 1, roundHalfEven(mid + ((i - (n - 1) / 2) * lst.length) / (n - 1))));
    out.push(lst[idx]);
  }
  return out.reverse();
}

function requireTap(catalog, id) {
  if (!catalogIndex(catalog).has(id)) throw new Error(`The model has no hole "${id}".`);
  return id;
}

/** Corner plate index for e.g. ("ventral", "dextral"). */
export function cornerPlate(catalog, sagittal, coronal) {
  const i = (catalog.corners || []).indexOf(`${sagittal}_${coronal}`);
  if (i < 0) throw new Error(`Unknown corner ${sagittal}_${coronal}.`);
  return i;
}

/** Midline ("ghost") start tap on the given sagittal side whose y lies on the coronal side. */
export function ghostTap(catalog, segment, sagittal, coronal) {
  const seg = getSegment(catalog, segment);
  const wantX = sagittal === "ventral" ? 1 : -1;
  const wantY = coronal === "sinistral" ? 1 : -1;
  for (const p of seg?.plates || []) {
    for (const t of p.taps) {
      if (t.kind === "ghost" && Math.sign(t.xy[0]) === wantX && Math.sign(t.xy[1]) === wantY) return t.id;
    }
  }
  throw new Error(`Segment ${segment} has no ${sagittal} midline hole on the ${coronal} side.`);
}

export const intermediateTap = (catalog, segment, plate, hole) =>
  requireTap(catalog, `segment_${segment}_plate_${plate}_intermediate_hm_tap_${hole}`);
export const endTap = (catalog, segment, plate) => requireTap(catalog, `segment_${segment}_plate_${plate}_end_hm_tap`);
export const mvmTap = (catalog, segment, coronal) => {
  const hit = catalogIndex(catalog);
  for (const tap of hit.values()) if (tap.segment === segment && tap.kind === "mvm" && tap.id.endsWith(`_mvm_tap_${coronal}`)) return tap.id;
  throw new Error(`Segment ${segment} has no MVM hole on the ${coronal} side.`);
};

/**
 * Point path of a paper-style HM: midline (ghost) tap in `start`, intermediate holes
 * `holes[i]` in segments start+1 .. end-1 on the corner plate, corner end tap in `end`.
 */
export function hmPath(catalog, { start, end, holes = null, sagittal = "ventral", coronal = "dextral" }) {
  const plate = cornerPlate(catalog, sagittal, coronal);
  const span = end - start;
  const h = holes || selectSymmetricPoints(span - 1);
  if (h.length !== span - 1) throw new Error(`An HM from segment ${start} to ${end} needs ${span - 1} hole indices (got ${h.length}).`);
  const path = [ghostTap(catalog, start, sagittal, coronal)];
  for (let s = start + 1; s < end; s++) path.push(intermediateTap(catalog, s, plate, h[s - start - 1]));
  path.push(endTap(catalog, end, plate));
  return path;
}

const HM_STYLE = {
  dextral: { color: "#2a78d6" },
  sinistral: { color: "#eb6834" },
};
const MVM_STYLE = {
  dextral: { color: "#4a3aa7" },
  sinistral: { color: "#e87ba4" },
};

/**
 * Symmetric HM pair (dextral + sinistral) as tendon objects. Optional fork:
 * `branch = {fromSegment, holes: [hole per segment after fromSegment], end}` continues from the
 * trunk point in `fromSegment` through those holes to the corner end tap in segment `end`.
 */
export function hmPair(catalog, { start, end, holes = null, sagittal = "ventral", branch = null, prefix = "hm" }) {
  return ["dextral", "sinistral"].map((coronal) => {
    const path = hmPath(catalog, { start, end, holes, sagittal, coronal });
    const branches = [];
    if (branch) {
      const plate = cornerPlate(catalog, sagittal, coronal);
      const from = branch.fromSegment - start;
      const bpath = [];
      for (let s = branch.fromSegment + 1; s < branch.end; s++) {
        bpath.push(intermediateTap(catalog, s, plate, branch.holes[s - branch.fromSegment - 1]));
      }
      bpath.push(endTap(catalog, branch.end, plate));
      branches.push({ from, path: bpath });
    }
    return {
      name: `${prefix}_${coronal}`, group: coronal, color: HM_STYLE[coronal].color,
      width: 0.001, damping: 0.01, stiffness: 0,
      actuator: { type: "motor", max_force: 10 },
      path, branches,
    };
  });
}

/**
 * The paper's MVMs: one short position-controlled tendon per adjacent segment pair and side,
 * through the MVM holes (ventral-sinistral plate of every segment).
 */
export function mvmChain(catalog, { first = 0, last = null, maxStrain = 0.26, kp = 1000, maxForce = 1000 } = {}) {
  const n = catalog.num_segments ?? catalog.segments.length;
  const lastSeg = last ?? n - 1;
  const out = [];
  for (const coronal of ["dextral", "sinistral"]) {
    for (let s = first; s < lastSeg; s++) {
      out.push({
        name: `mvm_${coronal}_${s}_${s + 1}`, group: `mvm_${coronal}`, color: MVM_STYLE[coronal].color,
        width: 0.001, damping: 1, stiffness: 0,
        actuator: { type: "position", kp, max_force: maxForce, max_strain: maxStrain },
        path: [mvmTap(catalog, s, coronal), mvmTap(catalog, s + 1, coronal)], branches: [],
      });
    }
  }
  return out;
}

function presetConfig(name, description, tendons) {
  return { ...newConfig(name), description, tendons };
}

/** Built-in presets. `build(catalog)` returns a full config. File names match web/configs/. */
export const PRESETS = [
  {
    file: "empty.json", name: "Empty",
    description: "No tendons. Start from scratch.",
    build: () => presetConfig("Empty", "No tendons. Start from scratch.", []),
  },
  {
    file: "hm_span4_paper.json", name: "HM span 4 (paper)",
    description: "Two symmetric ventral HMs from segment 6 to the tip (segment 10), routed as in the paper.",
    build: (c) => presetConfig("HM span 4 (paper)",
      "Two symmetric ventral HMs from segment 6 to segment 10 (tip): midline start hole, holes 9, 5, 0 in segments 7-9, corner anchor at the tip. Same routing as the paper (hm_segment_span = 4).",
      hmPair(c, { start: 6, end: 10, holes: selectSymmetricPoints(3) })),
  },
  {
    file: "hm_fork_5_9_10.json", name: "HM fork 5→9 + 10",
    description: "Ventral HMs from segment 5 to 9 that fork at segment 8 into a second end at the tip.",
    build: (c) => presetConfig("HM fork 5→9 + 10",
      "Ventral HM pair from segment 5 to segment 9 (holes 9, 5, 0 in segments 6-8). At segment 8 each HM splits: the branch passes hole 0 of segment 9 and ends at the corner of segment 10. One actuator pulls both ends.",
      hmPair(c, { start: 5, end: 9, holes: selectSymmetricPoints(3), branch: { fromSegment: 8, holes: [0], end: 10 } })),
  },
  {
    file: "hm_span10_full.json", name: "HM span 10 (full length)",
    description: "Ventral HMs along the whole tail, from segment 0 to the tip.",
    build: (c) => presetConfig("HM span 10 (full length)",
      "Two symmetric ventral HMs from segment 0 to segment 10, through one hole in every segment in between (select_symmetric_points(9)).",
      hmPair(c, { start: 0, end: 10, holes: selectSymmetricPoints(9) })),
  },
  {
    file: "mvm_chain.json", name: "MVM chain",
    description: "The paper's MVMs: one short muscle per joint and side, position-controlled (26 % shortening).",
    build: (c) => presetConfig("MVM chain",
      "One short MVM per pair of neighbouring segments on each side (20 in total), through the MVM holes. Position control: kp 1000, max force 1000 N, up to 26 % shortening.",
      mvmChain(c)),
  },
  {
    file: "hm_span4_mvm.json", name: "HM span 4 + MVMs",
    description: "The paper's HM pair (segment 6 to 10) together with the full MVM chain.",
    build: (c) => presetConfig("HM span 4 + MVMs",
      "Ventral HM pair from segment 6 to 10 (paper routing) plus one MVM per joint and side.",
      [...hmPair(c, { start: 6, end: 10, holes: selectSymmetricPoints(3) }), ...mvmChain(c)]),
  },
];

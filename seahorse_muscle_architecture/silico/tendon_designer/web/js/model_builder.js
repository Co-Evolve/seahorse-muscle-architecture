// model_builder.js — turn (base.xml, catalog.json, config) into a MuJoCo MJCF string.
//
// This is the JavaScript half of the "config -> MJCF" contract in DESIGN.md. The Python
// side (config.py) must produce the same sites/pulleys for the same config; keep the
// two in sync when changing anything here.
//
// The XML is edited with the standard DOMParser / XMLSerializer. In the browser these
// exist globally. In Node (tests) assign them to globalThis first, e.g. from
// @xmldom/xmldom, or call setXmlImplementation({DOMParser, XMLSerializer}).

/** Allowed tendon / free-point name characters (also used for MuJoCo element names). */
export const NAME_PATTERN = /^[A-Za-z0-9_-]+$/;

let xmlImpl = null;

/**
 * Override the DOMParser/XMLSerializer implementation (for Node tests).
 * @param {{DOMParser: Function, XMLSerializer: Function}|null} impl  null = use globals.
 */
export function setXmlImplementation(impl) {
  xmlImpl = impl;
}

function getXml() {
  const impl = xmlImpl || globalThis;
  if (!impl.DOMParser || !impl.XMLSerializer) {
    throw new Error("model_builder: no DOMParser/XMLSerializer available " +
      "(in Node, call setXmlImplementation() with @xmldom/xmldom)");
  }
  return impl;
}

// ---------------------------------------------------------------------------
// Catalog lookups
// ---------------------------------------------------------------------------

const catalogIndexCache = new WeakMap();

/**
 * Index of all catalog taps: id -> {id, segment, plate, kind, sites, xy, body}.
 * Cached per catalog object.
 * @param {object} catalog  parsed catalog.json
 * @returns {Map<string, object>}
 */
export function catalogTapIndex(catalog) {
  let index = catalogIndexCache.get(catalog);
  if (index) return index;
  index = new Map();
  for (const seg of catalog.segments) {
    for (const plate of seg.plates) {
      for (const tap of plate.taps) {
        index.set(tap.id, {
          id: tap.id, segment: seg.index, plate: plate.index, kind: tap.kind,
          sites: tap.sites, xy: tap.xy, body: plate.body,
        });
      }
    }
  }
  catalogIndexCache.set(catalog, index);
  return index;
}

function findPlate(catalog, segment, plate) {
  const seg = catalog.segments.find((s) => s.index === segment);
  if (!seg) return null;
  return seg.plates.find((p) => p.index === plate) || null;
}

/**
 * Resolve a point id (catalog tap id or free point id) to {segment, sites}.
 * @returns {{id: string, segment: number, sites: string[]}}
 */
function resolvePoint(id, catalog, freePoints, tendonName) {
  const tap = catalogTapIndex(catalog).get(id);
  if (tap) return { id, segment: tap.segment, sites: tap.sites };
  const fp = (freePoints || []).find((f) => f.id === id);
  if (fp) return { id, segment: fp.segment, sites: [`${id}_0`, `${id}_1`] };
  throw new Error(`Tendon "${tendonName}": unknown point "${id}" ` +
    "(not a catalog tap and not a free point)");
}

const signOrPlus = (x) => (x < 0 ? -1 : 1);

// ---------------------------------------------------------------------------
// Site expansion (DESIGN.md "Site expansion rules")
// ---------------------------------------------------------------------------

/**
 * Expand a tendon's point path (+ branches) into the ordered MuJoCo spatial-tendon
 * path: an array of `{site: name}` and `{pulley: 1}` entries.
 *
 * Rules (DESIGN.md): the trunk is oriented by D = sign(seg(last) - seg(first)) (+1 if 0);
 * each point contributes its sites in that order. A branch {from: k, path} emits a pulley,
 * the exit site of trunk point k (its last site in branch order), then the branch points'
 * sites in branch order Db = sign(seg(Q_m) - seg(P_k)). Consecutive duplicate sites are dropped.
 *
 * @param {object} tendon      config tendon ({name, path, branches})
 * @param {object} catalog     parsed catalog.json
 * @param {Array}  freePoints  config.free_points (may be undefined)
 * @returns {Array<{site: string}|{pulley: 1}>}
 */
export function expandTendonSites(tendon, catalog, freePoints) {
  const name = tendon.name;
  const path = tendon.path || [];
  if (path.length < 2) {
    throw new Error(`Tendon "${name}": the path needs at least 2 points (has ${path.length})`);
  }
  const trunk = path.map((id) => resolvePoint(id, catalog, freePoints, name));
  const out = [];
  let prev = null;
  const emit = (site) => {
    if (site === prev) return; // rule 4
    out.push({ site });
    prev = site;
  };
  const ordered = (sites, dir) => (dir > 0 ? sites : sites.slice().reverse());

  const D = signOrPlus(trunk[trunk.length - 1].segment - trunk[0].segment);
  for (const p of trunk) for (const s of ordered(p.sites, D)) emit(s);

  (tendon.branches || []).forEach((branch, bi) => {
    const k = branch.from;
    if (!Number.isInteger(k) || k < 0 || k >= trunk.length) {
      throw new Error(`Tendon "${name}", branch ${bi + 1}: "from" must be a trunk point ` +
        `index 0..${trunk.length - 1} (got ${JSON.stringify(k)})`);
    }
    const bpath = branch.path || [];
    if (bpath.length < 1) {
      throw new Error(`Tendon "${name}", branch ${bi + 1}: the branch path needs at least 1 point`);
    }
    const pts = bpath.map((id) => resolvePoint(id, catalog, freePoints, name));
    const Pk = trunk[k];
    const Db = signOrPlus(pts[pts.length - 1].segment - Pk.segment);
    out.push({ pulley: 1 });
    prev = null; // rule 4 applies within a branch
    const exitSites = ordered(Pk.sites, Db);
    emit(exitSites[exitSites.length - 1]);
    for (const q of pts) for (const s of ordered(q.sites, Db)) emit(s);
  });
  return out;
}

// ---------------------------------------------------------------------------
// MJCF generation
// ---------------------------------------------------------------------------

/** "#rgb" | "#rrggbb" | "#rrggbbaa" -> "r g b a" (0..1). Unknown -> grey. */
export function colorToRgba(color) {
  let hex = String(color || "").trim().replace(/^#/, "");
  if (/^[0-9a-f]{3}$/i.test(hex)) hex = hex.split("").map((c) => c + c).join("");
  if (!/^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(hex)) return "0.5 0.5 0.5 1";
  const v = [];
  for (let i = 0; i < hex.length; i += 2) v.push(parseInt(hex.slice(i, i + 2), 16) / 255);
  if (v.length === 3) v.push(1);
  return v.map((x) => +x.toFixed(4)).join(" ");
}

const num = (x) => String(+(+x).toPrecision(12));

function childElements(el, tag) {
  const res = [];
  for (let c = el.firstChild; c; c = c.nextSibling) {
    if (c.nodeType === 1 && (!tag || c.nodeName === tag)) res.push(c);
  }
  return res;
}

function getOrCreateSection(doc, root, tag) {
  const existing = childElements(root, tag)[0];
  if (existing) return existing;
  const el = doc.createElement(tag);
  root.appendChild(el);
  return el;
}

/** Map "tag:name" -> element for every named element in the document. */
function indexNamedElements(doc) {
  const map = new Map();
  const all = doc.getElementsByTagName("*");
  for (let i = 0; i < all.length; i++) {
    const el = all[i];
    const n = el.getAttribute("name");
    if (n) map.set(`${el.nodeName}:${n}`, el);
  }
  return map;
}

function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) el.setAttribute(k, String(v));
  }
}

const isSet = (v) => v !== undefined && v !== null;

function applyParams(doc, root, named, catalog, params) {
  if (!params) return;
  const n = catalog.num_segments;

  // --- vertebral joints --------------------------------------------------
  const vert = params.vertebra || {};
  const taper = isSet(params.stiffness_taper) ? +params.stiffness_taper : 1.0;
  for (let i = 1; i < n; i++) {
    // linear 1.0 at segment 1 -> taper at segment n-1
    const f = n > 2 ? 1 + (taper - 1) * (i - 1) / (n - 2) : taper;
    for (const axis of ["pitch", "roll", "yaw"]) {
      const j = named.get(`joint:segment_${i}_vertebrae_vertebrae_joint_${axis}`);
      if (!j) continue;
      const k = vert[`${axis}_stiffness`];
      if (isSet(k) || taper !== 1) {
        const base = isSet(k) ? +k
          : (j.hasAttribute("stiffness") ? +j.getAttribute("stiffness")
            : +(catalog.defaults?.vertebra?.[`${axis}_stiffness`] ?? 0));
        j.setAttribute("stiffness", num(base * f));
      }
      const d = vert[`${axis}_damping`];
      if (isSet(d)) j.setAttribute("damping", num(d));
      const r = vert[`${axis}_range_deg`];
      if (isSet(r)) {
        const rad = Math.abs(+r) * Math.PI / 180;
        j.setAttribute("range", `${num(-rad)} ${num(rad)}`);
        j.setAttribute("limited", "true");
      }
    }
  }

  // --- plate glide joints & struts ----------------------------------------
  const glide = params.plate_glide || {};
  const strut = params.strut || {};
  const glideRe = /^segment_\d+_plate_\d+_[xy]_axis$/;
  const strutRe = /^segment_\d+_vertebral_vertebral_strut_/;
  for (const [key, el] of named) {
    const [tag, name] = [key.slice(0, key.indexOf(":")), key.slice(key.indexOf(":") + 1)];
    if (tag === "joint" && glideRe.test(name)) {
      if (isSet(glide.stiffness)) el.setAttribute("stiffness", num(glide.stiffness));
      if (isSet(glide.damping)) el.setAttribute("damping", num(glide.damping));
    } else if (tag === "spatial" && strutRe.test(name)) {
      if (isSet(strut.stiffness)) el.setAttribute("stiffness", num(strut.stiffness));
      if (isSet(strut.damping)) el.setAttribute("damping", num(strut.damping));
    }
  }

  // --- orientation -----------------------------------------------------------
  if (isSet(params.orientation)) {
    const b0 = named.get("body:segment_0");
    if (!b0) throw new Error('params.orientation: body "segment_0" not found in base model');
    if (params.orientation !== "upright" && params.orientation !== "hanging") {
      throw new Error(`params.orientation must be "upright" or "hanging" (got "${params.orientation}")`);
    }
    for (const a of ["quat", "axisangle", "xyaxes", "zaxis"]) b0.removeAttribute(a);
    b0.setAttribute("euler", params.orientation === "hanging" ? "3.141592653589793 0 0" : "0 0 0");
  }

  // --- option: gravity / timestep ---------------------------------------
  if (isSet(params.gravity) || isSet(params.timestep)) {
    const option = getOrCreateSection(doc, root, "option");
    if (isSet(params.timestep)) {
      if (!(+params.timestep > 0)) throw new Error("params.timestep must be > 0");
      option.setAttribute("timestep", num(params.timestep));
    }
    if (isSet(params.gravity)) {
      const flag = getOrCreateSection(doc, option, "flag");
      flag.setAttribute("gravity", params.gravity ? "enable" : "disable");
    }
  }
}

/**
 * Build the full MJCF for a configuration: base model + free-point sites + tendons,
 * actuators, sensors and parameter overrides (DESIGN.md "Generated MJCF" / "params").
 *
 * @param {string} baseXml   contents of web/model/base.xml
 * @param {object} catalog   parsed web/model/catalog.json
 * @param {object} config    tendon configuration (DESIGN.md "Configuration JSON")
 * @param {{shareMeshes?: boolean}} [options]  shareMeshes: merge <mesh> assets that load the
 *        same file with the same attributes into one asset and point all geoms at it.
 *        Physics are identical; the compiled model is ~10x smaller. Used by sim.js; off by
 *        default so the XML matches the Python builder.
 * @returns {string} MJCF XML string, ready for MjModel.from_xml_string
 * @throws {Error} with a readable message if the config references unknown points etc.
 */
export function buildModelXml(baseXml, catalog, config, { shareMeshes = false } = {}) {
  const { DOMParser, XMLSerializer } = getXml();
  const doc = new DOMParser().parseFromString(baseXml, "text/xml");
  const root = doc.documentElement;
  if (!root || root.nodeName !== "mujoco" || doc.getElementsByTagName("parsererror").length) {
    throw new Error("base.xml is not a valid MJCF document");
  }
  const named = indexNamedElements(doc);
  config = config || {};

  // --- free points: two sites on the plate body ---------------------------
  const freeIds = new Set();
  for (const fp of config.free_points || []) {
    if (!NAME_PATTERN.test(fp.id || "")) throw new Error(`Free point id "${fp.id}" is not a valid name`);
    if (freeIds.has(fp.id) || catalogTapIndex(catalog).has(fp.id)) {
      throw new Error(`Free point id "${fp.id}" is used twice`);
    }
    freeIds.add(fp.id);
    const plate = findPlate(catalog, fp.segment, fp.plate);
    if (!plate) throw new Error(`Free point "${fp.id}": no plate ${fp.plate} in segment ${fp.segment}`);
    const body = named.get(`body:${plate.body}`);
    if (!body) throw new Error(`Free point "${fp.id}": body "${plate.body}" not in base model`);
    const [x, y] = fp.xy;
    plate.free_site_z.forEach((z, k) => {
      const siteName = `${fp.id}_${k}`;
      if (named.has(`site:${siteName}`)) throw new Error(`Free point "${fp.id}": site "${siteName}" already exists`);
      const site = doc.createElement("site");
      setAttrs(site, {
        name: siteName, type: "sphere", size: "0.0005", rgba: "0.95 0.6 0.1 1",
        pos: `${num(x)} ${num(y)} ${num(z)}`,
      });
      body.appendChild(site);
      named.set(`site:${siteName}`, site);
    });
  }

  // --- tendons -----------------------------------------------------------------
  const tendons = config.tendons || [];
  if (tendons.length) {
    const tendonSec = getOrCreateSection(doc, root, "tendon");
    const seen = new Set();
    let actuatorSec = null;
    const sensorSec = getOrCreateSection(doc, root, "sensor");
    for (const t of tendons) {
      const N = t.name;
      if (!NAME_PATTERN.test(N || "")) throw new Error(`Tendon name "${N}" may only contain letters, digits, "_" and "-"`);
      if (N.startsWith("segment_")) throw new Error(`Tendon name "${N}" must not start with "segment_"`);
      if (seen.has(N) || named.has(`spatial:${N}`) || named.has(`fixed:${N}`)) {
        throw new Error(`Tendon name "${N}" is used twice`);
      }
      seen.add(N);

      const spatial = doc.createElement("spatial");
      setAttrs(spatial, {
        name: N, width: num(t.width ?? 0.001), rgba: colorToRgba(t.color),
        damping: num(t.damping ?? 0), stiffness: num(t.stiffness ?? 0), springlength: "-1 -1",
      });
      for (const item of expandTendonSites(t, catalog, config.free_points)) {
        const el = doc.createElement(item.site ? "site" : "pulley");
        if (item.site) {
          if (!named.has(`site:${item.site}`)) throw new Error(`Tendon "${N}": site "${item.site}" does not exist in the model`);
          el.setAttribute("site", item.site);
        } else {
          el.setAttribute("divisor", "1");
        }
        spatial.appendChild(el);
      }
      tendonSec.appendChild(spatial);

      const act = t.actuator || { type: "none" };
      let actuated = false;
      if (act.type === "motor" || act.type === "position") {
        actuatorSec = actuatorSec || getOrCreateSection(doc, root, "actuator");
        const F = +(act.max_force ?? (act.type === "motor" ? 10 : 1000));
        if (!(F > 0)) throw new Error(`Tendon "${N}": actuator max_force must be > 0`);
        const el = doc.createElement(act.type);
        if (act.type === "motor") {
          setAttrs(el, {
            name: N, tendon: N, gear: num(F), ctrllimited: "true", ctrlrange: "-1 0",
            forcelimited: "true", forcerange: `${num(-F)} 0`,
          });
        } else {
          setAttrs(el, {
            name: N, tendon: N, kp: num(act.kp ?? 1000), ctrllimited: "false",
            forcelimited: "true", forcerange: `${num(-F)} 0`,
          });
        }
        actuatorSec.appendChild(el);
        actuated = true;
      } else if (act.type !== "none") {
        throw new Error(`Tendon "${N}": unknown actuator type "${act.type}"`);
      }

      const tp = doc.createElement("tendonpos");
      setAttrs(tp, { name: `${N}_length`, tendon: N });
      sensorSec.appendChild(tp);
      if (actuated) {
        const af = doc.createElement("actuatorfrc");
        setAttrs(af, { name: `${N}_force`, actuator: N });
        sensorSec.appendChild(af);
      }
    }
  }

  applyParams(doc, root, named, catalog, config.params);
  if (shareMeshes) shareMeshAssets(doc);

  return new XMLSerializer().serializeToString(doc);
}

/** Merge duplicate <mesh> assets (same attributes except name) and retarget geoms. */
function shareMeshAssets(doc) {
  const canonical = new Map(); // attribute signature -> name
  const rename = new Map();    // dropped name -> kept name
  for (const mesh of Array.from(doc.getElementsByTagName("mesh"))) {
    const name = mesh.getAttribute("name");
    if (!name || !mesh.getAttribute("file")) continue;
    const attrs = [];
    for (let i = 0; i < mesh.attributes.length; i++) {
      const a = mesh.attributes[i];
      if (a.name !== "name") attrs.push(`${a.name}=${a.value}`);
    }
    const key = attrs.sort().join("|");
    const kept = canonical.get(key);
    if (kept) {
      rename.set(name, kept);
      mesh.parentNode.removeChild(mesh);
    } else {
      canonical.set(key, name);
    }
  }
  if (!rename.size) return;
  for (const tag of ["geom", "site", "default"]) {
    for (const el of Array.from(doc.getElementsByTagName(tag))) {
      const m = el.getAttribute("mesh");
      if (m && rename.has(m)) el.setAttribute("mesh", rename.get(m));
    }
  }
}

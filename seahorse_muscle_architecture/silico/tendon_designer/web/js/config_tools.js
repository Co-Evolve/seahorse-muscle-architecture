// UI-only config helpers. config.js is the single source of truth for the config schema,
// defaults, palette and validation; this module only adds a few helpers the UI shell needs.

import { catalogIndex } from "./config.js";

export { TENDON_COLORS, NAME_RE, ACTUATOR_DEFAULTS } from "./config.js";

/** True if the tendon has a motor or position actuator (gets an activation slider). */
export function isActuated(t) { return !!t?.actuator?.type && t.actuator.type !== "none"; }

/** Group names of actuated tendons, in config order ("(no group)" for an empty group). */
export function actuatedGroups(config) {
  const g = [];
  for (const t of config?.tendons || []) if (isActuated(t)) { const k = t.group || "(no group)"; if (!g.includes(k)) g.push(k); }
  return g;
}

/** Map point id -> {segment, plate, corner, kind, label, xy} for catalog taps and free points. */
export function pointIndex(catalog, freePoints = []) {
  const idx = new Map(catalogIndex(catalog));
  for (const fp of freePoints || []) {
    idx.set(fp.id, { id: fp.id, segment: fp.segment, plate: fp.plate, corner: catalog?.corners?.[fp.plate], kind: "free", label: "free point", xy: fp.xy });
  }
  return idx;
}

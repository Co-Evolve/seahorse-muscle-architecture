// Shared helpers for the Node tests: XML polyfill, asset loading from disk, MuJoCo.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DOMParser, XMLSerializer } from "@xmldom/xmldom";
import { setXmlImplementation } from "../../web/js/model_builder.js";

setXmlImplementation({ DOMParser, XMLSerializer });

const here = path.dirname(fileURLToPath(import.meta.url));
export const WEB_DIR = path.resolve(here, "../../web");
export const MODEL_DIR = path.join(WEB_DIR, "model");

/** Same shape as sim.js loadAssets(), but read from disk. */
export function loadAssetsFromDisk(dir = MODEL_DIR) {
  const baseXml = fs.readFileSync(path.join(dir, "base.xml"), "utf8");
  const catalog = JSON.parse(fs.readFileSync(path.join(dir, "catalog.json"), "utf8"));
  const meshes = {};
  for (const f of catalog.mesh_files) meshes[f] = new Uint8Array(fs.readFileSync(path.join(dir, f)));
  return { baseXml, catalog, meshes };
}

let cachedAssets = null;
export function assets() {
  if (!cachedAssets) cachedAssets = loadAssetsFromDisk();
  return cachedAssets;
}

/** The HM fork from mjcf_export_hm_fork (dextral side) written as a config tendon. */
export function hmForkTendon(overrides = {}) {
  return {
    name: "hm_dextral", group: "dextral", color: "#3b82f6", width: 0.001, damping: 0.01,
    stiffness: 0, actuator: { type: "motor", max_force: 10 },
    path: [
      "segment_5_plate_1_ghost_hm_tap_b",
      "segment_6_plate_0_intermediate_hm_tap_9",
      "segment_7_plate_0_intermediate_hm_tap_5",
      "segment_8_plate_0_intermediate_hm_tap_0",
      "segment_9_plate_0_end_hm_tap",
    ],
    branches: [{ from: 3, path: ["segment_9_plate_0_intermediate_hm_tap_0", "segment_10_plate_0_end_hm_tap"] }],
    ...overrides,
  };
}

export function config(tendons, extra = {}) {
  return { format: "seahorse-tendon-config", version: 1, name: "test", tendons, ...extra };
}

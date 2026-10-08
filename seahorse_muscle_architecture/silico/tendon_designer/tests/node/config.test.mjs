// Tests for web/js/config.js (pure config helpers) and the preset files in web/configs/.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  newConfig, newTendon, addTendon, normaliseConfig, validateConfig, hasErrors, pointInfo,
  describeTendon, mirrorTendon, addMirroredTendon, selectSymmetricPoints, hmPath, PRESETS,
  removeTrunkPoint, moveTrunkPoint, pointInPolygons, plateAt, TENDON_COLORS,
} from "../../web/js/config.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(here, "../../web");
const catalog = JSON.parse(fs.readFileSync(path.join(web, "model/catalog.json"), "utf8"));
const loadPreset = (file) => JSON.parse(fs.readFileSync(path.join(web, "configs", file), "utf8"));
const errorsOf = (cfg) => validateConfig(cfg, catalog).filter((i) => i.level === "error");

test("newConfig / newTendon give unique names, palette colours and a 10 N motor", () => {
  let c = newConfig();
  const t1 = newTendon(c);
  assert.equal(t1.name, "tendon_1");
  assert.deepEqual(t1.actuator, { type: "motor", max_force: 10 });
  assert.equal(t1.color, TENDON_COLORS[0]);
  c = addTendon(c, t1);
  const t2 = newTendon(c);
  assert.equal(t2.name, "tendon_2");
  assert.equal(t2.color, TENDON_COLORS[1]);
  assert.equal(newTendon(c, { name: "tendon_1" }).name, "tendon_2");
  assert.equal(c.tendons.length, 1, "addTendon returns a new config");
});

test("select_symmetric_points port matches the Python helper", () => {
  // Values from seahorse_muscle_architecture/silico/experiments/utils.py (Python round = half-even).
  assert.deepEqual(selectSymmetricPoints(1), [5]);
  assert.deepEqual(selectSymmetricPoints(3), [9, 5, 0]);
  assert.deepEqual(selectSymmetricPoints(5), [9, 8, 5, 2, 0]);
  assert.deepEqual(selectSymmetricPoints(9), [9, 9, 8, 6, 5, 4, 2, 1, 0]);
});

test("every preset file validates against the catalog with zero errors and matches its generator", () => {
  const index = JSON.parse(fs.readFileSync(path.join(web, "configs/index.json"), "utf8"));
  assert.ok(index.length >= 6);
  for (const entry of index) {
    const cfg = loadPreset(entry.file);
    assert.deepEqual(errorsOf(cfg), [], `${entry.file} has errors`);
    const gen = PRESETS.find((p) => p.file === entry.file);
    assert.ok(gen, `no generator for ${entry.file}`);
    assert.deepEqual(cfg, gen.build(catalog), `${entry.file} is out of date; regenerate it from PRESETS`);
  }
});

test("HM span 4 preset reproduces the paper's routing", () => {
  const cfg = loadPreset("hm_span4_paper.json");
  const dex = cfg.tendons.find((t) => t.name === "hm_dextral");
  const sin = cfg.tendons.find((t) => t.name === "hm_sinistral");
  assert.deepEqual(dex.path, [
    "segment_6_plate_1_ghost_hm_tap_b",
    "segment_7_plate_0_intermediate_hm_tap_9",
    "segment_8_plate_0_intermediate_hm_tap_5",
    "segment_9_plate_0_intermediate_hm_tap_0",
    "segment_10_plate_0_end_hm_tap",
  ]);
  assert.deepEqual(sin.path, [
    "segment_6_plate_1_ghost_hm_tap_a",
    "segment_7_plate_1_intermediate_hm_tap_9",
    "segment_8_plate_1_intermediate_hm_tap_5",
    "segment_9_plate_1_intermediate_hm_tap_0",
    "segment_10_plate_1_end_hm_tap",
  ]);
  assert.equal(dex.group, "dextral");
  assert.equal(sin.group, "sinistral");
});

test("fork preset branches from the segment-8 point through segment 9 hole 0 to the segment 10 corner", () => {
  const cfg = loadPreset("hm_fork_5_9_10.json");
  const dex = cfg.tendons.find((t) => t.name === "hm_dextral");
  assert.equal(dex.path.length, 5);
  assert.equal(dex.branches.length, 1);
  assert.equal(pointInfo(dex.path[dex.branches[0].from], catalog, cfg).segment, 8);
  assert.deepEqual(dex.branches[0].path, ["segment_9_plate_0_intermediate_hm_tap_0", "segment_10_plate_0_end_hm_tap"]);
});

test("MVM chain: one position-controlled tendon per joint and side", () => {
  const cfg = loadPreset("mvm_chain.json");
  assert.equal(cfg.tendons.length, 2 * (catalog.num_segments - 1));
  for (const t of cfg.tendons) {
    assert.deepEqual(t.actuator, { type: "position", kp: 1000, max_force: 1000, max_strain: 0.26 });
    assert.ok(t.group === "mvm_dextral" || t.group === "mvm_sinistral");
    const [a, b] = t.path.map((id) => pointInfo(id, catalog, cfg));
    assert.equal(b.segment, a.segment + 1);
    assert.equal(a.kind, "mvm");
  }
});

test("validateConfig reports problems in plain language", () => {
  const c = normaliseConfig({
    tendons: [
      { name: "bad name", path: [] },
      { name: "segment_x", path: ["segment_3_plate_0_intermediate_hm_tap_1"] },
      { name: "t", path: ["nope", "segment_3_plate_0_end_hm_tap"], branches: [{ from: 7, path: [] }] },
      { name: "t", path: ["segment_2_plate_0_end_hm_tap", "segment_5_plate_0_end_hm_tap"], actuator: { type: "position", kp: 0, max_force: 5, max_strain: 2 } },
    ],
    free_points: [{ id: "free_1", segment: 3, plate: 0, xy: [0, 0] }],
  });
  const issues = validateConfig(c, catalog);
  const text = issues.map((i) => i.message).join("\n");
  assert.ok(hasErrors(issues));
  assert.match(text, /letters, digits/);
  assert.match(text, /cannot start with "segment_"/);
  assert.match(text, /has no points yet/);
  assert.match(text, /has a start but no end/);
  assert.match(text, /"nope"\) does not exist/);
  assert.match(text, /branch 1 splits at point 8/);
  assert.match(text, /same name/);
  assert.match(text, /controller gain/);
  assert.match(text, /maximum shortening/);
  assert.match(text, /jumps from segment 2 to 5/);
  assert.match(text, /outside its plate/);
  assert.match(text, /not used by any tendon/);
  for (const i of issues) assert.ok(["error", "warning"].includes(i.level) && typeof i.message === "string");
});

test("normaliseConfig fills defaults and migrates older shapes", () => {
  const c = normaliseConfig({
    tendons: [{ name: "a", points: ["segment_1_plate_0_end_hm_tap", "segment_2_plate_0_end_hm_tap"], actuator: "position",
      branches: [{ from: "segment_1_plate_0_end_hm_tap", path: ["segment_3_plate_0_end_hm_tap"] }] }],
  });
  const t = c.tendons[0];
  assert.equal(c.format, "seahorse-tendon-config");
  assert.deepEqual(t.path, ["segment_1_plate_0_end_hm_tap", "segment_2_plate_0_end_hm_tap"]);
  assert.equal(t.points, undefined);
  assert.equal(t.actuator.type, "position");
  assert.equal(t.actuator.max_strain, 0.26);
  assert.equal(t.branches[0].from, 0);
  assert.equal(t.width, 0.001);
  assert.throws(() => normaliseConfig({ format: "something-else" }), /format/);
});

test("pointInfo and describeTendon", () => {
  const cfg = loadPreset("hm_fork_5_9_10.json");
  const info = pointInfo("segment_5_plate_1_ghost_hm_tap_b", catalog, cfg);
  assert.equal(info.segment, 5);
  assert.equal(info.plate, 1);
  assert.equal(info.kind, "ghost");
  assert.equal(info.label, "midline hole b");
  assert.equal(pointInfo("missing", catalog, cfg), null);
  const d = describeTendon(cfg.tendons[0], catalog, cfg);
  assert.equal(d, "starts seg 5 (midline hole b) → 3 via points → ends seg 9 (corner); branch at seg 8 → 1 via point → ends seg 10 (corner)");
  assert.equal(describeTendon({ path: [] }, catalog, cfg), "no points yet");
});

test("mirrorTendon maps dextral HM taps onto the sinistral ones (and back)", () => {
  const cfg = loadPreset("hm_fork_5_9_10.json");
  const dex = cfg.tendons.find((t) => t.name === "hm_dextral");
  const sin = cfg.tendons.find((t) => t.name === "hm_sinistral");
  const without = { ...cfg, tendons: [dex] };
  const m = mirrorTendon(dex, catalog, without);
  assert.equal(m.tendon.name, "hm_sinistral");
  assert.equal(m.tendon.group, "sinistral");
  assert.deepEqual(m.tendon.path, sin.path);
  assert.deepEqual(m.tendon.branches, sin.branches);
  assert.deepEqual(m.free_points, []);
  // Name collision -> unique name.
  assert.equal(mirrorTendon(dex, catalog, cfg).tendon.name, "hm_sinistral_2");
  // MVM holes mirror onto each other too.
  const mvm = loadPreset("mvm_chain.json").tendons[0];
  assert.deepEqual(mirrorTendon(mvm, catalog, newConfig()).tendon.path, mvm.path.map((id) => id.replace("dextral", "sinistral")));
});

test("mirroring free points creates mirrored free points on the mirrored plate", () => {
  let c = newConfig();
  c.free_points.push({ id: "free_1", segment: 4, plate: 0, xy: [0.031, -0.02] });
  c = addTendon(c, newTendon(c, { name: "x_dextral", path: ["segment_3_plate_0_end_hm_tap", "free_1"] }));
  const r = addMirroredTendon(c, "x_dextral", catalog);
  assert.equal(r.name, "x_sinistral");
  const t = r.config.tendons.find((tt) => tt.name === "x_sinistral");
  assert.equal(t.path[0], "segment_3_plate_1_end_hm_tap");
  const fp = r.config.free_points.find((f) => f.id === t.path[1]);
  assert.ok(fp && fp.id !== "free_1");
  assert.deepEqual(fp.xy, [0.031, 0.02]);
  assert.equal(fp.plate, 1);
  assert.deepEqual(errorsOf(r.config), []);
  assert.equal(c.tendons.length, 1, "input config untouched");
});

test("trunk edit helpers keep branch split indices attached to their point", () => {
  const t = { path: ["a", "b", "c", "d"], branches: [{ from: 2, path: ["x"] }] };
  assert.deepEqual(removeTrunkPoint(t, 0).branches[0].from, 1);
  assert.deepEqual(removeTrunkPoint(t, 2).branches[0].from, 1);
  assert.deepEqual(moveTrunkPoint(t, 2, 1).branches[0].from, 3);
  assert.deepEqual(moveTrunkPoint(t, 1, 1).path, ["a", "c", "b", "d"]);
  assert.deepEqual(t.path, ["a", "b", "c", "d"], "input untouched");
});

test("plate hit-testing uses the even-odd rule", () => {
  const square = [[[0, 0], [10, 0], [10, 10], [0, 10]], [[4, 4], [6, 4], [6, 6], [4, 6]]];
  assert.equal(pointInPolygons([2, 2], square), true);
  assert.equal(pointInPolygons([5, 5], square), false, "inside the hole");
  assert.equal(pointInPolygons([12, 5], square), false);
  // Every tap sits on its own plate (except where the catalog marks it otherwise, none expected for holes).
  const p = plateAt(catalog, 6, [0.031, -0.02]);
  assert.equal(p.corner, "ventral_dextral");
  assert.equal(plateAt(catalog, 6, [0.0, 0.0]), null);
});

test("hmPath builds dorsal routes from catalog geometry", () => {
  const path = hmPath(catalog, { start: 3, end: 6, sagittal: "dorsal", coronal: "dextral" });
  const segs = path.map((id) => pointInfo(id, catalog, null).segment);
  assert.deepEqual(segs, [3, 4, 5, 6]);
  assert.equal(pointInfo(path[0], catalog, null).kind, "ghost");
  assert.ok(pointInfo(path[0], catalog, null).xy[1] < 0, "dextral ghost hole is at -y");
});

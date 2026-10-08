// Tests for web/js/model_builder.js (no WASM needed). Run: npm test (in this folder).
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { DOMParser } from "@xmldom/xmldom";
import { assets, hmForkTendon, config, WEB_DIR } from "./helpers.mjs";
import { expandTendonSites, buildModelXml, colorToRgba } from "../../web/js/model_builder.js";

const sitesOf = (items) => items.map((x) => (x.site ? x.site : "|pulley|"));

// --- a tiny synthetic catalog to exercise every rule in isolation -------------------
function synthCatalog() {
  const seg = (i, taps) => ({
    index: i, frame_body: `segment_${i}_vertebrae`,
    plates: [{ index: 0, corner: "ventral_dextral", body: `segment_${i}_plate_0`, free_site_z: [-0.001, 0.001], taps }],
  });
  const tap = (i, name, n = 2) => ({
    id: `s${i}_${name}`, kind: "intermediate", xy: [0, 0],
    sites: Array.from({ length: n }, (_, k) => `s${i}_${name}_${k}`),
  });
  return {
    num_segments: 4,
    segments: [0, 1, 2, 3].map((i) => seg(i, [tap(i, "a"), tap(i, "b"), tap(i, "one", 1)])),
  };
}

test("trunk forward: sites in catalog order", () => {
  const c = synthCatalog();
  const s = sitesOf(expandTendonSites({ name: "t", path: ["s1_a", "s2_a", "s3_a"] }, c));
  assert.deepEqual(s, ["s1_a_0", "s1_a_1", "s2_a_0", "s2_a_1", "s3_a_0", "s3_a_1"]);
});

test("trunk reverse direction (distal -> proximal): every point's sites reversed", () => {
  const c = synthCatalog();
  const s = sitesOf(expandTendonSites({ name: "t", path: ["s3_a", "s2_a", "s1_a"] }, c));
  assert.deepEqual(s, ["s3_a_1", "s3_a_0", "s2_a_1", "s2_a_0", "s1_a_1", "s1_a_0"]);
});

test("trunk within one segment counts as forward (D = +1)", () => {
  const c = synthCatalog();
  const s = sitesOf(expandTendonSites({ name: "t", path: ["s2_a", "s2_b"] }, c));
  assert.deepEqual(s, ["s2_a_0", "s2_a_1", "s2_b_0", "s2_b_1"]);
});

test("branch distal: pulley, exit site = last site of P_k, then branch sites", () => {
  const c = synthCatalog();
  const s = sitesOf(expandTendonSites({
    name: "t", path: ["s0_a", "s1_a", "s2_a"], branches: [{ from: 1, path: ["s2_b", "s3_b"] }],
  }, c));
  assert.deepEqual(s, ["s0_a_0", "s0_a_1", "s1_a_0", "s1_a_1", "s2_a_0", "s2_a_1",
    "|pulley|", "s1_a_1", "s2_b_0", "s2_b_1", "s3_b_0", "s3_b_1"]);
});

test("branch proximal (Db = -1): exit site is the proximal site, branch sites reversed", () => {
  const c = synthCatalog();
  const s = sitesOf(expandTendonSites({
    name: "t", path: ["s2_a", "s3_a"], branches: [{ from: 0, path: ["s1_b", "s0_b"] }],
  }, c));
  assert.deepEqual(s, ["s2_a_0", "s2_a_1", "s3_a_0", "s3_a_1",
    "|pulley|", "s2_a_0", "s1_b_1", "s1_b_0", "s0_b_1", "s0_b_0"]);
});

test("rule 4: consecutive duplicate sites are dropped (also right after the pulley)", () => {
  const c = synthCatalog();
  const s = sitesOf(expandTendonSites({
    name: "t", path: ["s1_one", "s1_one", "s2_a"], branches: [{ from: 0, path: ["s1_one", "s3_a"] }],
  }, c));
  assert.deepEqual(s, ["s1_one_0", "s2_a_0", "s2_a_1", "|pulley|", "s1_one_0", "s3_a_0", "s3_a_1"]);
});

test("free points expand to id_0, id_1 and follow direction rules", () => {
  const c = synthCatalog();
  const fps = [{ id: "fp", segment: 3, plate: 0, xy: [0.01, 0.02] }];
  const s = sitesOf(expandTendonSites({ name: "t", path: ["fp", "s1_a"] }, c, fps));
  assert.deepEqual(s, ["fp_1", "fp_0", "s1_a_1", "s1_a_0"]);
});

test("expansion errors are readable", () => {
  const c = synthCatalog();
  assert.throws(() => expandTendonSites({ name: "t", path: ["s1_a"] }, c), /at least 2 points/);
  assert.throws(() => expandTendonSites({ name: "t", path: ["s1_a", "nope"] }, c), /unknown point "nope"/);
  assert.throws(() => expandTendonSites({ name: "t", path: ["s1_a", "s2_a"], branches: [{ from: 5, path: ["s3_a"] }] }, c),
    /"from" must be a trunk point/);
});

test("HM fork config reproduces the site list of mjcf_export_hm_fork (Python export parity)", () => {
  const { catalog } = assets();
  const s = sitesOf(expandTendonSites(hmForkTendon(), catalog, []));
  const ref = path.resolve(WEB_DIR, "../../../../mjcf_export_hm_fork/seahorse_hm_fork.xml");
  if (!fs.existsSync(ref)) return; // reference export not present
  const doc = new DOMParser().parseFromString(fs.readFileSync(ref, "utf8"), "text/xml");
  const sp = Array.from(doc.getElementsByTagName("spatial")).find((e) => e.getAttribute("name") === "hm_beam_ventral_dextral_9_10");
  const expected = [];
  for (let c = sp.firstChild; c; c = c.nextSibling) {
    if (c.nodeType !== 1) continue;
    expected.push(c.nodeName === "pulley" ? "|pulley|" : c.getAttribute("site"));
  }
  assert.deepEqual(s, expected);
});

function parse(xml) { return new DOMParser().parseFromString(xml, "text/xml"); }
function byName(doc, tag, name) {
  return Array.from(doc.getElementsByTagName(tag)).find((e) => e.getAttribute("name") === name);
}

test("buildModelXml: tendon, motor, sensors, free point sites", () => {
  const { baseXml, catalog } = assets();
  const plate = catalog.segments[8].plates[0];
  const cfg = config([
    hmForkTendon(),
    { name: "free_t", group: "x", color: "#ff0000", actuator: { type: "position", kp: 500, max_force: 50, max_strain: 0.2 },
      path: ["segment_2_plate_0_intermediate_hm_tap_3", "fp1"] },
    { name: "passive", color: "#00ff00", stiffness: 5, actuator: { type: "none" },
      path: ["segment_2_plate_2_intermediate_hm_tap_3", "segment_4_plate_2_end_hm_tap"] },
  ], { free_points: [{ id: "fp1", segment: 8, plate: 0, xy: [0.031, -0.02] }] });
  const doc = parse(buildModelXml(baseXml, catalog, cfg));

  const sp = byName(doc, "spatial", "hm_dextral");
  assert.equal(sp.getAttribute("springlength"), "-1 -1");
  assert.equal(sp.getAttribute("rgba"), colorToRgba("#3b82f6"));
  const motor = byName(doc, "motor", "hm_dextral");
  assert.equal(motor.getAttribute("gear"), "10");
  assert.equal(motor.getAttribute("forcerange"), "-10 0");
  assert.equal(motor.getAttribute("ctrlrange"), "-1 0");
  const pos = byName(doc, "position", "free_t");
  assert.equal(pos.getAttribute("kp"), "500");
  assert.equal(pos.getAttribute("forcerange"), "-50 0");
  assert.equal(byName(doc, "motor", "passive"), undefined);
  assert.ok(byName(doc, "tendonpos", "passive_length"));
  assert.equal(byName(doc, "actuatorfrc", "passive_force"), undefined);
  assert.ok(byName(doc, "actuatorfrc", "hm_dextral_force"));

  const s0 = byName(doc, "site", "fp1_0"), s1 = byName(doc, "site", "fp1_1");
  assert.equal(s0.parentNode.getAttribute("name"), plate.body);
  assert.deepEqual(s0.getAttribute("pos").split(" ").map(Number), [0.031, -0.02, plate.free_site_z[0]]);
  assert.deepEqual(s1.getAttribute("pos").split(" ").map(Number), [0.031, -0.02, plate.free_site_z[1]]);
});

test("buildModelXml: params overrides incl. taper, ranges, struts, glide, hanging, gravity, timestep", () => {
  const { baseXml, catalog } = assets();
  const n = catalog.num_segments;
  const cfg = config([], {
    params: {
      vertebra: { pitch_stiffness: 0.01, roll_damping: 0.002, yaw_range_deg: 10 },
      stiffness_taper: 3.0,
      plate_glide: { stiffness: 2, damping: 0.5 },
      strut: { stiffness: 7 },
      orientation: "hanging", gravity: true, timestep: 0.001,
    },
  });
  const doc = parse(buildModelXml(baseXml, catalog, cfg));
  const j = (i, ax) => byName(doc, "joint", `segment_${i}_vertebrae_vertebrae_joint_${ax}`);
  assert.equal(+j(1, "pitch").getAttribute("stiffness"), 0.01);
  assert.ok(Math.abs(+j(n - 1, "pitch").getAttribute("stiffness") - 0.03) < 1e-12);
  const mid = 1 + 2 * (5 - 1) / (n - 2);
  assert.ok(Math.abs(+j(5, "pitch").getAttribute("stiffness") - 0.01 * mid) < 1e-12);
  // taper also scales the (non-overridden) base roll/yaw stiffness
  assert.ok(Math.abs(+j(n - 1, "yaw").getAttribute("stiffness") - 3 * catalog.defaults.vertebra.yaw_stiffness) < 1e-12);
  assert.equal(+j(3, "roll").getAttribute("damping"), 0.002);
  const r = j(3, "yaw").getAttribute("range").split(" ").map(Number);
  assert.ok(Math.abs(r[1] - 10 * Math.PI / 180) < 1e-12 && Math.abs(r[0] + r[1]) < 1e-15);
  assert.equal(+byName(doc, "joint", "segment_4_plate_2_x_axis").getAttribute("stiffness"), 2);
  assert.equal(+byName(doc, "joint", "segment_4_plate_2_y_axis").getAttribute("damping"), 0.5);
  assert.equal(+byName(doc, "spatial", "segment_3_vertebral_vertebral_strut_dorsal").getAttribute("stiffness"), 7);
  assert.equal(byName(doc, "body", "segment_0").getAttribute("euler"), "3.141592653589793 0 0");
  const option = doc.getElementsByTagName("option")[0];
  assert.equal(option.getAttribute("timestep"), "0.001");
  const flag = option.getElementsByTagName("flag")[0];
  assert.equal(flag.getAttribute("gravity"), "enable");
  assert.equal(flag.getAttribute("contact"), "disable"); // untouched
});

test("buildModelXml: no params -> base attributes untouched", () => {
  const { baseXml, catalog } = assets();
  const a = parse(baseXml), b = parse(buildModelXml(baseXml, catalog, config([])));
  const name = "segment_5_vertebrae_vertebrae_joint_pitch";
  assert.equal(byName(b, "joint", name).getAttribute("stiffness"), byName(a, "joint", name).getAttribute("stiffness"));
});

test("buildModelXml: validation errors", () => {
  const { baseXml, catalog } = assets();
  const t = hmForkTendon();
  assert.throws(() => buildModelXml(baseXml, catalog, config([t, { ...t }])), /used twice/);
  assert.throws(() => buildModelXml(baseXml, catalog, config([{ ...t, name: "segment_x" }])), /must not start with "segment_"/);
  assert.throws(() => buildModelXml(baseXml, catalog, config([{ ...t, name: "bad name" }])), /may only contain/);
  assert.throws(() => buildModelXml(baseXml, catalog, config([{ ...t, actuator: { type: "muscle" } }])), /unknown actuator type/);
  assert.throws(() => buildModelXml(baseXml, catalog, config([], { free_points: [{ id: "f", segment: 99, plate: 0, xy: [0, 0] }] })),
    /no plate 0 in segment 99/);
});

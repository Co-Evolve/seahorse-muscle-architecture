// Tests for web/js/sim.js: load MuJoCo WASM in Node, compile, step, measure.
import test from "node:test";
import assert from "node:assert/strict";
import { assets, hmForkTendon, config } from "./helpers.mjs";
import { loadMujoco, Simulation } from "../../web/js/sim.js";

const mujoco = await loadMujoco();
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} ${a} vs ${b} (tol ${tol})`);

/** Hamilton product of quaternions [w, x, y, z]. */
const qmul = (a, b) => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const qaxis = (ax, deg) => {
  const h = deg * Math.PI / 360, q = [Math.cos(h), 0, 0, 0];
  q[ax] = Math.sin(h);
  return q;
};
/** Twist (deg) of the vertebral joint chain pitch(y) -> roll(x) -> yaw(z), as defined in DESIGN.md. */
function expectedTwist({ pitch, roll, yaw }) {
  const q = qmul(qmul(qaxis(2, pitch), qaxis(1, roll)), qaxis(3, yaw));
  return 2 * Math.atan2(q[3], q[0]) * 180 / Math.PI;
}

/** Length of the polyline through the given site names, from data.site_xpos. */
function polyline(sim, names) {
  const xp = sim.data.site_xpos;
  let L = 0;
  for (let k = 1; k < names.length; k++) {
    const a = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE.value, names[k - 1]);
    const b = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE.value, names[k]);
    assert.ok(a >= 0 && b >= 0, `sites ${names[k - 1]} ${names[k]}`);
    L += Math.hypot(xp[3 * a] - xp[3 * b], xp[3 * a + 1] - xp[3 * b + 1], xp[3 * a + 2] - xp[3 * b + 2]);
  }
  return L;
}

/** Split the generated <spatial> of tendon `name` into its pulley-separated branches. */
function branchesFromXml(xml, name) {
  const body = new RegExp(`<spatial name="${name}"[^>]*>([\\s\\S]*?)</spatial>`).exec(xml)[1];
  const out = [[]];
  for (const m of body.matchAll(/<(site|pulley)\b([^>]*)\/>/g)) {
    if (m[1] === "pulley") out.push([]);
    else out[out.length - 1].push(/site="([^"]+)"/.exec(m[2])[1]);
  }
  return out;
}

test("forked tendon compiles; length at qpos0 = sum of the branch polylines", async () => {
  const sim = await Simulation.create(mujoco, assets(), config([hmForkTendon()]));
  try {
    const t = sim.tendons[0];
    assert.equal(t.type, "motor");
    assert.ok(t.actuatorId !== null && t.tendonId >= 0);
    const branches = branchesFromXml(sim.xml, "hm_dextral");
    assert.equal(branches.length, 2);
    const sum = branches.reduce((s, b) => s + polyline(sim, b), 0);
    close(t.length0, sum, 1e-9, "length0");
    close(sim.data.ten_length[t.tendonId], sum, 1e-9, "ten_length");
    const m = sim.measure();
    close(m.tendons[0].length, sum, 1e-9);
    close(m.tendons[0].strain, 0, 1e-12);
    close(m.tip.ventral, 0, 1e-6, "rest tip ventral");
    console.log(`  compile ${sim.compileMs.toFixed(0)} ms; fork length0 = ${(sum * 1000).toFixed(2)} mm (trunk ${(polyline(sim, branches[0]) * 1000).toFixed(2)} + branch ${(polyline(sim, branches[1]) * 1000).toFixed(2)})`);
  } finally { sim.dispose(); }
});

test("motor activation 1 bends the tail ventrally; activation 0 relaxes", async () => {
  const sim = await Simulation.create(mujoco, assets(), config([hmForkTendon()]));
  try {
    sim.setActivation("hm_dextral", 1);
    const t0 = performance.now();
    const nsteps = sim.advance(2.0);
    const ms = performance.now() - t0;
    assert.equal(nsteps, Math.round(2.0 / sim.timestep));
    close(sim.time, 2.0, 1e-9);
    const m = sim.measure();
    const tm = m.tendons[0];
    console.log(`  ${nsteps} steps in ${ms.toFixed(0)} ms (${(ms / nsteps * 1000).toFixed(0)} us/step); ` +
      `tip ventral ${m.tip.ventral.toFixed(2)} deg, lateral ${m.tip.lateral.toFixed(2)} deg, twist ${m.tip.twist.toFixed(2)} deg, ` +
      `dist ${(m.tip.distance * 1000).toFixed(1)} mm, force ${tm.force.toFixed(2)} N, strain ${(tm.strain * 100).toFixed(2)} %, work ${(tm.work * 1000).toFixed(3)} mJ`);
    console.log("  segment ventral:", m.segments.map((s) => s.ventral.toFixed(2)).join(" "));
    assert.ok(m.tip.ventral > 5, `tip should bend ventrally (got ${m.tip.ventral})`);
    close(tm.force, 10, 1e-6, "motor force at a=1");
    assert.ok(tm.excursion > 0 && tm.strain > 0);
    assert.ok(tm.work > 0);
    // segments spanned by the tendon (6..10) bend ventrally (the proximal ones move a
    // little, dorsally, through the plate coupling)
    assert.ok(m.segments.slice(5).every((s) => s.ventral > 1));
    // a dextral tendon also bends towards dextral (-y)
    assert.ok(m.tip.lateral > 0 && m.tip.pos[1] < 0);
    close(m.cumulative_ventral[m.cumulative_ventral.length - 1],
      m.segments.reduce((s, x) => s + x.ventral, 0), 1e-9);
    // joint angle readout agrees with the frame-based ventral angle (pitch about y)
    // per segment: ventral ~ pitch (about y), lateral ~ -roll (roll about +x tilts z to -y = dextral),
    // twist ~ yaw (about z); exact up to the coupling of the three hinge rotations
    for (const s of m.segments) {
      close(s.ventral, s.joint_angle.pitch, 0.5, `seg ${s.index} ventral/pitch`);
      close(s.lateral, s.joint_angle.roll, 0.5, `seg ${s.index} lateral/roll`);
      // twist = swing-twist angle about z of q = q_y(pitch) q_x(roll) q_z(yaw)
      close(s.twist, expectedTwist(s.joint_angle), 1e-6, `seg ${s.index} twist`);
    }

    // releasing: the force drops to 0 (the base model's vertebral springs are very weak
    // and the joints have friction loss, so the tail does not necessarily spring back)
    sim.setActivation("hm_dextral", 0);
    sim.advance(0.5);
    const r = sim.measure();
    close(r.tendons[0].force, 0, 1e-12);
    assert.ok(Math.abs(r.tip.ventral) <= Math.abs(m.tip.ventral) + 1);
  } finally { sim.dispose(); }
});

test("per-tendon torque contributions sum to qfrc_actuator at every vertebral dof", async () => {
  const sinistral = hmForkTendon({
    name: "hm_sinistral", color: "#ef4444",
    path: ["segment_5_plate_1_ghost_hm_tap_a", "segment_6_plate_1_intermediate_hm_tap_9",
      "segment_7_plate_1_intermediate_hm_tap_5", "segment_8_plate_1_intermediate_hm_tap_0",
      "segment_9_plate_1_end_hm_tap"],
    branches: [{ from: 3, path: ["segment_9_plate_1_intermediate_hm_tap_0", "segment_10_plate_1_end_hm_tap"] }],
  });
  const dorsal = {
    name: "dorsal_pos", color: "#22c55e", actuator: { type: "position", kp: 2000, max_force: 20, max_strain: 0.3 },
    path: ["segment_2_plate_2_intermediate_hm_tap_4", "segment_4_plate_2_intermediate_hm_tap_4", "segment_6_plate_3_end_hm_tap"],
  };
  const sim = await Simulation.create(mujoco, assets(), config([hmForkTendon(), sinistral, dorsal]));
  try {
    sim.setActivations({ hm_dextral: 1, hm_sinistral: 0.6, dorsal_pos: 1 }); // dorsal target strain 30 % is out of reach -> saturates
    sim.advance(0.5);
    const m = sim.measure();
    let maxAbs = 0;
    for (const s of m.segments) {
      for (const ax of ["pitch", "roll", "yaw"]) {
        const sum = m.tendons.reduce((acc, t) => acc + t.torque[s.index - 1][ax], 0);
        close(sum, s.torque[ax], 1e-9 + 1e-9 * Math.abs(s.torque[ax]), `seg ${s.index} ${ax}`);
        maxAbs = Math.max(maxAbs, Math.abs(s.torque[ax]));
      }
    }
    assert.ok(maxAbs > 1e-4, "some actuator torque present");
    console.log("  pitch torque (N·mm) per segment:", m.segments.map((s) => (s.torque.pitch * 1000).toFixed(2)).join(" "));
    const p = m.tendons.find((t) => t.name === "dorsal_pos");
    assert.ok(p.force > 0 && p.force <= 20 + 1e-9, `position actuator force ${p.force}`);
  } finally { sim.dispose(); }
});

test("moment arms equal the finite-difference dL/dq of the tendon", async () => {
  const sim = await Simulation.create(mujoco, assets(), config([hmForkTendon()]));
  try {
    const { model, data } = sim;
    const tid = sim.tendons[0].tendonId;
    const arms = sim.measure().tendons[0].moment_arms;
    for (const [seg, ax] of [[7, "pitch"], [9, "roll"], [3, "pitch"]]) {
      const j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, `segment_${seg}_vertebrae_vertebrae_joint_${ax}`);
      const qa = model.jnt_qposadr[j], h = 1e-6;
      data.qpos[qa] += h; mujoco.mj_forward(model, data); const Lp = data.ten_length[tid];
      data.qpos[qa] -= 2 * h; mujoco.mj_forward(model, data); const Lm = data.ten_length[tid];
      data.qpos[qa] += h; mujoco.mj_forward(model, data);
      close(arms[seg - 1][ax], (Lp - Lm) / (2 * h), 1e-6, `moment arm seg ${seg} ${ax}`);
    }
    assert.ok(arms[7].pitch !== 0, "tendon spans segment 8");
    assert.ok(Math.abs(arms[2].pitch) < 1e-12, "tendon does not span segment 3");
  } finally { sim.dispose(); }
});

test("reverse direction tendon has the same length as the forward one", async () => {
  const fwd = { name: "fwd", path: ["segment_3_plate_1_intermediate_hm_tap_2", "segment_5_plate_1_intermediate_hm_tap_2", "segment_7_plate_1_end_hm_tap"] };
  const rev = { name: "rev", path: [...fwd.path].reverse() };
  const sim = await Simulation.create(mujoco, assets(), config([fwd, rev]));
  try {
    close(sim.tendons[0].length0, sim.tendons[1].length0, 1e-12);
    assert.equal(sim.tendons[0].actuatorId, null);
    assert.equal(sim.tendons[0].type, "none");
  } finally { sim.dispose(); }
});

test("free point sites are placed on the plate at (x, y, free_site_z)", async () => {
  const { catalog } = assets();
  const plate = catalog.segments[8].plates[0];
  const fp = { id: "fp1", segment: 8, plate: 0, xy: [0.031, -0.02] };
  const t = { name: "via_free", actuator: { type: "motor", max_force: 5 }, path: ["segment_6_plate_0_intermediate_hm_tap_3", "fp1"] };
  const sim = await Simulation.create(mujoco, assets(), config([t], { free_points: [fp] }));
  try {
    const sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE.value, "fp1_1");
    const bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY.value, plate.body);
    assert.equal(sim.model.site_bodyid[sid], bid);
    // world position = segment frame (identity plate rest pose) applied to local pos
    const fb = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY.value, catalog.segments[8].frame_body);
    const xp = sim.data.xpos, sp = sim.data.site_xpos;
    close(sp[3 * sid] - xp[3 * fb], 0.031, 1e-9);
    close(sp[3 * sid + 1] - xp[3 * fb + 1], -0.02, 1e-9);
    close(sp[3 * sid + 2] - xp[3 * fb + 2], plate.free_site_z[1], 1e-9);
    sim.setActivation("via_free", 1);
    sim.advance(0.2);
    assert.ok(sim.measure().tendons[0].force > 4.99);
  } finally { sim.dispose(); }
});

test("hanging orientation: tail along -z in the world, measurements in base frame unchanged", async () => {
  const up = await Simulation.create(mujoco, assets(), config([hmForkTendon()]));
  const hang = await Simulation.create(mujoco, assets(), config([hmForkTendon()], { params: { orientation: "hanging" } }));
  try {
    const tipBody = mujoco.mj_name2id(hang.model, mujoco.mjtObj.mjOBJ_BODY.value, "segment_10_vertebrae");
    assert.ok(up.data.xpos[3 * tipBody + 2] > 0.25);
    assert.ok(hang.data.xpos[3 * tipBody + 2] < -0.25);
    for (const s of [up, hang]) { s.setActivation("hm_dextral", 1); s.advance(1.0); }
    const a = up.measure(), b = hang.measure();
    close(a.tip.ventral, b.tip.ventral, 1e-6);
    close(a.tip.pos[2], b.tip.pos[2], 1e-9);
  } finally { up.dispose(); hang.dispose(); }
});

test("hanging + gravity: the tail sags (tip moves) without activation", async () => {
  const sim = await Simulation.create(mujoco, assets(), config([hmForkTendon()], { params: { orientation: "hanging", gravity: true } }));
  try {
    sim.advance(1.0);
    const m = sim.measure();
    assert.ok(Number.isFinite(m.tip.distance));
    console.log(`  hanging + gravity, 1 s passive: tip ventral ${m.tip.ventral.toFixed(3)} deg, distance ${(m.tip.distance * 1000).toFixed(3)} mm`);
  } finally { sim.dispose(); }
});

test("position actuator: ctrl = length0 * (1 - a * max_strain); reset keeps activations", async () => {
  const t = hmForkTendon({ actuator: { type: "position", kp: 1000, max_force: 1000, max_strain: 0.05 } });
  const sim = await Simulation.create(mujoco, assets(), config([t]));
  try {
    sim.setActivation("hm_dextral", 1);
    const tt = sim.tendons[0];
    close(sim.data.ctrl[tt.actuatorId], tt.length0 * 0.95, 1e-12);
    sim.advance(1.0);
    const m1 = sim.measure();
    assert.ok(m1.tendons[0].strain > 0.005, `strain ${m1.tendons[0].strain}`);
    sim.reset();
    assert.equal(sim.time, 0);
    assert.equal(sim.getActivation("hm_dextral"), 1);
    close(sim.data.ctrl[tt.actuatorId], tt.length0 * 0.95, 1e-12);
    const m0 = sim.measure();
    close(m0.tip.ventral, 0, 1e-9);
    assert.equal(m0.tendons[0].work, 0);
    // same trajectory after reset (deterministic)
    sim.advance(1.0);
    close(sim.measure().tip.ventral, m1.tip.ventral, 1e-9);
    // measure({torques:false}) skips the torque fields
    const lite = sim.measure({ torques: false });
    assert.equal(lite.segments[0].torque, undefined);
    assert.equal(lite.tendons[0].moment_arms, undefined);
    assert.ok(lite.segments[0].joint_angle);
  } finally { sim.dispose(); }
});

test("activation is clamped to [0, 1]; unknown tendon names throw", async () => {
  const sim = await Simulation.create(mujoco, assets(), config([hmForkTendon()]));
  try {
    sim.setActivation("hm_dextral", 3);
    assert.equal(sim.getActivation("hm_dextral"), 1);
    sim.setActivation("hm_dextral", -1);
    assert.equal(sim.getActivation("hm_dextral"), 0);
    assert.throws(() => sim.setActivation("nope", 1), /Unknown tendon/);
  } finally { sim.dispose(); }
});

test("compile errors are readable Errors", async () => {
  await assert.rejects(Simulation.create(mujoco, assets(), config([{ name: "t", path: ["segment_1_plate_0_end_hm_tap", "missing_point"] }])),
    /unknown point "missing_point"/);
  const bad = { ...assets(), baseXml: assets().baseXml.replace("<worldbody>", "<worldbody><geom type=\"mesh\" mesh=\"nope\"/>") };
  await assert.rejects(Simulation.create(mujoco, bad, config([])), (e) => /MuJoCo could not compile the model: .*nope/s.test(e.message) && !!e.xml);
});

test("repeated create/dispose does not grow WASM memory unboundedly", async () => {
  let first = await Simulation.create(mujoco, assets(), config([hmForkTendon()]));
  const view = first.data.qpos; // any typed-array view onto the WASM heap
  const heap = () => view.buffer.byteLength;
  first.dispose();
  const before = heap();
  for (let i = 0; i < 4; i++) (await Simulation.create(mujoco, assets(), config([hmForkTendon()]))).dispose();
  const after = heap();
  console.log(`  WASM heap ${(before / 2 ** 20).toFixed(0)} MB -> ${(after / 2 ** 20).toFixed(0)} MB after 4 rebuilds`);
  assert.ok(after - before < 64 * 2 ** 20);
});

test("every preset in web/configs compiles, steps and measures finite numbers", async () => {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const { WEB_DIR } = await import("./helpers.mjs");
  const dir = path.join(WEB_DIR, "configs");
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith(".json") && x !== "index.json")) {
    const cfg = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    const sim = await Simulation.create(mujoco, assets(), cfg);
    try {
      for (const t of sim.tendons) sim.setActivation(t.name, 1);
      sim.advance(0.2);
      const m = sim.measure();
      assert.ok(Number.isFinite(m.tip.ventral) && Number.isFinite(m.tip.distance), f);
      for (const t of m.tendons) assert.ok(Number.isFinite(t.length) && Number.isFinite(t.force), `${f} ${t.name}`);
      console.log(`  ${f}: ${sim.tendons.length} tendons, tip ventral ${m.tip.ventral.toFixed(1)} deg after 0.2 s at a=1`);
    } finally { sim.dispose(); }
  }
});

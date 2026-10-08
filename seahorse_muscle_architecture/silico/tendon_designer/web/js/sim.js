// sim.js — MuJoCo WebAssembly simulation of one tendon configuration.
//
//   const mujoco = await loadMujoco();
//   const assets = await loadAssets("model/");
//   const sim = await Simulation.create(mujoco, assets, config);
//   sim.setActivation("hm_dextral", 1);
//   sim.advance(1.0);                 // simulate one second
//   const m = sim.measure();          // angles, tendon lengths, forces, torques
//   sim.dispose();                    // REQUIRED: frees WASM memory (model + data)
//
// MuJoCo objects live on the WASM heap and are not garbage collected. Every Simulation
// must be dispose()d once it is no longer used (e.g. before building a new one after an
// edit), otherwise each rebuild leaks the whole compiled model (~35 MB).

import { buildModelXml } from "./model_builder.js";

let mujocoPromise = null;

/**
 * Load the MuJoCo WASM module (singleton; later calls return the same module).
 * @param {object} [moduleOptions]  passed to the Emscripten factory on first call only
 * @returns {Promise<object>} the MuJoCo module (`mujoco.MjModel`, `mujoco.mj_step`, ...)
 */
export async function loadMujoco(moduleOptions = {}) {
  if (!mujocoPromise) {
    mujocoPromise = import("../vendor/mujoco/mujoco.js")
      .then((m) => m.default(moduleOptions))
      .catch((e) => { mujocoPromise = null; throw e; });
  }
  return mujocoPromise;
}

/**
 * Fetch base.xml, catalog.json and every mesh listed in catalog.mesh_files.
 * @param {string} [baseUrl="model/"]  URL of the model folder (trailing slash optional)
 * @param {{onProgress?: (loaded: number, total: number) => void}} [options]
 * @returns {Promise<{baseXml: string, catalog: object, meshes: Object<string, Uint8Array>}>}
 */
export async function loadAssets(baseUrl = "model/", { onProgress } = {}) {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const get = async (name) => {
    const res = await fetch(base + name);
    if (!res.ok) throw new Error(`Could not load ${base + name} (HTTP ${res.status})`);
    return res;
  };
  const [baseXml, catalog] = await Promise.all([
    get("base.xml").then((r) => r.text()),
    get("catalog.json").then((r) => r.json()),
  ]);
  const files = catalog.mesh_files || [];
  const meshes = {};
  let loaded = 0;
  await Promise.all(files.map(async (f) => {
    meshes[f] = new Uint8Array(await (await get(f)).arrayBuffer());
    onProgress?.(++loaded, files.length);
  }));
  return { baseXml, catalog, meshes };
}

// Meshes go into the Emscripten in-memory file system (MEMFS) of the MuJoCo module, one
// folder per assets object, written once. (MjVFS.addBuffer is avoided on purpose: in
// @mujoco/mujoco 3.14 it converts the bytes element by element and takes ~5 s for our
// 11 MB of meshes; FS.writeFile is a memcpy.) Each compile writes the XML next to the
// meshes and loads it with MjModel.from_xml_path.
const assetDirs = new WeakMap(); // assets -> {mujoco, dir}
let assetDirCounter = 0;
let xmlCounter = 0;

function getAssetDir(mujoco, assets) {
  const hit = assetDirs.get(assets);
  if (hit && hit.mujoco === mujoco) return hit.dir;
  const FS = mujoco.FS;
  const dir = `/tendon_designer_assets_${++assetDirCounter}`;
  FS.mkdir(dir);
  let meshDir = dir;
  const sub = /<compiler[^>]*\b(?:meshdir|assetdir)="([^"]*)"/.exec(assets.baseXml)?.[1];
  if (sub && !sub.startsWith("/")) {
    for (const part of sub.split("/").filter(Boolean)) {
      meshDir += "/" + part;
      try { FS.mkdir(meshDir); } catch (_) { /* exists */ }
    }
  }
  for (const [name, bytes] of Object.entries(assets.meshes || {})) {
    FS.writeFile(`${meshDir}/${name}`, bytes);
  }
  assetDirs.set(assets, { mujoco, dir, meshDir });
  return dir;
}

/**
 * Remove an assets object's meshes from the MuJoCo in-memory file system (only needed
 * if you switch to a different model folder and want the old meshes out of memory).
 */
export function releaseAssets(assets) {
  const hit = assetDirs.get(assets);
  if (!hit) return;
  const FS = hit.mujoco.FS;
  for (const name of Object.keys(assets.meshes || {})) {
    try { FS.unlink(`${hit.meshDir}/${name}`); } catch (_) { /* ignore */ }
  }
  assetDirs.delete(assets);
}

function compileXml(mujoco, assets, xml) {
  const dir = getAssetDir(mujoco, assets);
  const path = `${dir}/config_${++xmlCounter}.xml`;
  mujoco.FS.writeFile(path, xml);
  try {
    return mujoco.MjModel.from_xml_path(path);
  } finally {
    try { mujoco.FS.unlink(path); } catch (_) { /* ignore */ }
  }
}

const RAD2DEG = 180 / Math.PI;
const AXES = ["pitch", "roll", "yaw"];

/** Relative orientation of frame B w.r.t. frame A (both row-major 3x3 world matrices). */
function relAngles(A, aOff, B, bOff, out) {
  // R = A^T B ; R[r][c] = sum_k A[k][r] B[k][c]
  const R = relAngles.R;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      R[r * 3 + c] = A[aOff + r] * B[bOff + c] + A[aOff + 3 + r] * B[bOff + 3 + c] +
        A[aOff + 6 + r] * B[bOff + 6 + c];
    }
  }
  // v = B's z axis in A = third column of R
  const vx = R[2], vy = R[5], vz = R[8];
  out.ventral = Math.atan2(vx, vz) * RAD2DEG;
  // Elevation out of the sagittal plane: stays well-defined beyond 90 deg of ventral bending.
  out.lateral = Math.atan2(-vy, Math.hypot(vx, vz)) * RAD2DEG;
  out.bend = Math.acos(Math.max(-1, Math.min(1, vz))) * RAD2DEG;
  // twist: swing-twist decomposition of R about its z axis -> 2 atan2(qz, qw)
  const tr = R[0] + R[4] + R[8];
  let qw, qz;
  if (tr > -0.999999) {
    // general case (any rotation that is not a ~180 deg flip)
    const s = Math.sqrt(Math.max(0, 1 + tr)) * 2; // 4 qw
    qw = 0.25 * s;
    qz = (R[3] - R[1]) / s;
  } else {
    qw = 0; qz = 1;
  }
  let tw = 2 * Math.atan2(qz, qw) * RAD2DEG;
  if (tw > 180) tw -= 360; else if (tw <= -180) tw += 360;
  out.twist = tw;
  return out;
}
relAngles.R = new Float64Array(9);

/** p_world -> coordinates in frame (xpos, xmat) */
function toFrame(xpos, pOff, xmat, mOff, p, qOff) {
  const dx = p[qOff] - xpos[pOff], dy = p[qOff + 1] - xpos[pOff + 1], dz = p[qOff + 2] - xpos[pOff + 2];
  return [
    xmat[mOff] * dx + xmat[mOff + 3] * dy + xmat[mOff + 6] * dz,
    xmat[mOff + 1] * dx + xmat[mOff + 4] * dy + xmat[mOff + 7] * dz,
    xmat[mOff + 2] * dx + xmat[mOff + 5] * dy + xmat[mOff + 8] * dz,
  ];
}

/**
 * A compiled tendon configuration with its MuJoCo model and data.
 *
 * Public fields: xml, model, data, config, catalog, mujoco, tendons, numSegments, timestep.
 * `tendons[i]` = {name, group, color, type: "motor"|"position"|"none", tendonId,
 * actuatorId|null, length0, maxStrain}.
 */
export class Simulation {
  /**
   * Build the MJCF for `config`, compile it and run mj_forward at qpos0.
   * @param {object} mujoco  module from loadMujoco()
   * @param {{baseXml: string, catalog: object, meshes: Object<string, Uint8Array>}} assets
   * @param {object} config  tendon configuration JSON
   * @returns {Promise<Simulation>}
   * @throws {Error} readable message (config error or MuJoCo compile error); `error.xml` holds the XML if it got that far
   */
  static async create(mujoco, assets, config) {
    // shareMeshes: one <mesh> asset per (file, scale) — identical physics, ~10x less
    // WASM memory and faster compile when base.xml repeats a mesh asset per geom.
    const xml = buildModelXml(assets.baseXml, assets.catalog, config, { shareMeshes: true });
    let model;
    const t0 = performance.now();
    try {
      model = compileXml(mujoco, assets, xml);
    } catch (e) {
      const msg = String(e?.message || e).replace(/^Error:\s*/, "").replace(/^MuJoCo Error:\s*/, "");
      const err = new Error(`MuJoCo could not compile the model: ${msg}`);
      err.xml = xml;
      throw err;
    }
    const compileMs = performance.now() - t0;
    const sim = new Simulation(mujoco, model, xml, assets.catalog, config);
    sim.compileMs = compileMs;
    return sim;
  }

  /** @private use Simulation.create() */
  constructor(mujoco, model, xml, catalog, config) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.xml = xml;
    this.catalog = catalog;
    this.config = config;
    this.timestep = model.opt.timestep;

    const OBJ = mujoco.mjtObj;
    const id = (type, name) => mujoco.mj_name2id(model, type.value, name);

    this.tendons = (config.tendons || []).map((t) => {
      const tendonId = id(OBJ.mjOBJ_TENDON, t.name);
      const a = id(OBJ.mjOBJ_ACTUATOR, t.name);
      const type = t.actuator?.type || "none";
      return {
        name: t.name, group: t.group ?? "", color: t.color ?? "#888888", type,
        tendonId, actuatorId: a >= 0 && type !== "none" ? a : null,
        length0: model.tendon_length0[tendonId],
        maxStrain: t.actuator?.max_strain ?? 0.26,
        stiffness: model.tendon_stiffness[tendonId],
        springLength: model.tendon_lengthspring[2 * tendonId + 1], // upper end of the spring dead band
      };
    });
    this._byName = new Map(this.tendons.map((t, i) => [t.name, i]));
    this._act = new Float64Array(this.tendons.length);
    this._work = new Float64Array(this.tendons.length);
    this._pendingTime = 0;

    // --- segment frames and vertebral joints ---------------------------------
    const n = catalog?.num_segments ?? this._countSegments();
    this.numSegments = n;
    this._frameBody = [];
    for (let i = 0; i < n; i++) {
      const name = catalog?.segments?.[i]?.frame_body ?? `segment_${i}_vertebrae`;
      const b = id(OBJ.mjOBJ_BODY, name);
      if (b < 0) throw new Error(`Body "${name}" not found in model`);
      this._frameBody.push(b);
    }
    // vertebral joints: per segment 1..n-1 -> {pitch, roll, yaw} -> {qpos, dof} (-1 if absent)
    this._vjQpos = new Int32Array(n * 3).fill(-1);
    this._vjDof = new Int32Array(n * 3).fill(-1);
    this._dofToVJ = new Int32Array(model.nv).fill(-1); // dof -> i*3+axis
    for (let i = 1; i < n; i++) {
      AXES.forEach((ax, k) => {
        const j = id(OBJ.mjOBJ_JOINT, `segment_${i}_vertebrae_vertebrae_joint_${ax}`);
        if (j < 0) return;
        this._vjQpos[i * 3 + k] = model.jnt_qposadr[j];
        const dof = model.jnt_dofadr[j];
        this._vjDof[i * 3 + k] = dof;
        this._dofToVJ[dof] = i * 3 + k;
      });
    }
    this._ma = new Float64Array(n * 3); // scratch for moment arms

    this.reset();
    // tip position at qpos0 (base frame) for displacement
    this._tip0 = this._tipInBase();
  }

  _countSegments() {
    const OBJ = this.mujoco.mjtObj.mjOBJ_BODY.value;
    let n = 0;
    while (this.mujoco.mj_name2id(this.model, OBJ, `segment_${n}_vertebrae`) >= 0) n++;
    return n;
  }

  /** Simulated time in seconds. */
  get time() { return this.data.time; }

  /** Index of a tendon by name (throws if unknown). */
  _idx(name) {
    const i = this._byName.get(name);
    if (i === undefined) throw new Error(`Unknown tendon "${name}"`);
    return i;
  }

  /**
   * Set the activation of one tendon actuator, a in [0, 1] (clamped). Takes effect
   * on the next step. Passive tendons ignore it.
   */
  setActivation(name, a) {
    const i = this._idx(name);
    this._act[i] = Math.max(0, Math.min(1, +a || 0));
    this._applyCtrl(i);
  }

  /** @returns {number} current activation of a tendon in [0, 1] */
  getActivation(name) { return this._act[this._idx(name)]; }

  /** Set several activations at once: {name: a, ...}. Unknown names throw. */
  setActivations(map) {
    for (const [name, a] of Object.entries(map)) this.setActivation(name, a);
  }

  _applyCtrl(i) {
    const t = this.tendons[i];
    if (t.actuatorId === null) return;
    const a = this._act[i];
    this.data.ctrl[t.actuatorId] = t.type === "position" ? t.length0 * (1 - a * t.maxStrain) : -a;
  }

  /**
   * Advance the physics by n mj_step calls and integrate tendon work.
   * @param {number} [n=1]
   */
  step(n = 1) {
    const { mujoco, model, data } = this;
    const dt = this.timestep;
    const nt = this.tendons.length;
    for (let s = 0; s < n; s++) {
      mujoco.mj_step(model, data);
      if (nt) {
        // work done by the tendon pulling while shortening: F * (-dL/dt) * dt
        // (left-point rule: force and velocity of the state after this step)
        const vel = data.ten_velocity;
        for (let i = 0; i < nt; i++) {
          this._work[i] += this._tension(i) * -vel[this.tendons[i].tendonId] * dt;
        }
      }
    }
  }

  /**
   * Step enough times to cover `seconds` of simulated time (fractions of a timestep
   * are carried over to the next call, so repeated small calls do not drift).
   * @returns {number} number of mj_step calls made
   */
  advance(seconds) {
    this._pendingTime += seconds;
    const n = Math.floor(this._pendingTime / this.timestep + 1e-9);
    if (n > 0) {
      this._pendingTime -= n * this.timestep;
      this.step(n);
    }
    return Math.max(0, n);
  }

  /**
   * Reset the state to qpos0 / zero velocity / time 0 and zero the work integrals.
   * Activations are KEPT (and re-applied); call setActivations({...: 0}) to relax.
   */
  reset() {
    const { mujoco, model, data } = this;
    mujoco.mj_resetData(model, data);
    for (let i = 0; i < this.tendons.length; i++) this._applyCtrl(i);
    this._work.fill(0);
    this._pendingTime = 0;
    mujoco.mj_forward(model, data);
  }

  /** Pulling force of the tendon's actuator (N, >= 0), or the passive spring force for type "none". */
  _tension(i) {
    const t = this.tendons[i];
    const d = this.data;
    if (t.actuatorId !== null) {
      return Math.max(0, -d.actuator_force[t.actuatorId] * this.model.actuator_gear[6 * t.actuatorId]);
    }
    return this._passiveTension(i);
  }

  _passiveTension(i) {
    const t = this.tendons[i];
    if (!t.stiffness) return 0;
    const L = this.data.ten_length[t.tendonId];
    return Math.max(0, t.stiffness * (L - t.springLength));
  }

  _tipInBase() {
    const { xpos, xmat } = this.data;
    const b = this._frameBody[0], tip = this._frameBody[this.numSegments - 1];
    return toFrame(xpos, 3 * b, xmat, 9 * b, xpos, 3 * tip);
  }

  /**
   * Measure the current state (DESIGN.md "Measurement"). Angles in degrees, lengths in
   * metres, forces in N, torques in N·m. Positions are expressed in the base frame
   * (segment_0_vertebrae), so "upright" and "hanging" give identical numbers.
   *
   * Per tendon, `force` is the actuator's pulling force (or, for a passive tendon,
   * its elastic force); `torque` = -moment_arm * force, so for actuated tendons the
   * per-tendon torques sum to `segments[].torque` (= qfrc_actuator).
   *
   * @param {{torques?: boolean}} [options]  torques:false skips joint torques and moment arms (faster)
   * @returns {object} Measurement
   */
  measure({ torques = true } = {}) {
    const { model, data } = this;
    const n = this.numSegments;
    const { xpos, xmat, qpos, qfrc_actuator, qfrc_passive, qfrc_constraint } = data;
    const fb = this._frameBody;
    const b0 = fb[0];

    const tipPos = this._tipInBase();
    const disp = [tipPos[0] - this._tip0[0], tipPos[1] - this._tip0[1], tipPos[2] - this._tip0[2]];
    const tip = {
      pos: tipPos, displacement: disp, distance: Math.hypot(disp[0], disp[1], disp[2]),
      ventral: 0, lateral: 0, bend: 0, twist: 0,
    };
    relAngles(xmat, 9 * b0, xmat, 9 * fb[n - 1], tip);

    const segments = [];
    const cumulative = [0];
    const vj = (arr, i) => {
      const o = {};
      for (let k = 0; k < 3; k++) {
        const dof = this._vjDof[i * 3 + k];
        o[AXES[k]] = dof >= 0 ? arr[dof] : 0;
      }
      return o;
    };
    for (let i = 1; i < n; i++) {
      const s = {
        index: i, pos: toFrame(xpos, 3 * b0, xmat, 9 * b0, xpos, 3 * fb[i]),
        ventral: 0, lateral: 0, bend: 0, twist: 0,
      };
      relAngles(xmat, 9 * fb[i - 1], xmat, 9 * fb[i], s);
      const ja = {};
      for (let k = 0; k < 3; k++) {
        const qa = this._vjQpos[i * 3 + k];
        ja[AXES[k]] = qa >= 0 ? qpos[qa] * RAD2DEG : 0;
      }
      s.joint_angle = ja;
      if (torques) {
        s.torque = vj(qfrc_actuator, i);
        s.passive_torque = vj(qfrc_passive, i);
        s.limit_torque = vj(qfrc_constraint, i);
      }
      segments.push(s);
      cumulative.push(cumulative[i - 1] + s.ventral);
    }

    const tendons = this.tendons.map((t, ti) => {
      const length = data.ten_length[t.tendonId];
      const excursion = t.length0 - length;
      const force = this._tension(ti);
      const out = {
        name: t.name, group: t.group, activation: this._act[ti], length, length0: t.length0,
        excursion, strain: t.length0 > 0 ? excursion / t.length0 : 0, force,
        passive_force: this._passiveTension(ti), work: this._work[ti],
      };
      if (torques) {
        // sparse row of the tendon Jacobian (MuJoCo >= 3.x: static sparsity in the model)
        const ma = this._ma;
        ma.fill(0);
        const nnz = model.ten_J_rownnz[t.tendonId], adr = model.ten_J_rowadr[t.tendonId];
        const colind = model.ten_J_colind, J = data.ten_J;
        for (let k = 0; k < nnz; k++) {
          const slot = this._dofToVJ[colind[adr + k]];
          if (slot >= 0) ma[slot] = J[adr + k];
        }
        const arms = [], tq = [];
        for (let i = 1; i < n; i++) {
          const p = ma[i * 3], r = ma[i * 3 + 1], y = ma[i * 3 + 2];
          arms.push({ segment: i, pitch: p, roll: r, yaw: y });
          tq.push({ segment: i, pitch: -p * force, roll: -r * force, yaw: -y * force });
        }
        out.moment_arms = arms;
        out.torque = tq;
      }
      return out;
    });

    return { time: data.time, tip, segments, cumulative_ventral: cumulative, tendons };
  }

  /** Free the MuJoCo model and data (WASM memory). The object is unusable afterwards. */
  dispose() {
    if (this.data) { this.data.delete(); this.data = null; }
    if (this.model) { this.model.delete(); this.model = null; }
  }
}

// render3d.js — three.js view of a compiled MuJoCo model (meshes, primitives, tendons, sites).
//
//   const viewer = new Viewer3D(document.getElementById("view"), {plateOpacity: 0.5});
//   viewer.setSimulation(sim);           // after every Simulation.create()
//   function frame() { sim.advance(1/60); viewer.update(); requestAnimationFrame(frame); }
//
// Geometry is built from the compiled MjModel (mesh_vert / mesh_face), so MuJoCo's mesh
// re-centring is respected and poses come straight from data.geom_xpos / geom_xmat.
// Meshes with identical content share one BufferGeometry. Tendons are drawn as instanced
// cylinders between consecutive path points (data.wrap_xpos), skipping pulley markers.
// The world is Z-up (MuJoCo convention).

import * as THREE from "../vendor/three/three.module.js";
import { OrbitControls } from "../vendor/three/addons/OrbitControls.js";

const DEFAULTS = {
  background: "#f4f5f7",
  showSites: false,
  showStruts: true,
  plateOpacity: 1.0,
  /** minimum drawn radius of user tendons (m); MuJoCo widths are often tiny */
  tendonMinRadius: 0.0007,
  /** radius multiplier for the highlighted tendon */
  highlightScale: 2.0,
};

const PLATE_GEOM = /^segment_\d+_plate_\d+_/;
const STRUT_TENDON = /^segment_\d+_vertebral_/;

// mjtGeom
const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

/** FNV-1a over the raw bytes of a typed-array slice (for mesh de-duplication). */
function hashSlice(arr, start, end, h = 0x811c9dc5) {
  const u32 = new Uint32Array(arr.buffer, arr.byteOffset + start * 4, end - start);
  for (let i = 0; i < u32.length; i++) {
    h ^= u32[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/** scratch objects (no per-frame allocation) */
const _m = new THREE.Matrix4();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _d = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
const _Y = new THREE.Vector3(0, 1, 0);
const _c = new THREE.Color();

/**
 * Interactive 3D view of a Simulation.
 */
export class Viewer3D {
  /**
   * @param {HTMLElement} container  element to fill (the canvas is appended to it)
   * @param {object} [options]  {background, showSites, showStruts, plateOpacity, tendonMinRadius, highlightScale}
   */
  constructor(container, options = {}) {
    this.container = container;
    this.options = { ...DEFAULTS, ...options };
    this.sim = null;
    this.highlight = null;

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(this.options.background);
    this.scene = scene;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(1, -1.2, 1.5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-1, 1, -0.5);
    scene.add(fill);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 50);
    camera.up.set(0, 0, 1);
    this.camera = camera;
    this.controls = new OrbitControls(camera, renderer.domElement);
    this.controls.addEventListener("change", () => this.requestRender());

    this.root = new THREE.Group(); // everything model-specific lives here
    scene.add(this.root);

    this._renderQueued = false;
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);
    this.resize();
  }

  // ------------------------------------------------------------------------
  // scene construction
  // ------------------------------------------------------------------------

  /**
   * (Re)build the scene for a simulation. Call after every Simulation.create();
   * the camera is only re-fitted the first time (or when `refitCamera` is true).
   * @param {import("./sim.js").Simulation|null} sim
   * @param {{refitCamera?: boolean}} [opts]
   */
  setSimulation(sim, { refitCamera } = {}) {
    const first = !this.sim && !this._fitted;
    this._clearModel();
    this.sim = sim;
    if (!sim) { this.requestRender(); return; }
    const { model, mujoco } = sim;
    const name = (type, id) => mujoco.mj_id2name(model, type, id) || "";
    const OBJ_GEOM = mujoco.mjtObj.mjOBJ_GEOM.value;
    const OBJ_TENDON = mujoco.mjtObj.mjOBJ_TENDON.value;

    // --- geoms -------------------------------------------------------------
    const meshGeoms = new Map();      // mesh id -> BufferGeometry
    const byHash = new Map();         // content hash -> BufferGeometry
    const mv = model.mesh_vert, mf = model.mesh_face;
    const vadr = model.mesh_vertadr, vnum = model.mesh_vertnum;
    const fadr = model.mesh_faceadr, fnum = model.mesh_facenum;
    const getMeshGeometry = (id) => {
      if (meshGeoms.has(id)) return meshGeoms.get(id);
      const v0 = vadr[id] * 3, v1 = v0 + vnum[id] * 3;
      const f0 = fadr[id] * 3, f1 = f0 + fnum[id] * 3;
      const key = `${vnum[id]}:${fnum[id]}:${hashSlice(mv, v0, v1)}:${hashSlice(mf, f0, f1)}`;
      let g = byHash.get(key);
      if (!g) {
        g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(mv.slice(v0, v1), 3));
        const idx = vnum[id] < 65536 ? new Uint16Array(f1 - f0) : new Uint32Array(f1 - f0);
        idx.set(mf.subarray(f0, f1));
        g.setIndex(new THREE.BufferAttribute(idx, 1));
        g.computeBoundingSphere();
        byHash.set(key, g);
        this._disposables.push(g);
      }
      meshGeoms.set(id, g);
      return g;
    };

    const gtype = model.geom_type, gsize = model.geom_size, grgba = model.geom_rgba;
    const ggroup = model.geom_group, gdata = model.geom_dataid, gmat = model.geom_matid;
    const matRgba = model.mat_rgba;
    this._geomObjs = [];
    for (let i = 0; i < model.ngeom; i++) {
      if (ggroup[i] > 2) continue; // MuJoCo's default viewer hides groups >= 3
      let r = grgba[4 * i], g = grgba[4 * i + 1], b = grgba[4 * i + 2], a = grgba[4 * i + 3];
      if (gmat[i] >= 0 && r === 0.5 && g === 0.5 && b === 0.5 && a === 1) {
        const k = gmat[i] * 4; r = matRgba[k]; g = matRgba[k + 1]; b = matRgba[k + 2]; a = matRgba[k + 3];
      }
      if (a === 0) continue;
      const isPlate = PLATE_GEOM.test(name(OBJ_GEOM, i));
      let geometry;
      const s0 = gsize[3 * i], s1 = gsize[3 * i + 1], s2 = gsize[3 * i + 2];
      switch (gtype[i]) {
        case GEOM.MESH: geometry = getMeshGeometry(gdata[i]); break;
        case GEOM.SPHERE: geometry = this._prim(new THREE.SphereGeometry(s0, 20, 14)); break;
        case GEOM.ELLIPSOID: geometry = this._prim(new THREE.SphereGeometry(1, 20, 14).scale(s0, s1, s2)); break;
        case GEOM.CAPSULE: geometry = this._prim(new THREE.CapsuleGeometry(s0, 2 * s1, 6, 16).rotateX(Math.PI / 2)); break;
        case GEOM.CYLINDER: geometry = this._prim(new THREE.CylinderGeometry(s0, s0, 2 * s1, 20).rotateX(Math.PI / 2)); break;
        case GEOM.BOX: geometry = this._prim(new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2)); break;
        case GEOM.PLANE: geometry = this._prim(new THREE.PlaneGeometry(2 * (s0 || 1), 2 * (s1 || 1))); break;
        default: continue; // hfield / sdf / flex not used by this model
      }
      const material = this._material(r, g, b, a, isPlate, gtype[i] === GEOM.MESH);
      const obj = new THREE.Mesh(geometry, material);
      obj.matrixAutoUpdate = false;
      obj.userData.geomId = i;
      this.root.add(obj);
      this._geomObjs.push(obj);
    }

    // --- tendons: instanced cylinders + joint caps --------------------------
    const cyl = this._prim(new THREE.CylinderGeometry(1, 1, 1, 10, 1, true));
    const cap = this._prim(new THREE.SphereGeometry(1, 12, 8));
    const cap2 = Math.max(16, 2 * model.nwrap);
    this._strutMesh = new THREE.InstancedMesh(cyl, this._basic(new THREE.MeshStandardMaterial({ color: 0x9aa0a8, roughness: 0.8 })), cap2);
    this._tendonMesh = new THREE.InstancedMesh(cyl, this._basic(new THREE.MeshStandardMaterial({ roughness: 0.45 })), cap2);
    this._tendonCaps = new THREE.InstancedMesh(cap, this._basic(new THREE.MeshStandardMaterial({ roughness: 0.45 })), cap2);
    for (const im of [this._strutMesh, this._tendonMesh, this._tendonCaps]) {
      im.frustumCulled = false;
      im.count = 0;
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.root.add(im);
    }
    this._tendonMesh.setColorAt(0, _c.set(0xffffff)); // allocate instanceColor
    this._tendonCaps.setColorAt(0, _c.set(0xffffff));
    this._tendonInfo = [];
    const trgba = model.tendon_rgba, twidth = model.tendon_width;
    for (let t = 0; t < model.ntendon; t++) {
      const nm = name(OBJ_TENDON, t);
      const strut = STRUT_TENDON.test(nm);
      this._tendonInfo.push({
        id: t, name: nm, strut,
        radius: strut ? Math.max(0.00025, twidth[t] * 0.5) : Math.max(this.options.tendonMinRadius, twidth[t]),
        color: new THREE.Color().setRGB(trgba[4 * t], trgba[4 * t + 1], trgba[4 * t + 2], THREE.SRGBColorSpace),
      });
    }

    // --- sites ----------------------------------------------------------------
    const nsite = model.nsite;
    this._siteMesh = new THREE.InstancedMesh(cap, this._basic(new THREE.MeshStandardMaterial({ roughness: 0.6 })), Math.max(1, nsite));
    this._siteMesh.frustumCulled = false;
    this._siteRadius = new Float32Array(nsite);
    const srgba = model.site_rgba, ssize = model.site_size;
    for (let i = 0; i < nsite; i++) {
      this._siteRadius[i] = Math.max(0.0004, ssize[3 * i]);
      this._siteMesh.setColorAt(i, _c.setRGB(srgba[4 * i], srgba[4 * i + 1], srgba[4 * i + 2], THREE.SRGBColorSpace));
    }
    this._siteMesh.count = nsite;
    this.root.add(this._siteMesh);

    this._applyVisibility();
    this.update();
    if (first || refitCamera) this.fitCamera();
  }

  _prim(geometry) { this._disposables.push(geometry); return geometry; }
  _basic(material) { this._disposables.push(material); return material; }

  _material(r, g, b, a, isPlate, isMesh) {
    const key = `${r},${g},${b},${a},${isPlate}`;
    let mat = this._materials.get(key);
    if (!mat) {
      mat = new THREE.MeshStandardMaterial({
        roughness: 0.65, metalness: 0.0, flatShading: isMesh,
      });
      mat.color.setRGB(r, g, b, THREE.SRGBColorSpace);
      mat.userData = { baseOpacity: a, isPlate };
      this._materials.set(key, mat);
      this._disposables.push(mat);
    }
    return mat;
  }

  _clearModel() {
    for (const d of this._disposables || []) d.dispose();
    for (const im of [this._strutMesh, this._tendonMesh, this._tendonCaps, this._siteMesh]) im?.dispose();
    this.root.clear();
    this._disposables = [];
    this._materials = new Map();
    this._geomObjs = [];
    this._tendonInfo = [];
    this._strutMesh = this._tendonMesh = this._tendonCaps = this._siteMesh = null;
  }

  _applyVisibility() {
    const o = this.options;
    for (const mat of this._materials.values()) {
      const op = mat.userData.baseOpacity * (mat.userData.isPlate ? o.plateOpacity : 1);
      const transparent = op < 0.999;
      mat.opacity = op;
      mat.transparent = transparent;
      mat.depthWrite = !transparent;
      mat.visible = op > 0.001;
      mat.needsUpdate = true;
    }
    if (this._strutMesh) this._strutMesh.visible = o.showStruts;
    if (this._siteMesh) this._siteMesh.visible = o.showSites;
    this.scene.background = new THREE.Color(o.background);
  }

  // ------------------------------------------------------------------------
  // per-frame update
  // ------------------------------------------------------------------------

  /** Copy poses from sim.data into the scene and render. Call once per animation frame. */
  update() {
    const sim = this.sim;
    if (sim && sim.data) this._sync(sim.data);
    this.render();
  }

  _sync(data) {
    const xpos = data.geom_xpos, xmat = data.geom_xmat;
    for (const obj of this._geomObjs) {
      const i = obj.userData.geomId, p = 3 * i, r = 9 * i;
      obj.matrix.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1);
      obj.matrixWorldNeedsUpdate = true;
    }

    // tendons
    const wadr = data.ten_wrapadr, wnum = data.ten_wrapnum, wobj = data.wrap_obj, wx = data.wrap_xpos;
    let ns = 0, nt = 0, nc = 0;
    const strutMesh = this._strutMesh, tMesh = this._tendonMesh, caps = this._tendonCaps;
    const hl = this.highlight;
    for (const info of this._tendonInfo) {
      const adr = wadr[info.id], num = wnum[info.id];
      const isHl = hl !== null && info.name === hl;
      const rad = info.radius * (isHl ? this.options.highlightScale : 1);
      if (!info.strut) {
        _c.copy(info.color);
        if (hl !== null && !isHl) _c.lerp(_WHITE, 0.45); // fade the others while one is highlighted
      }
      for (let j = adr; j < adr + num; j++) {
        if (wobj[j] === -2) continue; // pulley marker
        if (!info.strut && nc < caps.instanceMatrix.count) {
          _s.set(rad * 1.15, rad * 1.15, rad * 1.15);
          _m.compose(_a.set(wx[3 * j], wx[3 * j + 1], wx[3 * j + 2]), _q.identity(), _s);
          caps.setMatrixAt(nc, _m);
          caps.setColorAt(nc, _c);
          nc++;
        }
        if (j + 1 >= adr + num || wobj[j + 1] === -2) continue;
        _a.set(wx[3 * j], wx[3 * j + 1], wx[3 * j + 2]);
        _b.set(wx[3 * j + 3], wx[3 * j + 4], wx[3 * j + 5]);
        _d.subVectors(_b, _a);
        const len = _d.length();
        if (len < 1e-9) continue;
        _q.setFromUnitVectors(_Y, _d.multiplyScalar(1 / len));
        _s.set(rad, len, rad);
        _m.compose(_a.add(_b).multiplyScalar(0.5), _q, _s);
        if (info.strut) {
          if (ns < strutMesh.instanceMatrix.count) strutMesh.setMatrixAt(ns++, _m);
        } else if (nt < tMesh.instanceMatrix.count) {
          tMesh.setMatrixAt(nt, _m);
          tMesh.setColorAt(nt, _c);
          nt++;
        }
      }
    }
    strutMesh.count = ns; tMesh.count = nt; caps.count = nc;
    strutMesh.instanceMatrix.needsUpdate = true;
    tMesh.instanceMatrix.needsUpdate = true;
    caps.instanceMatrix.needsUpdate = true;
    if (tMesh.instanceColor) tMesh.instanceColor.needsUpdate = true;
    if (caps.instanceColor) caps.instanceColor.needsUpdate = true;

    // sites
    if (this.options.showSites && this._siteMesh) {
      const sx = data.site_xpos, sr = this._siteRadius;
      for (let i = 0; i < sr.length; i++) {
        _s.set(sr[i], sr[i], sr[i]);
        _m.compose(_a.set(sx[3 * i], sx[3 * i + 1], sx[3 * i + 2]), _q.identity(), _s);
        this._siteMesh.setMatrixAt(i, _m);
      }
      this._siteMesh.instanceMatrix.needsUpdate = true;
    }
  }

  /** Render now. */
  render() {
    this._renderQueued = false;
    this.renderer.render(this.scene, this.camera);
  }

  /** Render on the next animation frame (coalesced). Used for camera moves while paused. */
  requestRender() {
    if (this._renderQueued) return;
    this._renderQueued = true;
    requestAnimationFrame(() => { if (this._renderQueued) this.render(); });
  }

  // ------------------------------------------------------------------------
  // options, camera, lifecycle
  // ------------------------------------------------------------------------

  /** Highlight one tendon (thicker; other user tendons faded) or none (null). */
  setTendonHighlight(name) {
    this.highlight = name ?? null;
    if (this.sim?.data) this._sync(this.sim.data);
    this.requestRender();
  }

  /** Change options, e.g. {plateOpacity: 0.3, showSites: true}. */
  setOptions(partial) {
    Object.assign(this.options, partial);
    if ("tendonMinRadius" in partial && this._tendonInfo) {
      for (const info of this._tendonInfo) {
        if (!info.strut) info.radius = Math.max(this.options.tendonMinRadius, this.sim.model.tendon_width[info.id]);
      }
    }
    this._applyVisibility();
    if (this.sim?.data) this._sync(this.sim.data);
    this.requestRender();
  }

  /**
   * Point the camera at the whole model, seen from the sinistral side (+y) so that
   * ventral (+x) bending is in the image plane.
   * @param {"side"|"front"|"top"} [view="side"]
   */
  fitCamera(view = "side") {
    const box = new THREE.Box3();
    for (const obj of this._geomObjs) {
      obj.updateMatrixWorld();
      if (!obj.geometry.boundingSphere) obj.geometry.computeBoundingSphere();
      const bs = obj.geometry.boundingSphere;
      _a.copy(bs.center).applyMatrix4(obj.matrixWorld);
      box.expandByPoint(_b.copy(_a).addScalar(bs.radius));
      box.expandByPoint(_b.copy(_a).addScalar(-bs.radius));
    }
    if (box.isEmpty()) box.set(new THREE.Vector3(-0.1, -0.1, 0), new THREE.Vector3(0.1, 0.1, 0.4));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    const dist = size / (2 * Math.tan((this.camera.fov * Math.PI) / 360)) * 1.05;
    const dir = view === "front" ? new THREE.Vector3(1, 0, 0.15)
      : view === "top" ? new THREE.Vector3(0.01, 0, 1)
        : new THREE.Vector3(0.35, 1, 0.25); // from +y (sinistral) slightly ventral
    this.camera.position.copy(center).addScaledVector(dir.normalize(), dist);
    this.camera.near = dist / 200;
    this.camera.far = dist * 20;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
    this._fitted = true;
    this.requestRender();
  }

  /** Match the canvas to the container size (called automatically via ResizeObserver). */
  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, true);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  /** Renderer statistics of the last frame: {calls, triangles}. */
  get stats() {
    const r = this.renderer.info.render;
    return { calls: r.calls, triangles: r.triangles };
  }

  /** Free GPU resources and remove the canvas. */
  dispose() {
    this._resizeObserver.disconnect();
    this.controls.dispose();
    this._clearModel();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.sim = null;
  }
}

const _WHITE = new THREE.Color(1, 1, 1);

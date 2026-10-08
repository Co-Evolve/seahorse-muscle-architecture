// Seahorse muscle playground: UI shell. Loads the real modules (sim.js, render3d.js, slice_editor.js,
// config.js) and falls back to clearly-labelled stand-ins (mock_sim.js) when one is missing.
// Owns the app state: current config (+ undo history, autosave), the live Simulation,
// activations, play/pause, and wires the panels together.

import { h, clear, icon, debounce, deepCopy, fmt, installTooltips, toast, downloadText, pickFile, safeFileName, attachMenu, segmented, toggle } from "./ui.js";
import { storage, library, History } from "./store.js";
import { isActuated } from "./config_tools.js";
import { ParamsPanel } from "./params_panel.js";
import { TendonPanel } from "./tendon_panel.js";
import { AnalysisDock, NMM, MM } from "./analysis.js";
import { ExperimentsView } from "./experiments.js";
import { HelpOverlay } from "./help.js";
import { meter } from "./plots.js";

const $ = (id) => document.getElementById(id);

const state = {
  mods: {}, stand: { sim: false, viewer: false, slice: false },
  mujoco: null, assets: null, catalog: null,
  config: null, history: new History(), selected: null, segment: 0,
  sim: null, buildToken: 0, buildError: null, excluded: [],
  activations: {}, playing: true, speed: 1,
  issues: [], view: "design", leftTab: "tendons",
  lastMeasure: null, simRate: 1,
};
window.__tendonDesigner = state; // handy for debugging in the browser console

// ------------------------------------------------------------------ module loading
async function tryImport(path) {
  try { return await import(path); } catch (e) { console.warn(`[muscle playground] could not load ${path}:`, e); return null; }
}

async function loadModules() {
  const [simMod, viewMod, sliceMod, cfgMod] = await Promise.all([
    tryImport("./sim.js"), tryImport("./render3d.js"), tryImport("./slice_editor.js"), tryImport("./config.js")]);
  const mock = (simMod?.Simulation && viewMod?.Viewer3D && sliceMod?.SliceEditor) ? null : await import("./mock_sim.js");
  state.mods.simMod = simMod;
  state.mods.Simulation = simMod?.Simulation || mock.MockSimulation;
  state.mods.Viewer3D = viewMod?.Viewer3D || mock.MockViewer3D;
  state.mods.SliceEditor = sliceMod?.SliceEditor || mock.MockSliceEditor;
  state.mods.cfg = cfgMod;
  state.mods.mock = mock;
  state.stand.sim = !simMod?.Simulation;
  state.stand.viewer = !viewMod?.Viewer3D;
  state.stand.slice = !sliceMod?.SliceEditor;
}

/** Config helpers for the panels; config.js is the single source of truth. */
function makeHelpers() {
  const C = state.mods.cfg;
  if (!C) throw new Error("config.js could not be loaded, so configurations cannot be read or checked.");
  const cat = () => state.catalog;
  return {
    normalise: (raw) => C.normaliseConfig(raw),
    validate: (config) => C.validateConfig(config, cat()),
    newConfig: (name) => C.newConfig(name),
    newTendon: (config, o) => C.newTendon(config, o),
    uniqueName: (config, base) => C.uniqueTendonName(config, base),
    pointInfo: (id, config) => C.pointInfo(id, cat(), config),
    describePoint: (id, config) => C.describePoint(id, cat(), config),
    removeTrunkPoint: (t, k) => C.removeTrunkPoint(t, k),
    mirror: (config, name) => { try { return C.addMirroredTendon(config, name, cat()); } catch (e) { toast(e.message, "error"); return null; } },
    replaceConfig: (config, key) => setConfig(config, { key }),
    notify: (msg, kind) => toast(msg, kind, 6000),
  };
}
let H = null;

// ------------------------------------------------------------------ boot
async function boot() {
  installTooltips();
  buildStaticChrome();
  setLoading("Loading the app…");
  await loadModules();
  H = makeHelpers();
  try {
    if (!state.stand.sim) {
      setLoading("Loading the tail model…");
      state.assets = await state.mods.simMod.loadAssets("model/", { onProgress: (a, b) => setLoading(`Loading 3D parts of the tail… ${a} / ${b}`) });
      setLoading("Starting the physics engine (MuJoCo)…");
      try {
        state.mujoco = await state.mods.simMod.loadMujoco();
      } catch (e) {
        console.error(e);
        useStandInSim(`The MuJoCo physics engine could not start (${e.message}).`);
      }
    } else {
      const catalog = await (await fetch("model/catalog.json")).json();
      state.assets = { baseXml: "", catalog, meshes: {} };
    }
  } catch (e) {
    console.error(e);
    setLoading(`Could not load the tail model: ${e.message}. Run export_assets.py, then reload this page.`, true);
    return;
  }
  state.catalog = state.assets.catalog;
  updateModeBadge();
  if (state.stand.sim && !state.mujoco) banner("standin", "warning", "The physics module (sim.js) could not load. A simplified stand-in model is used: the numbers are NOT physically meaningful.");

  // initial configuration: autosave > first real preset > empty
  let cfg = null;
  const saved = storage.get("current");
  if (saved) { try { cfg = H.normalise(saved); } catch { cfg = null; } }
  if (!cfg) {
    const presets = await listPresets();
    const first = presets.find((p) => !/empty/i.test(p.name)) || presets[0];
    if (first) { try { cfg = await fetchPreset(first); } catch { cfg = null; } }
  }
  if (!cfg) cfg = H.newConfig("My configuration");
  state.selected = cfg.tendons[0]?.name || null;
  state.history.reset(cfg);

  buildPanels();
  setConfig(cfg, { noHistory: true, immediate: true });
  requestAnimationFrame(frame);
  if (!storage.get("help-seen")) { help.open(); storage.set("help-seen", true); }
}

function setLoading(text, isError = false) {
  const el = $("loading");
  if (!text) { el.classList.add("done"); setTimeout(() => { el.hidden = true; }, 400); return; }
  el.hidden = false; el.classList.remove("done");
  el.classList.toggle("error", isError);
  $("loading-text").textContent = text;
}

function useStandInSim(reason) {
  state.mods.Simulation = state.mods.mock?.MockSimulation;
  state.stand.sim = true;
  if (!state.mods.Simulation) {
    import("./mock_sim.js").then((m) => { state.mods.mock = m; state.mods.Simulation = m.MockSimulation; scheduleRebuild.flush(); });
  }
  banner("standin", "warning", `${reason} Using a simplified stand-in model: the numbers are NOT physically meaningful.`);
}

function updateModeBadge() {
  const stand = Object.entries(state.stand).filter(([, v]) => v).map(([k]) => ({ sim: "physics", viewer: "3D view", slice: "slice editor" }[k]));
  const b = $("mode-badge");
  b.hidden = !stand.length;
  b.textContent = stand.length ? "Stand-in parts" : "";
  b.dataset.tip = stand.length ? `These parts are simplified stand-ins because the real module could not load: ${stand.join(", ")}.` : "";
}

// ------------------------------------------------------------------ static chrome (top bar)
let help, analysis, params, tendonPanel, slice, viewer, experiments, viewSwitch;

function buildStaticChrome() {
  help = new HelpOverlay();
  $("btn-help").addEventListener("click", () => help.open());
  $("btn-undo").append(icon("undo", 16));
  $("btn-redo").append(icon("redo", 16));
  $("btn-undo").addEventListener("click", undo);
  $("btn-redo").addEventListener("click", redo);
  $("btn-new").prepend(icon("file", 15));
  $("btn-open").prepend(icon("open", 15));
  $("btn-save").prepend(icon("save", 15));
  $("btn-presets").prepend(icon("layers", 15));
  $("btn-export").prepend(icon("code", 15));
  $("btn-presets").append(icon("chevron", 14));
  $("btn-save").append(icon("chevron", 14));
  $("btn-open").append(icon("chevron", 14));

  $("btn-new").addEventListener("click", () => {
    setConfig(H.newConfig("My configuration"), { key: "new" });
    state.selected = null;
    renderAll();
    toast("Started a new configuration. Undo (Ctrl+Z) brings the previous one back.");
  });
  attachMenu($("btn-open"), () => {
    const items = [{ label: "Open a file…", hint: ".json configuration", onClick: openFile }];
    const lib = library.list();
    items.push({ separator: true }, { heading: "My library" });
    if (!lib.length) items.push({ label: "Empty: use Save → Save to my library", disabled: true });
    for (const l of lib) items.push({ label: l.name, hint: new Date(l.saved).toLocaleString(), onClick: () => loadConfig(l.config, `Opened “${l.name}” from your library`) });
    return items;
  });
  attachMenu($("btn-save"), () => {
    const inLib = library.get(state.config.name);
    return [
      { label: "Download file", hint: "keep it for good (.json)", onClick: saveFile },
      { label: inLib ? "Update in my library" : "Save to my library", hint: "kept in this browser", onClick: saveToLibrary },
      ...(inLib ? [{ label: `Remove “${state.config.name}” from my library`, onClick: () => { library.remove(state.config.name); toast("Removed from your library"); } }] : []),
    ];
  });
  attachMenu($("btn-presets"), async () => {
    const presets = await listPresets();
    if (!presets.length) return [{ label: "No presets found (web/configs/index.json)", disabled: true }];
    return presets.map((p) => ({ label: p.name, hint: p.description, onClick: async () => {
      try { loadConfig(await fetchPreset(p), `Loaded preset “${p.name}”`); } catch (e) { toast(e.message, "error", 6000); }
    } }));
  });
  $("btn-export").addEventListener("click", exportMjcf);
  const nameIn = $("config-name");
  nameIn.addEventListener("change", () => editConfig((c) => { c.name = nameIn.value.trim() || "Untitled"; }, "name"));
  nameIn.addEventListener("keydown", (e) => { if (e.key === "Enter") nameIn.blur(); });

  viewSwitch = segmented([{ value: "design", label: "Design", icon: "edit" }, { value: "experiments", label: "Experiments", icon: "flask" }], "design", (v) => setView(v), { label: "View" });
  $("view-switch").append(viewSwitch);

  const themeBtn = $("btn-theme");
  const applyTheme = () => {
    const t = storage.get("theme", "auto");
    if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
    clear(themeBtn).append(icon(t === "dark" ? "moon" : t === "light" ? "sun" : "contrast", 16));
    themeBtn.dataset.tip = `Colour theme: ${t === "auto" ? "follows your computer" : t}. Click to change.`;
    viewer?.setOptions?.({ background: viewportBackground() });
  };
  themeBtn.addEventListener("click", () => {
    const order = ["auto", "light", "dark"];
    const t = storage.get("theme", "auto");
    storage.set("theme", order[(order.indexOf(t) + 1) % 3]);
    applyTheme();
  });
  applyTheme();
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", applyTheme);

  document.addEventListener("keydown", onKey);
}

function onKey(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = ["input", "textarea", "select"].includes(tag) || e.target.isContentEditable;
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === "z" && !typing) { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
  if (mod && e.key.toLowerCase() === "y" && !typing) { e.preventDefault(); redo(); return; }
  if (typing || mod || e.altKey) return;
  if (e.key === " " && tag !== "button" && !e.target.closest?.(".se-pop, [role=menu]") && state.view === "design" && !help.isOpen) { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.key === "?") { help.open(); }
}

// ------------------------------------------------------------------ panels
function buildPanels() {
  // left tabs
  const tabs = segmented([{ value: "tendons", label: "Tendons" }, { value: "body", label: "Body parameters" }], state.leftTab, (v) => {
    state.leftTab = v; $("pane-tendons").hidden = v !== "tendons"; $("pane-body").hidden = v !== "body";
  }, { label: "Left panel" });
  $("left-tabs").append(tabs);

  tendonPanel = new TendonPanel($("tendon-list"), $("tendon-detail"), {
    catalog: state.catalog, helpers: H,
    edit: (fn, key) => editConfig(fn, key),
    select: (name, oldName) => selectTendon(name, oldName),
  });
  params = new ParamsPanel($("params-panel"), state.catalog, {
    onChange: (p, key) => editConfig((c) => { c.params = p; }, key, { fromParams: true }),
  });

  // segment picker (the real SliceEditor has its own segment filmstrip) + slice editor
  if (state.stand.slice) buildSegmentPicker(); else { $("segment-picker").hidden = true; $("slice-head").hidden = true; }
  try {
    slice = new state.mods.SliceEditor($("slice-editor"), state.catalog, {
      onChange: (cfg) => setConfig(cfg, { key: "slice", fromEditor: true }),
      onSelectTendon: (name) => selectTendon(name, null, { fromEditor: true }),
      onSegmentChange: (i) => { state.segment = i; updateSegmentPicker(); },
    });
  } catch (e) {
    console.error(e);
    slice = null;
    $("slice-editor").append(h("div", { class: "issue error" }, `The slice editor could not start: ${e.message}`));
  }

  // 3D viewer
  try {
    viewer = new state.mods.Viewer3D($("viewer3d"), viewerOptions());
  } catch (e) {
    console.error(e);
    state.mods.Viewer3D = state.mods.mock ? state.mods.mock.MockViewer3D : null;
    state.stand.viewer = true; updateModeBadge();
    if (state.mods.Viewer3D) viewer = new state.mods.Viewer3D($("viewer3d"), viewerOptions());
  }

  if (viewer?.fitCamera) {
    const cam = segmented([{ value: "side", label: "Side" }, { value: "front", label: "Front" }, { value: "top", label: "Top" }], null,
      (v) => { try { viewer.fitCamera(v); } catch (e) { console.error(e); } cam.setValue(null); }, { label: "Camera", small: true });
    cam.classList.add("cam-switch");
    cam.querySelectorAll(".seg-btn").forEach((b) => { b.dataset.tip = `Look at the tail from the ${b.dataset.value}`; });
    $("viewport-status").append(cam);
  }

  analysis = new AnalysisDock($("analysis-dock"));
  buildSimControls();
  experiments = new ExperimentsView($("view-experiments"), {
    getConfig: () => state.config,
    createSim: (cfg) => createSim(simulatable(cfg).config),
    createViewer: (el) => new state.mods.Viewer3D(el, viewerOptions()),
    listPresets, fetchPreset, attachMenu,
    parseConfigText: (text) => parseConfigText(text),
    normalise: (c) => H.normalise(c),
    validate: (c) => H.validate(c),
  });
}

function viewportBackground() {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--viewport-bg").trim();
  return v || "#0e2a31";
}
function viewerOptions() { return { background: viewportBackground(), showSites: true, showStruts: true, plateOpacity: 0.55 }; }

function buildSegmentPicker() {
  const el = $("segment-picker");
  clear(el);
  const n = state.catalog?.num_segments || 11;
  el.append(h("button", { class: "icon-btn", "aria-label": "Previous segment", tip: "Previous segment (towards the base)", onClick: () => setSegment(state.segment - 1) }, icon("chevron", 14)));
  const strip = h("div", { class: "seg-strip" });
  for (let i = 0; i < n; i++) {
    strip.append(h("button", { class: "seg-chip", dataset: { seg: i }, "aria-label": `Segment ${i}`, tip: i === 0 ? "Segment 0 (base, fixed)" : i === n - 1 ? `Segment ${i} (tip)` : `Segment ${i}`, onClick: () => setSegment(i) }, String(i)));
  }
  el.append(strip);
  el.append(h("button", { class: "icon-btn next", "aria-label": "Next segment", tip: "Next segment (towards the tip)", onClick: () => setSegment(state.segment + 1) }, icon("chevron", 14)));
  updateSegmentPicker();
}
function setSegment(i) {
  const n = state.catalog?.num_segments || 11;
  state.segment = Math.max(0, Math.min(n - 1, i));
  try { slice?.setSegment(state.segment); } catch (e) { console.error(e); }
  updateSegmentPicker();
}
function updateSegmentPicker() {
  const t = state.config?.tendons.find((x) => x.name === state.selected);
  const used = new Set();
  if (t) for (const id of [...t.path, ...(t.branches || []).flatMap((b) => b.path)]) { const p = H.pointInfo(id, state.config); if (p) used.add(p.segment); }
  for (const b of document.querySelectorAll(".seg-chip")) {
    const i = Number(b.dataset.seg);
    b.classList.toggle("active", i === state.segment);
    b.classList.toggle("used", used.has(i));
    b.setAttribute("aria-pressed", String(i === state.segment));
    if (t) b.style.setProperty("--chip-tendon", t.color);
  }
  const sub = $("slice-sub");
  if (sub) sub.textContent = `Segment ${state.segment}${state.segment === 0 ? " (base)" : state.segment === (state.catalog?.num_segments || 11) - 1 ? " (tip)" : ""}, seen from the base`;
}

// ------------------------------------------------------------------ config state
function editConfig(fn, key, opts = {}) {
  const next = deepCopy(state.config);
  fn(next);
  setConfig(next, { key, ...opts });
}

function setConfig(cfg, { key = null, noHistory = false, fromEditor = false, fromParams = false, immediate = false } = {}) {
  const prev = state.config;
  state.config = cfg;
  if (!noHistory) state.history.push(cfg, key, key === "slice" ? 200 : undefined);
  autosave(cfg);
  // keep activations of tendons that still exist (rename handled in selectTendon)
  for (const k of Object.keys(state.activations)) if (!cfg.tendons.some((t) => t.name === k)) delete state.activations[k];
  if (state.selected && !cfg.tendons.some((t) => t.name === state.selected)) state.selected = cfg.tendons[0]?.name || null;
  state.issues = H.validate(cfg);
  renderAll({ fromEditor, fromParams });
  // rebuild the simulation only if something that affects the model changed
  if (!prev || modelKey(prev) !== modelKey(cfg)) { if (immediate) scheduleRebuild.flush(); else scheduleRebuild(); }
  else if (state.sim) applySimMeta();
}

/** Everything that changes the compiled model (not the name / description / colours). */
function modelKey(cfg) {
  return JSON.stringify({ p: cfg.params, f: cfg.free_points, t: cfg.tendons.map((t) => ({ ...t, color: undefined })) });
}
function applySimMeta() {
  // colour changes do not need a recompile: update the live tendon metadata
  for (const t of state.sim.tendons) { const c = state.config.tendons.find((x) => x.name === t.name); if (c) t.color = c.color; }
  renderActivationPanel();
}

const autosave = debounce((cfg) => {
  const ok = storage.set("current", cfg);
  const el = $("save-state");
  el.textContent = ok ? "Saved in this browser" : "Not saved (browser storage is off)";
  el.classList.toggle("warn", !ok);
}, 400);

function undo() { const c = state.history.undo(); if (c) setConfig(c, { noHistory: true }); }
function redo() { const c = state.history.redo(); if (c) setConfig(c, { noHistory: true }); }

function selectTendon(name, oldName = null, { fromEditor = false } = {}) {
  if (oldName && oldName !== name && oldName in state.activations) {
    state.activations[name] = state.activations[oldName];
    delete state.activations[oldName];
  }
  state.selected = name;
  try { if (!fromEditor) slice?.setActiveTendon(name); } catch (e) { console.error(e); }
  try { viewer?.setTendonHighlight(name); } catch (e) { console.error(e); }
  tendonPanel.render(state.config, state.selected, state.issues);
  $("props-card").hidden = !state.selected;
  updateSegmentPicker();
  renderActivationPanel();
  // jump the slice view to the tendon's first point, so the student sees it
  const t = state.config.tendons.find((x) => x.name === name);
  if (t && t.path.length && !fromEditor && state.stand.slice) {
    const p = H.pointInfo(t.path[0], state.config);
    if (p && p.segment !== state.segment) setSegment(p.segment);
  }
}

function renderAll({ fromEditor = false, fromParams = false } = {}) {
  const cfg = state.config;
  const nameIn = $("config-name");
  if (document.activeElement !== nameIn) nameIn.value = cfg.name || "";
  $("btn-undo").disabled = !state.history.canUndo();
  $("btn-redo").disabled = !state.history.canRedo();
  tendonPanel.render(cfg, state.selected, state.issues);
  $("props-card").hidden = !state.selected;
  if (!fromParams) params.setParams(cfg.params || {});
  try {
    if (!fromEditor) { slice?.setConfig(cfg); slice?.setActiveTendon(state.selected); }
  } catch (e) { console.error(e); }
  updateSegmentPicker();
  renderSimControls();
  renderActivationPanel();
  renderIssueBanner();
  if (state.view === "experiments") experiments.refresh();
}

// ------------------------------------------------------------------ presets & files
let presetCache = null;
async function listPresets() {
  if (presetCache) return presetCache;
  try {
    const res = await fetch("configs/index.json", { cache: "no-store" });
    if (!res.ok) return [];
    let j = await res.json();
    if (!Array.isArray(j)) j = j.presets || j.configs || [];
    presetCache = j.map((p) => (typeof p === "string" ? { file: p, name: p.replace(/\.json$/, "") } : { file: p.file || p.path || p.url, name: p.name || p.title || p.file, description: p.description || "" }))
      .filter((p) => p.file);
    return presetCache;
  } catch { return []; }
}
async function fetchPreset(p) {
  const res = await fetch(`configs/${p.file}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Could not load the preset “${p.name}” (HTTP ${res.status}).`);
  return H.normalise(await res.json());
}
function parseConfigText(text) {
  let raw;
  try { raw = JSON.parse(text); } catch (e) { throw new Error(`This file is not valid JSON: ${e.message}`); }
  return H.normalise(raw);
}
async function openFile() {
  const f = await pickFile();
  if (!f) return;
  try { loadConfig(parseConfigText(f.text), `Opened ${f.name}`); } catch (e) { toast(e.message, "error", 7000); }
}
function loadConfig(cfg, message) {
  cfg = H.normalise(cfg);
  state.selected = cfg.tendons[0]?.name || null;
  state.activations = {};
  setConfig(cfg, { key: "load" });
  selectTendon(state.selected);
  if (message) toast(message, "success");
}
function saveFile() {
  downloadText(safeFileName(state.config.name, "json"), JSON.stringify(state.config, null, 2) + "\n", "application/json");
  toast("Downloaded the configuration file", "success");
}
function saveToLibrary() {
  const name = library.save(deepCopy(state.config));
  if (name) toast(`Saved “${name}” to your library`, "success");
  else toast("Could not save: browser storage is switched off or full. Use Download file instead.", "error", 6000);
}
function exportMjcf() {
  if (!state.sim?.xml || state.sim.isMock) {
    toast(state.sim?.isMock ? "Export needs the real physics model, which is not loaded." : "There is no compiled model yet.", "error");
    return;
  }
  downloadText(safeFileName(state.config.name, "xml"), state.sim.xml, "application/xml");
  const left = state.excluded.length ? ` ${state.excluded.length} unfinished tendon(s) are not included.` : "";
  toast(`Downloaded the MJCF model. It needs the mesh files from web/model/ next to it. For a self-contained folder, save the configuration and run build_mjcf.py on it.${left}`, "success", 8000);
}

// ------------------------------------------------------------------ simulation
/** Remove tendons with errors so the rest of the design still simulates. */
function simulatable(cfg) {
  const issues = H.validate(cfg);
  const bad = new Set(issues.filter((i) => i.level === "error" && i.tendon).map((i) => i.tendon));
  const config = deepCopy(cfg);
  config.tendons = config.tendons.filter((t) => !bad.has(t.name));
  return { config, excluded: [...bad] };
}

async function createSim(cfg) {
  const S = state.mods.Simulation;
  return S.create(state.mujoco, state.assets, cfg);
}

const scheduleRebuild = debounce(rebuild, 250);

async function rebuild() {
  if (!state.assets) return;
  const token = ++state.buildToken;
  const { config, excluded } = simulatable(state.config);
  state.excluded = excluded;
  setStatusBuilding(true);
  let sim;
  try {
    await new Promise((r) => setTimeout(r, 0));
    sim = await createSim(config);
  } catch (e) {
    if (token !== state.buildToken) return;
    console.error(e);
    state.buildError = e;
    setStatusBuilding(false);
    banner("build", "error", `The model could not be built. ${e.message}`, state.sim ? "The 3D view still shows the last version that worked." : "");
    setLoading(null);
    return;
  }
  if (token !== state.buildToken) { sim.dispose(); return; }
  state.buildError = null;
  banner("build", null);
  const old = state.sim;
  state.sim = sim;
  // keep activations across rebuilds
  const acts = {};
  for (const t of sim.tendons) if (t.actuatorId !== null) acts[t.name] = state.activations[t.name] ?? 0;
  try { sim.setActivations(acts); } catch (e) { console.error(e); }
  const orientation = config.params?.orientation || "upright";
  const refit = !old || orientation !== state.lastOrientation;
  state.lastOrientation = orientation;
  try { viewer?.setSimulation(sim, { refitCamera: refit }); viewer?.setTendonHighlight(state.selected); } catch (e) { console.error(e); banner("viewer", "error", `The 3D view failed: ${e.message}`); }
  analysis.setSimulation(sim);
  if (old) { try { old.dispose(); } catch (e) { console.error(e); } }
  setStatusBuilding(false);
  setLoading(null);
  renderActivationPanel();
  renderIssueBanner();
  state.lastMeasure = null;
}

function setStatusBuilding(on) {
  $("viewport").classList.toggle("building", on);
}

// ------------------------------------------------------------------ banners
function banner(id, kind, text, detail = "") {
  const host = $("banners");
  let el = host.querySelector(`[data-banner="${id}"]`);
  if (!kind) { el?.remove(); return; }
  if (!el) { el = h("div", { class: "banner", dataset: { banner: id }, role: kind === "error" ? "alert" : "status" }); host.append(el); }
  el.className = `banner banner-${kind}`;
  clear(el);
  el.append(h("div", { class: "banner-text" }, h("strong", {}, text), detail ? h("span", {}, ` ${detail}`) : null),
    h("button", { class: "icon-btn", "aria-label": "Dismiss", onClick: () => el.remove() }, icon("close", 14)));
}
function renderIssueBanner() {
  const ex = state.excluded.filter((n) => state.config.tendons.some((t) => t.name === n));
  const general = state.issues.filter((i) => i.level === "error" && !i.tendon);
  if (!ex.length && !general.length) { banner("issues", null); return; }
  const msgs = [];
  if (ex.length) msgs.push(`${ex.length === 1 ? `Tendon “${ex[0]}” is` : `${ex.length} tendons are`} not simulated until ${ex.length === 1 ? "it is" : "they are"} finished.`);
  for (const g of general.slice(0, 2)) msgs.push(g.message);
  const first = state.issues.find((i) => i.level === "error" && i.tendon && ex.includes(i.tendon));
  banner("issues", "warning", msgs.join(" "), first ? first.message : "");
}

// ------------------------------------------------------------------ sim controls (right panel)
let ctl = {};
function buildSimControls() {
  const el = $("sim-controls");
  const playBtn = h("button", { class: "btn btn-primary play-btn", onClick: () => setPlaying(!state.playing) });
  const resetBtn = h("button", { class: "btn btn-secondary", tip: "Put the tail back to straight. Activations stay as they are.", onClick: () => { state.sim?.reset(); analysis.resetHistory(); } }, icon("reset", 15), "Reset");
  const speeds = [0.25, 0.5, 1, 2, 4];
  const speedSel = h("select", { class: "select", "aria-label": "Simulation speed" }, speeds.map((s) => h("option", { value: s, selected: s === state.speed }, `${s}×`)));
  speedSel.addEventListener("change", () => { state.speed = Number(speedSel.value); });
  const orient = segmented([{ value: "upright", label: "Upright", icon: "upright" }, { value: "hanging", label: "Hanging", icon: "hanging" }], "upright",
    (v) => editConfig((c) => { c.params = { ...(c.params || {}), orientation: v }; }, "orientation"), { label: "Orientation", small: true });
  const grav = toggle("Gravity", false, (on) => editConfig((c) => { c.params = { ...(c.params || {}), gravity: on }; }, "gravity"),
    "Switch the weight of the tail parts on or off. Without gravity only the tendons and springs act.");
  const clock = h("div", { class: "sim-clock", tip: "Simulated time since the last reset, and how fast it runs compared to real time" });
  el.append(h("header", { class: "card-head" }, h("h2", { class: "card-title" }, "Simulation"), clock),
    h("div", { class: "ctl-row" }, playBtn, resetBtn, h("label", { class: "speed", tip: "Playback speed: simulated seconds per real second" }, speedSel)),
    h("div", { class: "ctl-row" }, orient, grav));
  ctl = { playBtn, orient, grav, clock, speedSel };
  renderSimControls();
}
function renderSimControls() {
  if (!ctl.playBtn) return;
  clear(ctl.playBtn).append(icon(state.playing ? "pause" : "play", 15), state.playing ? "Pause" : "Play");
  ctl.playBtn.setAttribute("aria-pressed", String(state.playing));
  ctl.orient.setValue(state.config?.params?.orientation || "upright");
  ctl.grav.setChecked(!!state.config?.params?.gravity);
}
function setPlaying(p) { state.playing = p; renderSimControls(); }

// ------------------------------------------------------------------ activation sliders
function renderActivationPanel() {
  const el = $("activation-panel");
  if (!el) return;
  const focusId = document.activeElement?.dataset?.act;
  clear(el);
  const cfg = state.config;
  const simNames = new Set((state.sim?.tendons || []).filter((t) => t.actuatorId !== null).map((t) => t.name));
  const driven = cfg.tendons.filter((t) => isActuated(t));
  el.append(h("header", { class: "card-head" }, h("h2", { class: "card-title" }, "Activation"),
    driven.length ? h("button", { class: "btn btn-ghost btn-small", tip: "Set every slider to 0 %", onClick: () => { for (const t of driven) setActivation(t.name, 0); renderActivationPanel(); } }, "Relax all") : null));
  if (!driven.length) {
    el.append(h("div", { class: "empty-state" }, h("p", {}, "No driven tendons yet. Give a tendon a motor or position actuator to get a slider here.")));
    return;
  }
  const groups = new Map();
  for (const t of driven) { const g = t.group || "(no group)"; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(t); }
  for (const [g, ts] of groups) {
    const grp = h("div", { class: "act-group" });
    if (ts.length > 1) {
      const mean = ts.reduce((a, t) => a + (state.activations[t.name] ?? 0), 0) / ts.length;
      grp.append(sliderRow({ label: g, sub: `group of ${ts.length}`, value: mean, master: true, colors: ts.map((t) => t.color), key: `group:${g}`,
        onInput: (v) => { for (const t of ts) setActivation(t.name, v); for (const r of grp.querySelectorAll(".act-row:not(.master)")) r.setValue(v); } }));
    }
    let host = grp;
    if (ts.length > 3) {
      const open = state.openGroups?.has(g) || ts.some((t) => t.name === state.selected);
      const det = h("details", { class: "act-members", open });
      det.addEventListener("toggle", () => { state.openGroups ||= new Set(); if (det.open) state.openGroups.add(g); else state.openGroups.delete(g); });
      det.append(h("summary", {}, `${ts.length} tendons`));
      grp.append(det);
      host = det;
    }
    for (const t of ts) {
      const live = simNames.has(t.name);
      host.append(sliderRow({ label: t.name, sub: ts.length > 1 ? "" : (t.group ? `group ${t.group}` : ""), value: state.activations[t.name] ?? 0, colors: [t.color], key: `t:${t.name}`, disabled: !live,
        note: live ? "" : "not simulated yet", selected: t.name === state.selected,
        onInput: (v) => { setActivation(t.name, v); const m = grp.querySelector(".act-row.master"); if (m) m.setValue(ts.reduce((a, x) => a + (state.activations[x.name] ?? 0), 0) / ts.length); } }));
    }
    el.append(grp);
  }
  if (focusId) el.querySelector(`[data-act="${CSS.escape(focusId)}"]`)?.focus();
}
function sliderRow({ label, sub, value, master = false, colors, key, onInput, disabled = false, note = "", selected = false }) {
  const inp = h("input", { type: "range", min: 0, max: 100, step: 1, value: Math.round(value * 100), disabled, dataset: { act: key }, "aria-label": `${label} activation` });
  const out = h("output", { class: "num-out" }, `${Math.round(value * 100)} %`);
  const paint = (v) => { inp.style.setProperty("--fill", `${v}%`); };
  paint(Math.round(value * 100));
  inp.style.setProperty("--thumb", colors[0]);
  inp.addEventListener("input", () => { out.textContent = `${inp.value} %`; paint(inp.value); onInput(Number(inp.value) / 100); });
  const row = h("div", { class: `act-row${master ? " master" : ""}${selected ? " selected" : ""}${disabled ? " disabled" : ""}` },
    h("div", { class: "act-label" },
      h("span", { class: "swatches" }, colors.slice(0, 4).map((c) => h("span", { class: "swatch small", style: { background: c } }))),
      h("span", { class: "act-name" }, label), sub ? h("span", { class: "act-sub" }, sub) : null, note ? h("span", { class: "act-sub warn" }, note) : null, out),
    inp);
  row.setValue = (v) => { inp.value = Math.round(v * 100); out.textContent = `${inp.value} %`; paint(inp.value); };
  return row;
}
function setActivation(name, a) {
  state.activations[name] = a;
  try { if (state.sim?.tendons.some((t) => t.name === name && t.actuatorId !== null)) state.sim.setActivation(name, a); } catch (e) { console.error(e); }
}

// ------------------------------------------------------------------ readout (right panel)
let readoutEls = null;
function renderReadout(m) {
  const el = $("readout");
  if (!m) return;
  const key = (state.sim?.tendons || []).map((t) => `${t.name}:${t.color}`).join("|");
  if (!readoutEls || readoutEls.key !== key) {
    clear(el);
    const hero = h("div", { class: "hero" },
      h("div", { class: "hero-label" }, "Tip bend towards ventral"),
      h("div", { class: "hero-value" }, h("span", { class: "hero-num" }, "0.0"), h("span", { class: "hero-unit" }, "°")));
    const stats = h("div", { class: "stats" });
    const mk = (label, tip) => { const v = h("strong", {}); stats.append(h("div", { class: "stat", tip }, h("span", { class: "stat-label" }, label), v)); return v; };
    const sLat = mk("Lateral", "Sideways bend of the tip; positive = towards dextral");
    const sTw = mk("Twist", "Rotation of the tip around the tail axis");
    const sDist = mk("Tip moved", "Straight-line distance the tip moved from its rest position");
    const tbl = h("table", { class: "readout-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Tendon"), h("th", { tip: "Pulling force" }, "N"), h("th", { tip: "Excursion: how much shorter the tendon became" }, "mm"), h("th", { tip: "Strain: excursion ÷ rest length" }, "%"), h("th", { tip: "Work delivered since the last reset" }, "mJ"))));
    const tb = h("tbody");
    tbl.append(tb);
    const rows = (state.sim?.tendons || []).map((t) => {
      const cells = [h("td", { class: "num" }), h("td", { class: "num" }), h("td", { class: "num" }), h("td", { class: "num" })];
      const bar = h("span", { class: "force-bar" });
      tb.append(h("tr", {}, h("td", { class: "tname" }, h("span", { class: "swatch small", style: { background: t.color } }), h("span", {}, t.name)), ...cells));
      return { cells, bar };
    });
    // joint torque table
    const tq = h("table", { class: "readout-table torque-table" },
      h("thead", {}, h("tr", {}, h("th", { tip: "Joint between this segment and the one before it" }, "Joint"), h("th", { tip: "Pitch torque: ventral / dorsal bending" }, "Pitch"), h("th", { tip: "Roll torque: sideways bending" }, "Roll"), h("th", { tip: "Yaw torque: twist" }, "Yaw"))));
    const tqb = h("tbody");
    tq.append(tqb);
    const n = (m.segments || []).length;
    const tqRows = m.segments.map((s) => {
      const cells = [h("td", { class: "num" }), h("td", { class: "num" }), h("td", { class: "num" })];
      tqb.append(h("tr", {}, h("td", {}, `seg ${s.index}`), ...cells));
      return cells;
    });
    el.append(...[h("header", { class: "card-head" }, h("h2", { class: "card-title" }, "Measurements"), h("span", { class: "card-sub" }, "live")),
      hero, stats,
      rows.length ? h("h3", { class: "sub-title" }, "Tendons") : null, rows.length ? tbl : null,
      h("details", { class: "fold", open: storage.get("torque-open", true) },
        h("summary", { tip: "Torque that all tendons together apply on each vertebral joint, in N·mm" }, "Joint torque from tendons", h("span", { class: "unit" }, "N·mm")), tq)].filter(Boolean));
    el.querySelector("details.fold:last-child")?.addEventListener("toggle", (e) => storage.set("torque-open", e.target.open));
    readoutEls = { key, hero: hero.querySelector(".hero-num"), sLat, sTw, sDist, rows, tqRows, n };
  }
  const R = readoutEls;
  R.hero.textContent = fmt(m.tip.ventral, 1);
  R.sLat.textContent = `${fmt(m.tip.lateral, 1)}°`;
  R.sTw.textContent = `${fmt(m.tip.twist, 1)}°`;
  R.sDist.textContent = `${fmt(m.tip.distance * MM, 1)} mm`;
  m.tendons.forEach((t, i) => {
    const r = R.rows[i];
    if (!r) return;
    r.cells[0].textContent = fmt(t.force, 2);
    r.cells[1].textContent = fmt(t.excursion * MM, 2);
    r.cells[2].textContent = fmt(t.strain * 100, 1);
    r.cells[3].textContent = fmt(t.work * 1000, 2);
  });
  let maxAbs = 0;
  for (const s of m.segments) if (s.torque) for (const a of ["pitch", "roll", "yaw"]) maxAbs = Math.max(maxAbs, Math.abs(s.torque[a]));
  m.segments.forEach((s, i) => {
    const cells = R.tqRows[i];
    if (!cells || !s.torque) return;
    ["pitch", "roll", "yaw"].forEach((a, k) => {
      const v = s.torque[a] * NMM;
      cells[k].textContent = fmt(v, 2);
      cells[k].classList.toggle("strong", maxAbs > 0 && Math.abs(s.torque[a]) >= 0.5 * maxAbs);
    });
  });
}

function updateClock(m) {
  if (!ctl.clock || !m) return;
  const rate = state.simRate;
  ctl.clock.textContent = `t = ${m.time.toFixed(2)} s${state.playing && rate < 0.9 * state.speed ? ` · ${(rate).toFixed(2)}× real time` : ""}`;
  ctl.clock.classList.toggle("slow", state.playing && rate < 0.9 * state.speed);
}

// ------------------------------------------------------------------ main loop
let lastFrame = performance.now(), lastMeasure = 0, lastChart = 0;
let rateAcc = { sim: 0, wall: 0 };
function frame(now) {
  const dtReal = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const sim = state.sim;
  if (sim && state.view === "design") {
    if (state.playing) {
      const t0 = performance.now();
      const simT0 = sim.time;
      try {
        // never spend more than ~35 ms per frame on physics; fall behind real time instead
        const want = Math.min(dtReal, 0.05) * state.speed;
        const chunk = Math.max(sim.timestep || 0.002, want / 4);
        let done = 0;
        while (done < want - 1e-9 && performance.now() - t0 < 35) { const d = Math.min(chunk, want - done); sim.advance(d); done += d; }
      } catch (e) {
        console.error(e);
        setPlaying(false);
        banner("step", "error", `The simulation stopped: ${e.message}`, "Try a smaller time step (Body parameters) or press Reset.");
      }
      rateAcc.sim += sim.time - simT0; rateAcc.wall += dtReal;
      if (rateAcc.wall > 0.5) { state.simRate = rateAcc.sim / rateAcc.wall; rateAcc = { sim: 0, wall: 0 }; }
      if (!Number.isFinite(sim.time) || (sim.data?.qpos && !Number.isFinite(sim.data.qpos[0]))) {
        setPlaying(false);
        banner("step", "error", "The simulation became unstable (numbers grew too large).", "Make the time step smaller in Body parameters, then press Reset.");
      }
    }
    try { viewer?.update(); } catch (e) { console.error(e); }
    if (now - lastMeasure > 66) {
      lastMeasure = now;
      try {
        const m = sim.measure();
        state.lastMeasure = m;
        analysis.push(m);
        renderReadout(m);
        updateClock(m);
        if (now - lastChart > 120) { lastChart = now; analysis.render(m); }
      } catch (e) { console.error(e); }
    }
  }
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ views
function setView(v) {
  state.view = v;
  viewSwitch.setValue(v);
  $("view-design").hidden = v !== "design";
  $("view-experiments").hidden = v !== "experiments";
  if (v === "experiments") experiments.refresh(); else experiments.onHide();
  if (v === "design") { lastFrame = performance.now(); requestAnimationFrame(() => viewer?.resize?.()); }
}

boot().catch((e) => {
  console.error(e);
  setLoading(`Something went wrong while starting: ${e.message}`, true);
});

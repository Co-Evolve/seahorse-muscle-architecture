// Persistence (localStorage, always optional) and undo/redo history for config edits.

const PREFIX = "tendon-designer.";

export const storage = {
  get(key, fallback = null) {
    try {
      const raw = window.localStorage.getItem(PREFIX + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch { return fallback; }
  },
  set(key, value) {
    try { window.localStorage.setItem(PREFIX + key, JSON.stringify(value)); return true; } catch { return false; }
  },
  remove(key) { try { window.localStorage.removeItem(PREFIX + key); } catch { /* storage unavailable */ } },
};

/** The student's own library of saved configurations: {name: {config, saved}}. */
export const library = {
  list() {
    const lib = storage.get("library", {}) || {};
    return Object.entries(lib).map(([name, v]) => ({ name, saved: v.saved, config: v.config }))
      .sort((a, b) => (b.saved || 0) - (a.saved || 0));
  },
  save(config) {
    const lib = storage.get("library", {}) || {};
    const name = (config.name || "Untitled").trim() || "Untitled";
    lib[name] = { config, saved: Date.now() };
    return storage.set("library", lib) ? name : null;
  },
  remove(name) {
    const lib = storage.get("library", {}) || {};
    delete lib[name];
    storage.set("library", lib);
  },
  get(name) { return (storage.get("library", {}) || {})[name]?.config || null; },
};

/**
 * Snapshot history. push() records the state *after* an edit; edits with the same
 * coalesce key within `windowMs` replace the last entry (so a slider drag is one undo step).
 */
export class History {
  constructor(limit = 200, windowMs = 700) {
    this.limit = limit; this.windowMs = windowMs;
    this.stack = []; this.index = -1;
    this.lastKey = null; this.lastTime = 0;
  }
  reset(state) { this.stack = [JSON.stringify(state)]; this.index = 0; this.lastKey = null; }
  push(state, key = null, windowMs = this.windowMs) {
    const snap = JSON.stringify(state);
    if (this.stack[this.index] === snap) return;
    const now = performance.now();
    this.stack.length = this.index + 1;
    if (key && key === this.lastKey && now - this.lastTime < windowMs && this.index > 0) {
      this.stack[this.index] = snap;
    } else {
      this.stack.push(snap);
      if (this.stack.length > this.limit) this.stack.shift();
      this.index = this.stack.length - 1;
    }
    this.lastKey = key; this.lastTime = now;
  }
  canUndo() { return this.index > 0; }
  canRedo() { return this.index < this.stack.length - 1; }
  undo() { if (!this.canUndo()) return null; this.index--; this.lastKey = null; return JSON.parse(this.stack[this.index]); }
  redo() { if (!this.canRedo()) return null; this.index++; this.lastKey = null; return JSON.parse(this.stack[this.index]); }
}

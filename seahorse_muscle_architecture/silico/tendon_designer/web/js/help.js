// Help overlay: the workflow in plain English for a biology student. Opens on first visit.

import { h, s, icon, clear } from "./ui.js";

function compass() {
  // cross-section of one segment, looking from the base towards the tip
  const label = (x, y, t, anchor = "middle") => s("text", { x, y, "text-anchor": anchor, class: "cmp-label" }, t);
  return s("svg", { class: "help-compass", viewBox: "-150 -92 300 184", role: "img", "aria-label": "Directions in a segment cross-section" },
    s("rect", { x: -54, y: -54, width: 108, height: 108, rx: 18, class: "cmp-body" }),
    ...[[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => s("rect", { x: sx > 0 ? 8 : -50, y: sy > 0 ? 8 : -50, width: 42, height: 42, rx: 8, class: "cmp-plate" })),
    s("circle", { r: 13, class: "cmp-vert" }),
    s("path", { d: "M0 -62V-80M0 62V80M-62 0H-80M62 0H80", class: "cmp-axis" }),
    label(0, -84, "dorsal (back)"), label(0, 91, "ventral (belly)"),
    label(-86, 4, "sinistral", "end"), label(-86, 18, "(left)", "end"),
    label(86, 4, "dextral", "start"), label(86, 18, "(right)", "start"));
}

const SECTIONS = [
  {
    id: "start", title: "What you are looking at",
    body: () => [
      h("p", {}, "This is a robot model of a seahorse tail. It is a chain of 11 ", h("b", {}, "segments"), ", numbered 0 at the base to 10 at the tip. Segment 0 is fixed in place."),
      h("p", {}, "Each segment has a central ", h("b", {}, "vertebra"), " and four bony ", h("b", {}, "plates"), " at its corners. The plates can slide a little over each other. Between two vertebrae there is a joint that can bend in three ways:"),
      h("ul", {},
        h("li", {}, h("b", {}, "Pitch"), ": bending towards ventral or dorsal (curling the tail forward or backward)."),
        h("li", {}, h("b", {}, "Roll"), ": bending sideways, towards sinistral or dextral."),
        h("li", {}, h("b", {}, "Yaw"), ": twisting around the long axis of the tail.")),
      h("p", {}, "You design ", h("b", {}, "tendons"), " (artificial muscles): strings that run through small holes in the plates. When a tendon pulls, it becomes shorter and the tail bends."),
    ],
  },
  {
    id: "dirs", title: "Directions",
    body: () => [
      h("div", { class: "help-split" }, compass(), h("div", {},
        h("p", {}, "The slice view shows one segment ", h("b", {}, "as seen from the base, looking towards the tip"), "."),
        h("ul", {},
          h("li", {}, h("b", {}, "Ventral"), " = belly side. ", h("b", {}, "Dorsal"), " = back side."),
          h("li", {}, h("b", {}, "Sinistral"), " = left. ", h("b", {}, "Dextral"), " = right."),
          h("li", {}, "In the numbers: a positive ventral angle bends towards the belly; a positive lateral angle bends towards dextral."))))],
  },
  {
    id: "tendon", title: "Make a tendon",
    body: () => [
      h("ol", { class: "steps" },
        h("li", {}, "Press ", h("b", {}, "Add tendon"), " in the left panel. A new, empty tendon is selected."),
        h("li", {}, "In the slice view, pick the segment where the tendon starts with the strip of small segment pictures (base on the left, tip on the right), or the ← → keys."),
        h("li", {}, "Click a hole (", h("b", {}, "tap"), ") in a plate. This is the ", h("b", {}, "start"), " point. The view then moves on to the next segment by itself."),
        h("li", {}, "Click the hole the tendon runs through in each next segment. These are ", h("b", {}, "via"), " points."),
        h("li", {}, "The last hole you click is the ", h("b", {}, "end"), " point, where the tendon is anchored."),
        h("li", {}, "As soon as the tendon has a start and an end, it appears in the 3D view and gets a slider on the right.")),
      h("p", {}, "Hover a hole to see its name. Click a numbered point to move it earlier or later, or to remove it. ", h("b", {}, "Shift"), " + click puts a free point anywhere on a plate. A tendon should pass a hole in every segment it crosses; the app warns you if it skips one."),
      h("p", {}, h("b", {}, "Split a tendon into two ends"), " (like a muscle with two distal tendons): first make the main path up to its first end. Then click the point where the split should start and choose ", h("b", {}, "Split here"), ". Click the holes of the second end. One actuator pulls both ends; a pulley shares the force between them."),
      h("p", {}, "The ", icon("mirror", 14), " button makes a mirror copy of a tendon on the other side (sinistral ↔ dextral)."),
    ],
  },
  {
    id: "sim", title: "Pull the tendons",
    body: () => [
      h("p", {}, "Use the ", h("b", {}, "activation sliders"), " on the right. 0 % = relaxed, 100 % = full pull. The ", h("b", {}, "group slider"), " moves all tendons of one group together."),
      h("ul", {},
        h("li", {}, h("b", {}, "Motor"), " tendons pull with a force: 100 % = their maximum force."),
        h("li", {}, h("b", {}, "Position"), " tendons try to become a set percentage shorter (like a muscle that contracts to a length)."),
        h("li", {}, h("b", {}, "Passive"), " tendons are not driven; with a stiffness they act like elastic bands.")),
      h("p", {}, "When you release a slider, the tail does not always spring back: the joint springs of the model are very weak. Make them stiffer in ", h("b", {}, "Body parameters"), " if you want the tail to return to straight."),
      h("p", {}, h("b", {}, "Upright / hanging"), " turns the whole tail. ", h("b", {}, "Gravity"), " switches the weight of the parts on or off. ", h("b", {}, "Reset"), " puts the tail back to straight; the sliders keep their values."),
    ],
  },
  {
    id: "plots", title: "What the numbers and plots mean",
    body: () => [
      h("ul", {},
        h("li", {}, h("b", {}, "Tip angle"), ": how far the tip segment points away from the base direction. Ventral and lateral (sideways) are shown separately."),
        h("li", {}, h("b", {}, "Curvature profile"), ": the bend of each joint on its own (segment compared to the segment before it). It shows ", h("i", {}, "where"), " the tail bends."),
        h("li", {}, h("b", {}, "Tail shape"), ": the tail seen from the side and from the front, in millimetres. The grey line is the straight rest pose."),
        h("li", {}, h("b", {}, "Excursion"), ": how much shorter a tendon became (mm). ", h("b", {}, "Strain"), " = excursion divided by the rest length."),
        h("li", {}, h("b", {}, "Work"), ": energy the tendon delivered while shortening (millijoule)."),
        h("li", {}, h("b", {}, "Joint torque"), " (N·mm): the turning effect of the tendons on each joint. The Torques view shows how much each tendon contributes; positive and negative shares stack up and down.")),
    ],
  },
  {
    id: "arms", title: "Moment arms",
    body: () => [
      h("p", {}, "A ", h("b", {}, "moment arm"), " is the lever arm of a tendon around a joint: the distance between the tendon's line of pull and the joint's axis. Torque = tendon force × moment arm. A tendon far from the joint axis has a long moment arm: it makes more torque per newton, but it has to shorten more for the same angle."),
      h("p", {}, "This app measures the moment arm as ", h("b", {}, "how many millimetres the tendon length changes when the joint turns by one radian"), " (57°). That is the same quantity, and it also works for tendons that run along a curved path. The sign tells the direction: a tendon that bends a joint towards ventral when it pulls has the opposite sign of one that bends it dorsally."),
      h("p", {}, "In the Torques view the table shows the moment arm of every tendon at every joint, for the chosen axis (pitch, roll or yaw). Empty cells mean the tendon does not cross that joint."),
    ],
  },
  {
    id: "body", title: "Body parameters",
    body: () => [
      h("p", {}, "The ", h("b", {}, "Body"), " tab in the left panel changes the material of the tail. ", h("b", {}, "Stiffness"), " is how strongly a part springs back (like a stronger spring). ", h("b", {}, "Damping"), " is how much it resists fast movement (like moving through honey): it slows the motion but does not change the final shape. Hover any label for an explanation and its unit."),
    ],
  },
  {
    id: "exp", title: "Experiments and comparisons",
    body: () => [
      h("p", {}, "Open ", h("b", {}, "Experiments"), " in the top bar. Choose a pattern for each tendon group (hold, step, ramp or sine), how strong (amplitude) and how long. Press ", h("b", {}, "Run"), ": the simulation runs as fast as possible without drawing."),
      h("p", {}, "With ", h("b", {}, "Compare A vs B"), ", the same protocol runs on your current design (A) and on a second one (B): a preset, a file, or a design you saved in your library. Tendons are driven by ", h("b", {}, "group name"), ", so give matching tendons in A and B the same group."),
      h("p", {}, h("b", {}, "Download CSV"), " gives every sample of every run as a table (open it in Excel, R or Python): time, activations, tip angles, the angle and torque of every joint, and force, length, excursion, torque and moment arm of every tendon."),
    ],
  },
  {
    id: "save", title: "Saving your work",
    body: () => [
      h("p", {}, "The app remembers your current design in this browser automatically. To keep a design for good, use ", h("b", {}, "Save → Download file"), " (a small .json file). ", h("b", {}, "Save to my library"), " keeps it in this browser so you can pick it as B in a comparison."),
      h("p", {}, h("b", {}, "Export MJCF"), " downloads the MuJoCo model file of the current design (the 3D mesh files are in the ", h("code", {}, "web/model/"), " folder)."),
      h("p", { class: "shortcuts" }, h("kbd", {}, "Space"), " play / pause · ", h("kbd", {}, "Ctrl"), "+", h("kbd", {}, "Z"), " undo · ", h("kbd", {}, "Ctrl"), "+", h("kbd", {}, "Shift"), "+", h("kbd", {}, "Z"), " redo · ", h("kbd", {}, "?"), " this help"),
      h("p", {}, h("a", { href: "README.md", target: "_blank", rel: "noopener" }, "Open the full student guide"), "."),
    ],
  },
];

export class HelpOverlay {
  constructor() {
    this.el = h("div", { class: "help-overlay", hidden: true, role: "dialog", "aria-modal": "true", "aria-labelledby": "help-title" });
    this.nav = h("nav", { class: "help-nav", "aria-label": "Help topics" });
    this.content = h("div", { class: "help-content" });
    const close = h("button", { class: "icon-btn help-close", "aria-label": "Close help", onClick: () => this.close() }, icon("close", 18));
    this.el.append(h("div", { class: "help-panel" },
      h("header", { class: "help-head" }, h("div", {}, h("h1", { id: "help-title", class: "help-title" }, "How to use the Tendon Designer"),
        h("p", { class: "help-lede" }, "Design artificial tendons for a seahorse tail, pull them, and measure how the tail bends.")), close),
      h("div", { class: "help-body" }, this.nav, this.content)));
    this.el.addEventListener("pointerdown", (e) => { if (e.target === this.el) this.close(); });
    this.el.addEventListener("keydown", (e) => { if (e.key === "Escape") this.close(); });
    document.body.append(this.el);
    this._render();
  }
  _render() {
    clear(this.nav); clear(this.content);
    for (const sec of SECTIONS) {
      const id = `help-${sec.id}`;
      this.nav.append(h("a", { href: `#${id}`, onClick: (e) => { e.preventDefault(); this.content.querySelector(`#${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }); } }, sec.title));
      this.content.append(h("section", { id, class: "help-section" }, h("h2", {}, sec.title), ...sec.body()));
    }
  }
  open(section) {
    this._prev = document.activeElement;
    this.el.hidden = false;
    requestAnimationFrame(() => this.el.classList.add("show"));
    this.el.querySelector(".help-close").focus();
    if (section) this.content.querySelector(`#help-${section}`)?.scrollIntoView({ block: "start" });
  }
  close() {
    this.el.classList.remove("show");
    setTimeout(() => { this.el.hidden = true; }, 180);
    this._prev?.focus?.();
  }
  get isOpen() { return !this.el.hidden; }
}

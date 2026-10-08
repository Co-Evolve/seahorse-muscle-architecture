# Vendored third-party code

Exact versions, copied from the npm tarballs (`npm pack <pkg>@<version>`). No CDN at runtime.

| Package | Version | Published | Source | Files |
|---|---|---|---|---|
| `@mujoco/mujoco` (official MuJoCo WASM bindings, Apache-2.0) | 3.14.0 | 2026-09-22 | https://www.npmjs.com/package/@mujoco/mujoco, https://github.com/google-deepmind/mujoco/tree/main/wasm | `mujoco/mujoco.js`, `mujoco/mujoco.wasm` (single-threaded build), `mujoco/mujoco.d.ts` (API reference), `mujoco/README.md` |
| `three` (MIT) | 0.186.0 | 2026-09-08 | https://www.npmjs.com/package/three, https://github.com/mrdoob/three.js | `three/three.module.js`, `three/three.core.js`, `three/LICENSE`, `three/addons/OrbitControls.js` |

Local modification: `three/addons/OrbitControls.js` (from `examples/jsm/controls/OrbitControls.js`)
imports from `'../three.module.js'` instead of the bare specifier `'three'`, so no import map is
needed. Nothing else is changed.

Not vendored: the multi-threaded MuJoCo build (`mt/`, needs COOP/COEP headers) and source maps.

To update: `npm pack @mujoco/mujoco@X three@Y`, unpack, copy the files above, re-apply the
OrbitControls import edit, run `tests/node` (`npm test`) and open `dev_sim.html`.

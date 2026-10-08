"""Parity check: compare a model XML produced by the JS builder with the Python-built model.

Given a configuration JSON and the MJCF string that ``buildModelXml(baseXml, catalog, config)``
(web/js/model_builder.js) produced for it, compile both and compare everything the config
controls: sites (incl. free points), tendon paths (sites/pulleys), tendon length0 and
parameters, actuators (order and parameters), sensors, joint parameters, body poses and options.

By default the Python model is built on ``web/model/base.xml`` (the same base the browser uses),
so only builder differences show up. ``--python-base morphology`` builds it with
``export_assets.build_base_mjcf`` instead (also catches a stale ``web/model`` export).

Usage:
    python -m seahorse_muscle_architecture.silico.tendon_designer.tests.parity_check \
        config.json js_model.xml [--python-base web|morphology]

Exit code 0 = identical (within tolerance), 1 = differences (listed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np
from dm_control import mjcf

from seahorse_muscle_architecture.silico.tendon_designer import config as td

RTOL = 1e-9
ATOL = 1e-12


def load_web_assets(
        model_dir: Path = td.WEB_MODEL_DIR
        ) -> Dict[str, bytes]:
    catalog = json.loads((model_dir / "catalog.json").read_text())
    return {name: (model_dir / name).read_bytes() for name in catalog["mesh_files"]}


def compile_js_xml(
        xml: str,
        model_dir: Path = td.WEB_MODEL_DIR
        ) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml, load_web_assets(model_dir))


def load_web_base_mjcf(
        model_dir: Path = td.WEB_MODEL_DIR
        ) -> mjcf.RootElement:
    """``web/model/base.xml`` as a dm_control model.

    dm_control writes the root default class as ``class="/"`` and anonymous elements as
    ``name="//unnamed_..."`` but cannot parse either back, so those are stripped first.
    """
    xml = (model_dir / "base.xml").read_text()
    xml = re.sub(r'\s+class="/"', "", xml)
    xml = re.sub(r"<default>\s*<default\s*/>\s*</default>", "", xml)
    xml = re.sub(r'\s+name="//unnamed_[^"]*"', "", xml)
    return mjcf.from_xml_string(xml, model_dir=str(model_dir))


def build_python_model(
        config: Dict,
        python_base: str = "web",
        model_dir: Path = td.WEB_MODEL_DIR
        ) -> mujoco.MjModel:
    catalog = json.loads((model_dir / "catalog.json").read_text())
    if python_base == "web":
        mjcf_model = load_web_base_mjcf(model_dir)
        td.apply_tendon_config(mjcf_model, config, catalog)
    else:
        mjcf_model = td.build_model(config, num_segments=catalog["num_segments"], catalog=catalog)
    return td.compile_model(mjcf_model)


def _name(
        model: mujoco.MjModel,
        obj: mujoco.mjtObj,
        index: int
        ) -> Optional[str]:
    return mujoco.mj_id2name(model, obj, int(index)) if index >= 0 else None


def _names(
        model: mujoco.MjModel,
        obj: mujoco.mjtObj,
        count: int
        ) -> List[str]:
    return [_name(model, obj, i) or f"#{i}" for i in range(count)]


def tendon_path(
        model: mujoco.MjModel,
        tendon: int
        ) -> List[str]:
    """Readable wrap list: 'site:<name>' / 'pulley:<divisor>' / 'geom:<name>'."""
    out = []
    for k in range(model.tendon_adr[tendon], model.tendon_adr[tendon] + model.tendon_num[tendon]):
        kind = model.wrap_type[k]
        if kind == mujoco.mjtWrap.mjWRAP_SITE:
            out.append("site:" + _name(model, mujoco.mjtObj.mjOBJ_SITE, model.wrap_objid[k]))
        elif kind == mujoco.mjtWrap.mjWRAP_PULLEY:
            out.append(f"pulley:{model.wrap_prm[k]:g}")
        else:
            out.append(f"wrap{int(kind)}:{model.wrap_objid[k]}")
    return out


def compare_models(
        py: mujoco.MjModel,
        js: mujoco.MjModel
        ) -> List[str]:
    """Differences between two compiled models (empty list = parity)."""
    diffs: List[str] = []

    def close(
            label: str,
            a,
            b
            ) -> None:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=RTOL, atol=ATOL):
            diffs.append(f"{label}: python={a.tolist()} js={b.tolist()}")

    def equal(
            label: str,
            a,
            b
            ) -> None:
        if a != b:
            diffs.append(f"{label}: python={a!r} js={b!r}")

    for attribute in ("nbody", "njnt", "nsite", "ntendon", "nwrap", "nu", "nsensor", "neq", "ngeom"):
        equal(attribute, getattr(py, attribute), getattr(js, attribute))
    close("opt.timestep", py.opt.timestep, js.opt.timestep)
    close("opt.gravity", py.opt.gravity, js.opt.gravity)
    equal("opt.disableflags", int(py.opt.disableflags), int(js.opt.disableflags))
    if diffs and any(d.startswith(("nbody", "njnt", "nsite", "ntendon", "nu", "nsensor")) for d in diffs):
        pass  # keep going: name-based comparisons below still give useful detail

    # Bodies
    js_bodies = {n: i for i, n in enumerate(_names(js, mujoco.mjtObj.mjOBJ_BODY, js.nbody))}
    for i, name in enumerate(_names(py, mujoco.mjtObj.mjOBJ_BODY, py.nbody)):
        j = js_bodies.get(name)
        if j is None:
            diffs.append(f"body {name}: missing in js")
            continue
        close(f"body {name} pos", py.body_pos[i], js.body_pos[j])
        close(f"body {name} quat", py.body_quat[i], js.body_quat[j])

    # Sites (free points included)
    js_sites = {n: i for i, n in enumerate(_names(js, mujoco.mjtObj.mjOBJ_SITE, js.nsite))}
    for i, name in enumerate(_names(py, mujoco.mjtObj.mjOBJ_SITE, py.nsite)):
        j = js_sites.get(name)
        if j is None:
            diffs.append(f"site {name}: missing in js")
            continue
        equal(f"site {name} body", _name(py, mujoco.mjtObj.mjOBJ_BODY, py.site_bodyid[i]),
              _name(js, mujoco.mjtObj.mjOBJ_BODY, js.site_bodyid[j]))
        close(f"site {name} pos", py.site_pos[i], js.site_pos[j])
    for name in set(js_sites) - set(_names(py, mujoco.mjtObj.mjOBJ_SITE, py.nsite)):
        diffs.append(f"site {name}: missing in python")

    # Joints
    js_joints = {n: i for i, n in enumerate(_names(js, mujoco.mjtObj.mjOBJ_JOINT, js.njnt))}
    for i, name in enumerate(_names(py, mujoco.mjtObj.mjOBJ_JOINT, py.njnt)):
        j = js_joints.get(name)
        if j is None:
            diffs.append(f"joint {name}: missing in js")
            continue
        close(f"joint {name} stiffness", py.jnt_stiffness[i], js.jnt_stiffness[j])
        close(f"joint {name} range", py.jnt_range[i], js.jnt_range[j])
        equal(f"joint {name} limited", int(py.jnt_limited[i]), int(js.jnt_limited[j]))
        close(f"joint {name} damping", py.dof_damping[py.jnt_dofadr[i]], js.dof_damping[js.jnt_dofadr[j]])
        close(f"joint {name} qpos0", py.qpos0[py.jnt_qposadr[i]], js.qpos0[js.jnt_qposadr[j]])

    # Tendons (struts + config tendons)
    py_tendons = _names(py, mujoco.mjtObj.mjOBJ_TENDON, py.ntendon)
    js_tendons = _names(js, mujoco.mjtObj.mjOBJ_TENDON, js.ntendon)
    equal("tendon order", py_tendons, js_tendons)
    js_index = {n: i for i, n in enumerate(js_tendons)}
    for i, name in enumerate(py_tendons):
        j = js_index.get(name)
        if j is None:
            diffs.append(f"tendon {name}: missing in js")
            continue
        equal(f"tendon {name} path", tendon_path(py, i), tendon_path(js, j))
        close(f"tendon {name} length0", py.tendon_length0[i], js.tendon_length0[j])
        close(f"tendon {name} lengthspring", py.tendon_lengthspring[i], js.tendon_lengthspring[j])
        close(f"tendon {name} stiffness", py.tendon_stiffness[i], js.tendon_stiffness[j])
        close(f"tendon {name} damping", py.tendon_damping[i], js.tendon_damping[j])
        close(f"tendon {name} width", py.tendon_width[i], js.tendon_width[j])
        close(f"tendon {name} rgba", py.tendon_rgba[i], js.tendon_rgba[j])
        equal(f"tendon {name} limited", int(py.tendon_limited[i]), int(js.tendon_limited[j]))

    # Actuators (order matters: it is the ctrl vector layout)
    py_actuators = _names(py, mujoco.mjtObj.mjOBJ_ACTUATOR, py.nu)
    js_actuators = _names(js, mujoco.mjtObj.mjOBJ_ACTUATOR, js.nu)
    equal("actuator order", py_actuators, js_actuators)
    js_index = {n: i for i, n in enumerate(js_actuators)}
    for i, name in enumerate(py_actuators):
        j = js_index.get(name)
        if j is None:
            diffs.append(f"actuator {name}: missing in js")
            continue
        equal(f"actuator {name} transmission", (int(py.actuator_trntype[i]),
                                                _name(py, mujoco.mjtObj.mjOBJ_TENDON, py.actuator_trnid[i, 0])),
              (int(js.actuator_trntype[j]), _name(js, mujoco.mjtObj.mjOBJ_TENDON, js.actuator_trnid[j, 0])))
        equal(f"actuator {name} types", (int(py.actuator_dyntype[i]), int(py.actuator_gaintype[i]),
                                         int(py.actuator_biastype[i])),
              (int(js.actuator_dyntype[j]), int(js.actuator_gaintype[j]), int(js.actuator_biastype[j])))
        for field in ("gear", "gainprm", "biasprm", "ctrlrange", "forcerange"):
            close(f"actuator {name} {field}", getattr(py, f"actuator_{field}")[i],
                  getattr(js, f"actuator_{field}")[j])
        equal(f"actuator {name} limits", (int(py.actuator_ctrllimited[i]), int(py.actuator_forcelimited[i])),
              (int(js.actuator_ctrllimited[j]), int(js.actuator_forcelimited[j])))

    # Sensors
    py_sensors = _names(py, mujoco.mjtObj.mjOBJ_SENSOR, py.nsensor)
    js_sensors = _names(js, mujoco.mjtObj.mjOBJ_SENSOR, js.nsensor)
    equal("sensor names", sorted(py_sensors), sorted(js_sensors))
    js_index = {n: i for i, n in enumerate(js_sensors)}
    for i, name in enumerate(py_sensors):
        j = js_index.get(name)
        if j is not None:
            equal(f"sensor {name} type/object", (int(py.sensor_type[i]), int(py.sensor_objtype[i]), int(py.sensor_objid[i])),
                  (int(js.sensor_type[j]), int(js.sensor_objtype[j]), int(js.sensor_objid[j])))
    return diffs


def check(
        config_path: Path,
        js_xml_path: Path,
        python_base: str = "web"
        ) -> List[str]:
    config = td.load_config(config_path)
    js = compile_js_xml(Path(js_xml_path).read_text())
    py = build_python_model(config, python_base=python_base)
    return compare_models(py, js)


def main(
        argv: Optional[list] = None
        ) -> int:
    parser = argparse.ArgumentParser(description="Compare a JS-built model XML with the Python-built model.")
    parser.add_argument("config", type=Path)
    parser.add_argument("js_xml", type=Path, help="XML string produced by buildModelXml for the same config")
    parser.add_argument("--python-base", choices=("web", "morphology"), default="web")
    args = parser.parse_args(argv)
    diffs = check(args.config, args.js_xml, args.python_base)
    if diffs:
        print(f"PARITY FAILED for {args.config.name}: {len(diffs)} difference(s)")
        for diff in diffs[:200]:
            print("  " + diff)
        return 1
    print(f"parity OK: {args.config.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

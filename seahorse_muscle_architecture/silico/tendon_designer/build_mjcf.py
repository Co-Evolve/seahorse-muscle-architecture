"""Turn a Seahorse muscle playground configuration JSON into a standalone MJCF folder.

The folder holds ``<name>.xml`` plus every mesh it references, and a copy of the config
(``<name>.config.json``). It loads in plain MuJoCo (``mujoco.MjModel.from_xml_path``,
``python -m mujoco.viewer --mjcf=out/<name>.xml``).

Usage:
    python -m seahorse_muscle_architecture.silico.tendon_designer.build_mjcf student_config.json \
        --out-dir out/ [--name X] [--num-segments 11]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import mujoco
from dm_control import mjcf

from seahorse_muscle_architecture.silico.tendon_designer import config as td


def build_mjcf_folder(
        config: Dict,
        out_dir: Path,
        name: str,
        num_segments: int = 11
        ) -> Path:
    """Write the MJCF folder and return the path of the XML file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mjcf_model = td.build_model(config, num_segments=num_segments)
    mjcf.export_with_assets(mjcf_model, str(out_dir), out_file_name=f"{name}.xml")
    (out_dir / f"{name}.config.json").write_text(json.dumps(config, indent=2))
    return out_dir / f"{name}.xml"


def summary(
        config: Dict,
        model: mujoco.MjModel
        ) -> str:
    lines = [f'Configuration "{config.get("name", "")}": {len(config.get("tendons") or [])} tendon(s), '
             f'{len(config.get("free_points") or [])} free point(s)',
             f"Model: {model.nbody} bodies, {model.njnt} joints, {model.ntendon} tendons, "
             f"{model.nu} actuators, {model.nsensor} sensors, timestep {model.opt.timestep:g} s, gravity "
             f"{'off' if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY else 'on'}",
             f"{'tendon':<22}{'group':<12}{'actuator':<22}{'branches':>9}{'sites':>7}{'length0 [mm]':>14}"]
    for tendon in config.get("tendons") or []:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon["name"])
        actuator = tendon.get("actuator") or {"type": "none"}
        kind = actuator.get("type", "none")
        if kind == "motor":
            kind = f"motor {actuator.get('max_force', td.DEFAULT_MOTOR_MAX_FORCE):g} N"
        elif kind == "position":
            kind = f"position kp={actuator.get('kp', td.DEFAULT_POSITION_KP):g}"
        num_sites = sum(1 for k in range(model.tendon_num[tid])
                        if model.wrap_type[model.tendon_adr[tid] + k] == mujoco.mjtWrap.mjWRAP_SITE)
        lines.append(f"{tendon['name']:<22}{str(tendon.get('group', '')):<12}{kind:<22}"
                     f"{1 + len(tendon.get('branches') or []):>9}{num_sites:>7}"
                     f"{1000 * model.tendon_length0[tid]:>14.2f}")
    return "\n".join(lines)


def main(
        argv: Optional[list] = None
        ) -> None:
    parser = argparse.ArgumentParser(description="Build a standalone MJCF folder from a Seahorse muscle playground config.")
    parser.add_argument("config", type=Path, help="configuration JSON saved by the Seahorse muscle playground")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="XML file name without .xml (default: config file name)")
    parser.add_argument("--num-segments", type=int, default=11)
    args = parser.parse_args(argv)

    config = td.load_config(args.config)
    name = args.name or re.sub(r"[^A-Za-z0-9_.-]+", "_", args.config.stem)
    try:
        xml_path = build_mjcf_folder(config, args.out_dir, name, num_segments=args.num_segments)
    except td.ConfigError as error:
        raise SystemExit(str(error))
    model = mujoco.MjModel.from_xml_path(str(xml_path))  # proves the folder loads in plain MuJoCo
    print(f"Wrote {xml_path}")
    print(summary(config, model))


if __name__ == "__main__":
    main()

"""Run a simple activation protocol on a Seahorse muscle playground configuration, headless, and save a CSV.

Protocol: optional settle time at a = 0, then a linear ramp of the activation from 0 to ``level``
over ``ramp`` seconds for the tendons of the chosen groups (default: every actuated tendon),
then ``hold`` seconds at ``level``. Other tendons stay at a = 0.

The measurements follow DESIGN.md "Measurement" (the same numbers the browser app shows), so
results from the web app can be reproduced here:

* angles (deg): for parent frame A and child frame B, ``v = R_A^T z_B``;
  ventral = atan2(v.x, v.z), lateral = atan2(-v.y, hypot(v.x, v.z)) (= asin(-v.y)), bend = acos(v.z),
  twist = swing-twist angle of B about its z axis relative to A.
  ``tip_*`` = last segment frame relative to ``segment_0_vertebrae``; ``seg{i}_*`` = segment i
  relative to segment i-1.
* ``seg{i}_torque_{axis}_Nm`` = ``qfrc_actuator`` at the vertebral joint dof (all tendon
  actuators together), ``..._passive_...`` = ``qfrc_passive``, ``..._limit_...`` = ``qfrc_constraint``.
* per tendon: force (pulling, N), length, excursion = length0 - length, strain = excursion / length0,
  work (J, integral of force * shortening velocity), and optionally moment arms
  (``ten_J`` at the vertebral dofs, m) and the torque this tendon alone applies (-moment_arm * force).

Usage:
    python -m seahorse_muscle_architecture.silico.tendon_designer.run_protocol config.json \
        --out results.csv [--groups dextral sinistral] [--ramp 1.0] [--hold 1.0] [--level 1.0] \
        [--settle 0.0] [--sample-dt 0.01] [--per-tendon-torques]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from seahorse_muscle_architecture.silico.tendon_designer import config as td


# ---------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------

def frame_angles(
        rot_a: np.ndarray,
        rot_b: np.ndarray
        ) -> Dict[str, float]:
    """Ventral / lateral / bend / twist (deg) of frame B relative to frame A (3x3 world rotations)."""
    rel = rot_a.T @ rot_b
    v = rel[:, 2]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rel.flatten())
    twist = 2.0 * math.atan2(quat[3], quat[0])
    twist = (twist + math.pi) % (2 * math.pi) - math.pi
    return {"ventral": math.degrees(math.atan2(v[0], v[2])),
            "lateral": math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2]))),
            "bend": math.degrees(math.acos(float(np.clip(v[2], -1.0, 1.0)))),
            "twist": math.degrees(twist)}


# ---------------------------------------------------------------------------------------------
# Simulation (mirrors web/js/sim.js Simulation)
# ---------------------------------------------------------------------------------------------

class TendonSimulation:
    """A compiled configuration with activations and DESIGN.md measurements."""

    def __init__(
            self,
            config: Dict,
            num_segments: int = 11,
            catalog: Optional[Dict] = None
            ) -> None:
        self.config = config
        self.catalog = catalog if catalog is not None else td.load_catalog(num_segments)
        self.mjcf_model = td.build_model(config, num_segments=num_segments, catalog=self.catalog)
        self.model = td.compile_model(self.mjcf_model)
        self.data = mujoco.MjData(self.model)
        self.num_segments = num_segments

        m = self.model
        self.frame_ids = [m.body(f"segment_{i}_vertebrae").id for i in range(num_segments)]
        self.joint_dofs = {(i, axis): int(m.jnt_dofadr[m.joint(f"segment_{i}_vertebrae_vertebrae_joint_{axis}").id])
                           for i in range(1, num_segments) for axis in td.AXES}
        self.joint_qpos = {(i, axis): int(m.jnt_qposadr[m.joint(f"segment_{i}_vertebrae_vertebrae_joint_{axis}").id])
                           for i in range(1, num_segments) for axis in td.AXES}
        self.tendons = []
        for tendon in config.get("tendons") or []:
            ids = td.tendon_ids(m, tendon)
            settings = td._actuator_settings(tendon)
            self.tendons.append({"name": tendon["name"], "group": tendon.get("group", ""), "cfg": tendon,
                                 "type": settings["type"], "tendon_id": ids["tendon"],
                                 "actuator_id": ids["actuator"],
                                 "length0": float(m.tendon_length0[ids["tendon"]])})
        self.activations = {t["name"]: 0.0 for t in self.tendons if t["actuator_id"] is not None}
        self.reset()

    @property
    def time(
            self
            ) -> float:
        return float(self.data.time)

    def set_activation(
            self,
            name: str,
            activation: float
            ) -> None:
        tendon = next(t for t in self.tendons if t["name"] == name)
        if tendon["actuator_id"] is None:
            return
        self.activations[name] = float(np.clip(activation, 0.0, 1.0))
        td.set_activation(self.model, self.data, tendon["cfg"], self.activations[name])

    def set_activations(
            self,
            activations: Dict[str, float]
            ) -> None:
        for name, activation in activations.items():
            self.set_activation(name, activation)

    def reset(
            self
            ) -> None:
        """State back to qpos0; activations are kept; work is zeroed."""
        mujoco.mj_resetData(self.model, self.data)
        for name, activation in self.activations.items():
            self.set_activation(name, activation)
        mujoco.mj_forward(self.model, self.data)
        self._tip_rest = self._tip_in_base()
        self.work = {t["name"]: 0.0 for t in self.tendons}

    def step(
            self,
            n: int = 1
            ) -> None:
        dt = self.model.opt.timestep
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            for tendon in self.tendons:
                self.work[tendon["name"]] += self.tendon_force(tendon) * -self.data.ten_velocity[
                    tendon["tendon_id"]] * dt

    def advance(
            self,
            seconds: float
            ) -> None:
        self.step(max(0, int(round(seconds / self.model.opt.timestep))))

    # --- measurements ---------------------------------------------------------------------------

    def tendon_force(
            self,
            tendon: Dict
            ) -> float:
        """Pulling force (N, >= 0), as in sim.js ``_tension``.

        Actuated: actuator tension = -actuator_force * gear (a motor's actuator_force is its
        ctrl, in [-1, 0]; its gear is max_force). Passive: ``passive_force``.
        """
        if tendon["actuator_id"] is not None:
            a = tendon["actuator_id"]
            return max(0.0, -float(self.data.actuator_force[a]) * float(self.model.actuator_gear[a, 0]))
        return self.passive_force(tendon)

    def passive_force(
            self,
            tendon: Dict
            ) -> float:
        """Elastic tendon force max(0, stiffness * (length - length0)) (N); damping not included."""
        t = tendon["tendon_id"]
        stretch = float(self.data.ten_length[t]) - tendon["length0"]
        return max(0.0, float(self.model.tendon_stiffness[t]) * stretch)

    def _tip_in_base(
            self
            ) -> np.ndarray:
        base, tip = self.frame_ids[0], self.frame_ids[-1]
        rot = self.data.xmat[base].reshape(3, 3)
        return rot.T @ (self.data.xpos[tip] - self.data.xpos[base])

    def _moment_arms(
            self,
            tendon_id: int
            ) -> np.ndarray:
        """Row of the tendon Jacobian (length nv) = moment arms d(length)/dq.

        With a sparse Jacobian (``mj_isSparse``, the default for this model) MuJoCo packs the
        rows into the (ntendon, nv) ``ten_J`` buffer (CSR: ten_J_rowadr / rownnz / colind).
        """
        m, d = self.model, self.data
        if not mujoco.mj_isSparse(m):
            return np.array(d.ten_J).reshape(m.ntendon, m.nv)[tendon_id].copy()
        # MuJoCo 3.2 keeps the sparsity pattern in mjData, newer versions in mjModel.
        pattern = d if hasattr(d, "ten_J_rowadr") else m
        values, columns = np.asarray(d.ten_J).ravel(), np.asarray(pattern.ten_J_colind).ravel()
        start, count = int(pattern.ten_J_rowadr[tendon_id]), int(pattern.ten_J_rownnz[tendon_id])
        row = np.zeros(m.nv)
        row[columns[start:start + count]] = values[start:start + count]
        return row

    def measure(
            self
            ) -> Dict:
        m, d = self.model, self.data
        rotations = [d.xmat[b].reshape(3, 3).copy() for b in self.frame_ids]
        base_pos, base_rot = d.xpos[self.frame_ids[0]], rotations[0]

        tip_pos = self._tip_in_base()
        displacement = tip_pos - self._tip_rest
        tip = {"pos": tip_pos.tolist(), "displacement": displacement.tolist(),
               "distance": float(np.linalg.norm(displacement)), **frame_angles(rotations[0], rotations[-1])}

        segments = []
        for i in range(1, self.num_segments):
            pos = base_rot.T @ (d.xpos[self.frame_ids[i]] - base_pos)
            entry = {"index": i, "pos": pos.tolist(), **frame_angles(rotations[i - 1], rotations[i])}
            entry["joint_angle"] = {axis: math.degrees(d.qpos[self.joint_qpos[(i, axis)]]) for axis in td.AXES}
            for key, source in (("torque", d.qfrc_actuator), ("passive_torque", d.qfrc_passive),
                                ("limit_torque", d.qfrc_constraint)):
                entry[key] = {axis: float(source[self.joint_dofs[(i, axis)]]) for axis in td.AXES}
            segments.append(entry)
        cumulative = [0.0] + list(np.cumsum([s["ventral"] for s in segments]))

        tendons = []
        for tendon in self.tendons:
            t = tendon["tendon_id"]
            length, length0 = float(d.ten_length[t]), tendon["length0"]
            force = self.tendon_force(tendon)
            row = self._moment_arms(t)
            arms, torques = [], []
            for i in range(1, self.num_segments):
                arm = {axis: float(row[self.joint_dofs[(i, axis)]]) for axis in td.AXES}
                arms.append({"segment": i, **arm})
                torques.append({"segment": i, **{axis: -arm[axis] * force for axis in td.AXES}})
            tendons.append({"name": tendon["name"], "group": tendon["group"],
                            "activation": self.activations.get(tendon["name"], 0.0),
                            "length": length, "length0": length0, "excursion": length0 - length,
                            "strain": (length0 - length) / length0 if length0 > 0 else 0.0,
                            "force": force, "passive_force": self.passive_force(tendon),
                            "work": self.work[tendon["name"]],
                            "moment_arms": arms, "torque": torques})
        return {"time": self.time, "tip": tip, "segments": segments,
                "cumulative_ventral": [float(c) for c in cumulative], "tendons": tendons}


# ---------------------------------------------------------------------------------------------
# Protocol + CSV
# ---------------------------------------------------------------------------------------------

def csv_header(
        sim: TendonSimulation,
        per_tendon_torques: bool = False
        ) -> List[str]:
    header = ["time_s"] + [f"act_{name}" for name in sim.activations]
    header += [f"tip_{q}_deg" for q in ("ventral", "lateral", "bend", "twist")]
    header += [f"tip_d{c}_m" for c in "xyz"] + ["tip_distance_m"]
    for i in range(1, sim.num_segments):
        header += [f"seg{i}_{q}_deg" for q in ("ventral", "lateral", "bend", "twist")]
        header += [f"seg{i}_joint_{axis}_deg" for axis in td.AXES]
        for kind in ("torque", "passive_torque", "limit_torque"):
            header += [f"seg{i}_{kind}_{axis}_Nm" for axis in td.AXES]
    header += [f"seg{i}_cumulative_ventral_deg" for i in range(1, sim.num_segments)]
    for tendon in sim.tendons:
        n = tendon["name"]
        header += [f"tendon_{n}_force_N", f"tendon_{n}_length_m", f"tendon_{n}_excursion_m",
                   f"tendon_{n}_strain", f"tendon_{n}_work_J"]
        if per_tendon_torques:
            for i in range(1, sim.num_segments):
                header += [f"tendon_{n}_moment_arm_seg{i}_{axis}_m" for axis in td.AXES]
                header += [f"tendon_{n}_torque_seg{i}_{axis}_Nm" for axis in td.AXES]
    return header


def csv_row(
        sim: TendonSimulation,
        measurement: Dict,
        per_tendon_torques: bool = False
        ) -> List[float]:
    tip = measurement["tip"]
    row = [measurement["time"]] + [sim.activations[name] for name in sim.activations]
    row += [tip[q] for q in ("ventral", "lateral", "bend", "twist")] + list(tip["displacement"]) + [tip["distance"]]
    for segment in measurement["segments"]:
        row += [segment[q] for q in ("ventral", "lateral", "bend", "twist")]
        row += [segment["joint_angle"][axis] for axis in td.AXES]
        for kind in ("torque", "passive_torque", "limit_torque"):
            row += [segment[kind][axis] for axis in td.AXES]
    row += measurement["cumulative_ventral"][1:]
    for tendon in measurement["tendons"]:
        row += [tendon["force"], tendon["length"], tendon["excursion"], tendon["strain"], tendon["work"]]
        if per_tendon_torques:
            for arm, torque in zip(tendon["moment_arms"], tendon["torque"]):
                row += [arm[axis] for axis in td.AXES] + [torque[axis] for axis in td.AXES]
    return row


def ramp_activation(
        t: float,
        settle: float,
        ramp: float,
        level: float
        ) -> float:
    if t <= settle:
        return 0.0
    if ramp <= 0:
        return level
    return level * min(1.0, (t - settle) / ramp)


def run_protocol(
        config: Dict,
        groups: Optional[Sequence[str]] = None,
        ramp: float = 1.0,
        hold: float = 1.0,
        level: float = 1.0,
        settle: float = 0.0,
        sample_dt: float = 0.01,
        num_segments: int = 11,
        per_tendon_torques: bool = False,
        out: Optional[Path] = None,
        sim: Optional[TendonSimulation] = None
        ) -> Tuple[List[str], List[List[float]], TendonSimulation]:
    """Run the ramp-and-hold protocol; returns (header, rows, simulation) and writes ``out`` if given.

    The activation is updated every physics step; a row is recorded every ``sample_dt`` seconds
    (rounded to whole steps), starting at t = 0.
    """
    sim = sim if sim is not None else TendonSimulation(config, num_segments=num_segments)
    driven = [t["name"] for t in sim.tendons if t["actuator_id"] is not None
              and (groups is None or t["group"] in groups)]
    if groups is not None and not driven:
        known = sorted({t["group"] for t in sim.tendons if t["actuator_id"] is not None})
        raise ValueError(f"no actuated tendon in groups {list(groups)} (groups in this config: {known})")
    for name in sim.activations:
        sim.set_activation(name, 0.0)
    sim.reset()

    dt = sim.model.opt.timestep
    total_steps = int(round((settle + ramp + hold) / dt))
    sample_every = max(1, int(round(sample_dt / dt)))
    header = csv_header(sim, per_tendon_torques)
    rows = [csv_row(sim, sim.measure(), per_tendon_torques)]
    for step in range(1, total_steps + 1):
        activation = ramp_activation(step * dt, settle, ramp, level)
        for name in driven:
            sim.set_activation(name, activation)
        sim.step()
        if step % sample_every == 0 or step == total_steps:
            rows.append(csv_row(sim, sim.measure(), per_tendon_torques))

    if out is not None:
        write_csv(out, header, rows)
    return header, rows, sim


def write_csv(
        path: Path,
        header: Sequence[str],
        rows: Iterable[Sequence[float]]
        ) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{value:.9g}" for value in row])


def main(
        argv: Optional[list] = None
        ) -> None:
    parser = argparse.ArgumentParser(description="Run a ramp-and-hold activation protocol on a Seahorse muscle playground config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="CSV file (default: <config>_protocol.csv)")
    parser.add_argument("--groups", nargs="*", default=None, help="tendon groups to activate (default: all actuated)")
    parser.add_argument("--ramp", type=float, default=1.0, help="ramp duration 0 -> level [s]")
    parser.add_argument("--hold", type=float, default=1.0, help="hold duration at level [s]")
    parser.add_argument("--level", type=float, default=1.0, help="final activation (0..1)")
    parser.add_argument("--settle", type=float, default=0.0, help="time at a = 0 before the ramp [s]")
    parser.add_argument("--sample-dt", type=float, default=0.01, help="CSV row interval [s]")
    parser.add_argument("--num-segments", type=int, default=11)
    parser.add_argument("--per-tendon-torques", action="store_true",
                        help="also write per-tendon moment arms and torque contributions")
    args = parser.parse_args(argv)

    config = td.load_config(args.config)
    out = args.out or args.config.with_name(args.config.stem + "_protocol.csv")
    try:
        header, rows, sim = run_protocol(config, groups=args.groups, ramp=args.ramp, hold=args.hold,
                                         level=args.level, settle=args.settle, sample_dt=args.sample_dt,
                                         num_segments=args.num_segments,
                                         per_tendon_torques=args.per_tendon_torques, out=out)
    except (td.ConfigError, ValueError) as error:
        raise SystemExit(str(error))

    final = sim.measure()
    print(f"Wrote {out} ({len(rows)} rows x {len(header)} columns, t = 0..{final['time']:.3f} s)")
    print(f"Tip: ventral {final['tip']['ventral']:.2f} deg, lateral {final['tip']['lateral']:.2f} deg, "
          f"displacement {1000 * final['tip']['distance']:.2f} mm")
    torques = ", ".join(f"{s['index']}: {1000 * s['torque']['pitch']:+.2f}" for s in final["segments"])
    print(f"Vertebral pitch torque [N.mm] per segment: {torques}")
    for tendon in final["tendons"]:
        print(f"  {tendon['name']:<20} a={tendon['activation']:.2f} force {tendon['force']:.3f} N, "
              f"excursion {1000 * tendon['excursion']:.2f} mm, strain {100 * tendon['strain']:.2f} %")


if __name__ == "__main__":
    main()

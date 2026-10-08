"""Tests for the Python side of the Seahorse muscle playground (config.py, build_mjcf.py, run_protocol.py).

Run from the repository root:
    .venv/bin/python -m unittest seahorse_muscle_architecture.silico.tendon_designer.tests.test_config -v
(or with pytest, if installed). Building a model takes ~1.5 s, so the full run takes ~1 min.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from seahorse_muscle_architecture.silico.tendon_designer import build_mjcf, config as td, run_protocol
from seahorse_muscle_architecture.silico.tendon_designer.tests import parity_check

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
PRESETS_DIR = td.TENDON_DESIGNER_DIR / "web" / "configs"
CATALOG = td.load_catalog(11)


def tap(
        segment: int,
        plate: int,
        kind: str = "end",
        key=None
        ) -> str:
    base = f"segment_{segment}_plate_{plate}_"
    return {"end": base + "end_hm_tap", "int": base + f"intermediate_hm_tap_{key}",
            "ghost": base + f"ghost_hm_tap_{key}", "mvm": base + f"mvm_tap_{key}"}[kind]


def sites(
        items
        ):
    return [item["site"] if "site" in item else "|" for item in items]


def config_files():
    """Fixtures + presets (web/configs/*.json without index.json)."""
    files = sorted(FIXTURES_DIR.glob("*.json"))
    if PRESETS_DIR.is_dir():
        for path in sorted(PRESETS_DIR.glob("*.json")):
            data = json.loads(path.read_text())
            if isinstance(data, dict) and data.get("format") == td.CONFIG_FORMAT:
                files.append(path)
    return files


_compiled = {}


def compiled(
        path: Path
        ) -> mujoco.MjModel:
    if path not in _compiled:
        _compiled[path] = td.compile_model(td.build_model(td.load_config(path), catalog=CATALOG))
    return _compiled[path]


class ExpansionTest(unittest.TestCase):
    def test_forward_trunk(
            self
            ):
        tendon = {"name": "t", "path": [tap(1, 0), tap(5, 0, "int", 3), tap(9, 0)]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, CATALOG)), [
                tap(1, 0) + "_0", tap(1, 0) + "_1",
                tap(5, 0, "int", 3) + "_0", tap(5, 0, "int", 3) + "_1",
                tap(9, 0) + "_0", tap(9, 0) + "_1"])

    def test_backward_trunk_reverses_every_point(
            self
            ):
        tendon = {"name": "t", "path": [tap(9, 1, "mvm", "sinistral"), tap(6, 1, "ghost", "a"), tap(2, 1)]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, CATALOG)), [
                tap(9, 1, "mvm", "sinistral") + "_1", tap(9, 1, "mvm", "sinistral") + "_0",
                tap(6, 1, "ghost", "a") + "_1", tap(6, 1, "ghost", "a") + "_0",
                tap(2, 1) + "_1", tap(2, 1) + "_0"])

    def test_same_segment_counts_as_forward(
            self
            ):
        tendon = {"name": "t", "path": [tap(4, 0, "int", 2), tap(7, 0), tap(4, 0, "int", 7)]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, CATALOG))[0], tap(4, 0, "int", 2) + "_0")

    def test_catalog_site_order_is_proximal_to_distal(
            self
            ):
        model = compiled(FIXTURES_DIR / "fork.json")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for segment in CATALOG["segments"]:
            frame = model.body(segment["frame_body"]).id
            rot = data.xmat[frame].reshape(3, 3)
            for plate in segment["plates"]:
                for t in plate["taps"]:
                    z = [(rot.T @ (data.site_xpos[model.site(s).id] - data.xpos[frame]))[2] for s in t["sites"]]
                    self.assertTrue(all(a < b for a, b in zip(z, z[1:])), t["id"])

    def test_forward_branch(
            self
            ):
        tendon = {"name": "t", "path": [tap(1, 0), tap(5, 0, "int", 3), tap(9, 0)],
                  "branches": [{"from": 1, "path": [tap(8, 0, "int", 5), tap(10, 0)]}]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, CATALOG))[6:], [
                "|", tap(5, 0, "int", 3) + "_1",
                tap(8, 0, "int", 5) + "_0", tap(8, 0, "int", 5) + "_1", tap(10, 0) + "_0", tap(10, 0) + "_1"])

    def test_backward_branch_uses_proximal_exit_site(
            self
            ):
        tendon = {"name": "t", "path": [tap(1, 1), tap(5, 1, "int", 3), tap(9, 1)],
                  "branches": [{"from": 1, "path": [tap(3, 1, "int", 2)]}]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, CATALOG))[6:], [
                "|", tap(5, 1, "int", 3) + "_0", tap(3, 1, "int", 2) + "_1", tap(3, 1, "int", 2) + "_0"])

    def test_branch_on_backward_trunk_has_its_own_direction(
            self
            ):
        tendon = {"name": "t", "path": [tap(9, 1, "mvm", "sinistral"), tap(2, 1, "mvm", "sinistral")],
                  "branches": [{"from": 0, "path": [tap(10, 1, "mvm", "dextral")]}]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, CATALOG))[4:], [
                "|", tap(9, 1, "mvm", "sinistral") + "_1",
                tap(10, 1, "mvm", "dextral") + "_0", tap(10, 1, "mvm", "dextral") + "_1"])

    def test_free_points(
            self
            ):
        free_points = [{"id": "free_1", "segment": 5, "plate": 0, "xy": [0.031, -0.02]}]
        forward = {"name": "t", "path": [tap(2, 0), "free_1"]}
        backward = {"name": "t", "path": ["free_1", tap(2, 0)]}
        self.assertEqual(sites(td.expand_tendon_sites(forward, CATALOG, free_points))[2:], ["free_1_0", "free_1_1"])
        self.assertEqual(sites(td.expand_tendon_sites(backward, CATALOG, free_points))[:2], ["free_1_1", "free_1_0"])

    def test_consecutive_duplicates_dropped_within_branch_only(
            self
            ):
        # Synthetic catalog with single-site points, where rule 4 matters.
        catalog = {"num_segments": 3, "segments": [
                {"index": s, "plates": [{"index": 0, "body": f"b{s}", "free_site_z": [0, 0], "taps": [
                        {"id": f"p{s}", "kind": "end", "xy": [0, 0], "sites": [f"s{s}"]},
                        {"id": f"q{s}", "kind": "end", "xy": [0, 0], "sites": [f"s{s}", f"t{s}"]}]}]}
                for s in range(3)]}
        tendon = {"name": "t", "path": ["p0", "p0", "q1", "p2"], "branches": [{"from": 1, "path": ["p0", "p2"]}]}
        # trunk: s0 (dup s0 dropped), s1, t1, s2 ; branch Db=+1: exit s0, p0 -> s0 dropped, s2
        self.assertEqual(sites(td.expand_tendon_sites(tendon, catalog)), ["s0", "s1", "t1", "s2", "|", "s0", "s2"])
        # a branch may repeat the trunk's last site (duplicates are only dropped within a branch)
        tendon = {"name": "t", "path": ["p0", "p2"], "branches": [{"from": 1, "path": ["p0"]}]}
        self.assertEqual(sites(td.expand_tendon_sites(tendon, catalog)), ["s0", "s2", "|", "s2", "s0"])

    def test_errors(
            self
            ):
        with self.assertRaises(td.ConfigError):
            td.expand_tendon_sites({"name": "t", "path": [tap(1, 0)]}, CATALOG)
        with self.assertRaises(td.ConfigError):
            td.expand_tendon_sites({"name": "t", "path": [tap(1, 0), "nope"]}, CATALOG)
        with self.assertRaises(td.ConfigError):
            td.expand_tendon_sites({"name": "t", "path": [tap(1, 0), tap(3, 0)],
                                    "branches": [{"from": 2, "path": [tap(4, 0)]}]}, CATALOG)
        with self.assertRaises(td.ConfigError):
            td.expand_tendon_sites({"name": "t", "path": [tap(1, 0), tap(3, 0)],
                                    "branches": [{"from": 0, "path": []}]}, CATALOG)


class ValidationTest(unittest.TestCase):
    def test_fixtures_are_valid(
            self
            ):
        for path in config_files():
            with self.subTest(path.name):
                self.assertEqual(td.validate_config(td.load_config(path), CATALOG), [])

    def test_invalid_configs(
            self
            ):
        good = {"name": "a", "path": [tap(1, 0), tap(3, 0)], "actuator": {"type": "motor", "max_force": 5}}
        cases = {
                "bad name": [dict(good, name="has space")],
                "segment prefix": [dict(good, name="segment_x")],
                "duplicate": [good, dict(good)],
                "unknown point": [dict(good, path=[tap(1, 0), "missing"])],
                "short path": [dict(good, path=[tap(1, 0)])],
                "branch index": [dict(good, branches=[{"from": 5, "path": [tap(4, 0)]}])],
                "branch bool": [dict(good, branches=[{"from": True, "path": [tap(4, 0)]}])],
                "actuator type": [dict(good, actuator={"type": "muscle"})],
                "max force": [dict(good, actuator={"type": "motor", "max_force": 0})],
                "max strain": [dict(good, actuator={"type": "position", "max_strain": 1.5})],
                "negative stiffness": [dict(good, stiffness=-1)],
                }
        for label, tendons in cases.items():
            with self.subTest(label):
                self.assertTrue(td.validate_config({"tendons": tendons}, CATALOG))
        self.assertEqual(td.validate_config({"tendons": [good]}, CATALOG), [])
        bad_free = [{"id": tap(1, 0), "segment": 1, "plate": 0, "xy": [0, 0]},
                    {"id": "f", "segment": 11, "plate": 0, "xy": [0, 0]},
                    {"id": "g", "segment": 1, "plate": 0, "xy": [0]}]
        self.assertEqual(len(td.validate_config({"free_points": bad_free}, CATALOG)), 3)
        for params in ({"orientation": "sideways"}, {"timestep": 0}, {"gravity": "yes"},
                       {"vertebra": {"pitch_stiffness": -1}}, {"stiffness_taper": "x"}):
            with self.subTest(params=params):
                self.assertTrue(td.validate_config({"params": params}, CATALOG))
        warnings = []
        td.validate_config({"params": {"unknown": 1}, "tendons": [dict(good, path=[tap(3, 0), tap(3, 1)])]},
                           CATALOG, warnings)
        self.assertEqual(len(warnings), 2)

    def test_apply_raises_with_all_errors(
            self
            ):
        from seahorse_muscle_architecture.silico.tendon_designer.export_assets import build_base_mjcf
        with self.assertRaises(td.ConfigError) as context:
            td.apply_tendon_config(build_base_mjcf(11), {"tendons": [{"name": "segment_a", "path": []}]}, CATALOG)
        self.assertEqual(len(context.exception.errors), 2)


class BuildTest(unittest.TestCase):
    def test_every_config_compiles(
            self
            ):
        files = config_files()
        self.assertGreaterEqual(len(files), 5)
        for path in files:
            with self.subTest(path.name):
                cfg = td.load_config(path)
                mjcf_model = td.build_model(cfg, catalog=CATALOG)
                model = mujoco.MjModel.from_xml_string(mjcf_model.to_xml_string(), mjcf_model.get_assets())
                for tendon in cfg["tendons"]:
                    self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon["name"]), 0)
                    self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                                              tendon["name"] + "_length"), 0)
                data = mujoco.MjData(model)
                mujoco.mj_step(model, data, 10)
                self.assertTrue(np.all(np.isfinite(data.qpos)))

    def test_fork_length0_is_sum_of_branch_polylines(
            self
            ):
        cfg = td.load_config(FIXTURES_DIR / "fork.json")
        model = compiled(FIXTURES_DIR / "fork.json")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for tendon in cfg["tendons"]:
            branches = td.split_branches(td.expand_tendon_sites(tendon, CATALOG, cfg["free_points"]))
            self.assertEqual(len(branches), 1 + len(tendon["branches"]))
            total = 0.0
            for branch in branches:
                points = np.array([data.site_xpos[model.site(s).id] for s in branch])
                total += np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            tid = model.tendon(tendon["name"]).id
            self.assertAlmostEqual(model.tendon_length0[tid], total, places=12)
            self.assertAlmostEqual(data.ten_length[tid], total, places=12)

    def test_tendon_actuator_and_sensor_elements(
            self
            ):
        model = compiled(FIXTURES_DIR / "passive.json")
        driver, passive = model.tendon("driver").id, model.tendon("passive_dorsal").id
        self.assertEqual(model.nu, 1)
        self.assertAlmostEqual(model.actuator_gear[0, 0], 10)
        np.testing.assert_allclose(model.actuator_ctrlrange[0], [-1, 0])
        np.testing.assert_allclose(model.actuator_forcerange[0], [-10, 0])
        self.assertTrue(model.actuator_ctrllimited[0] and model.actuator_forcelimited[0])
        self.assertAlmostEqual(model.tendon_stiffness[passive], 200)
        self.assertAlmostEqual(model.tendon_damping[passive], 0.05)
        np.testing.assert_allclose(model.tendon_lengthspring[passive], [model.tendon_length0[passive]] * 2)
        np.testing.assert_allclose(model.tendon_rgba[driver], td.color_to_rgba("#2a78d6"))
        names = {model.sensor(i).name for i in range(model.nsensor)}
        self.assertTrue({"driver_length", "driver_force", "passive_dorsal_length"} <= names)
        self.assertNotIn("passive_dorsal_force", names)

        model = compiled(FIXTURES_DIR / "backward.json")
        a = model.actuator("back").id
        self.assertEqual(model.actuator_biastype[a], mujoco.mjtBias.mjBIAS_AFFINE)
        self.assertAlmostEqual(model.actuator_gainprm[a, 0], 1000)
        self.assertAlmostEqual(model.actuator_biasprm[a, 1], -1000)
        self.assertFalse(model.actuator_ctrllimited[a])
        np.testing.assert_allclose(model.actuator_forcerange[a], [-1000, 0])

    def test_free_point_sites(
            self
            ):
        cfg = td.load_config(FIXTURES_DIR / "free_point.json")
        model = compiled(FIXTURES_DIR / "free_point.json")
        for fp in cfg["free_points"]:
            plate = td.find_plate(CATALOG, fp["segment"], fp["plate"])
            for k, z in enumerate(plate["free_site_z"]):
                site = model.site(f"{fp['id']}_{k}")
                self.assertEqual(model.body(site.bodyid[0]).name, plate["body"])
                np.testing.assert_allclose(site.pos, [*fp["xy"], z])

    def test_params_land_in_compiled_model(
            self
            ):
        cfg = td.load_config(FIXTURES_DIR / "hanging_params.json")
        params = cfg["params"]
        model = compiled(FIXTURES_DIR / "hanging_params.json")
        defaults = CATALOG["defaults"]["vertebra"]
        n = CATALOG["num_segments"]
        for i in range(1, n):
            factor = 1 + (params["stiffness_taper"] - 1) * (i - 1) / (n - 2)
            for axis in td.AXES:
                joint = model.joint(f"segment_{i}_vertebrae_vertebrae_joint_{axis}")
                base = params["vertebra"].get(f"{axis}_stiffness", defaults[f"{axis}_stiffness"])
                self.assertAlmostEqual(joint.stiffness[0], base * factor, places=10)
                damping = params["vertebra"].get(f"{axis}_damping", defaults[f"{axis}_damping"])
                self.assertAlmostEqual(model.dof_damping[joint.dofadr[0]], damping, places=12)
                rng = math.radians(params["vertebra"].get(f"{axis}_range_deg", defaults[f"{axis}_range_deg"]))
                np.testing.assert_allclose(joint.range, [-rng, rng], atol=1e-12)
        self.assertAlmostEqual(model.joint("segment_10_vertebrae_vertebrae_joint_pitch").stiffness[0], 0.006)
        glide = model.joint("segment_3_plate_2_y_axis")
        self.assertAlmostEqual(glide.stiffness[0], 2.0)
        self.assertAlmostEqual(model.dof_damping[glide.dofadr[0]], 0.5)
        strut = model.tendon("segment_4_vertebral_vertebral_strut_dorsal").id
        self.assertAlmostEqual(model.tendon_stiffness[strut], 5.0)
        self.assertAlmostEqual(model.tendon_damping[strut], 2.0)
        self.assertAlmostEqual(model.opt.timestep, 0.001)
        self.assertFalse(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        self.assertTrue(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        np.testing.assert_allclose(model.body("segment_0").quat, [0, 1, 0, 0], atol=1e-12)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        tip = data.xpos[model.body("segment_10_vertebrae").id]
        self.assertLess(tip[2], -0.3)  # hanging: tail along -z

    def test_base_model_untouched_without_params(
            self
            ):
        model = compiled(FIXTURES_DIR / "fork.json")
        self.assertAlmostEqual(model.opt.timestep, CATALOG["defaults"]["timestep"])
        self.assertTrue(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        joint = model.joint("segment_5_vertebrae_vertebrae_joint_pitch")
        self.assertAlmostEqual(joint.stiffness[0], CATALOG["defaults"]["vertebra"]["pitch_stiffness"])

    def test_web_base_matches_morphology(
            self
            ):
        """web/model/base.xml (used by the browser) is in sync with build_base_mjcf."""
        cfg = td.load_config(FIXTURES_DIR / "free_point.json")
        diffs = parity_check.compare_models(parity_check.build_python_model(cfg, "web"), compiled(
                FIXTURES_DIR / "free_point.json"))
        self.assertEqual(diffs, [])

    def test_build_mjcf_cli(
            self
            ):
        out_dir = Path(tempfile.mkdtemp())
        try:
            build_mjcf.main([str(FIXTURES_DIR / "fork.json"), "--out-dir", str(out_dir), "--name", "x"])
            model = mujoco.MjModel.from_xml_path(str(out_dir / "x.xml"))
            self.assertEqual(model.nu, 2)
            self.assertTrue((out_dir / "x.config.json").exists())
        finally:
            shutil.rmtree(out_dir)


class ActivationAndMeasurementTest(unittest.TestCase):
    def test_activation_to_ctrl(
            self
            ):
        model = compiled(FIXTURES_DIR / "backward.json")
        position = td.load_config(FIXTURES_DIR / "backward.json")["tendons"][0]
        length0 = model.tendon_length0[model.tendon("back").id]
        self.assertAlmostEqual(td.activation_to_ctrl(model, position, 0.0), length0)
        self.assertAlmostEqual(td.activation_to_ctrl(model, position, 0.5), length0 * (1 - 0.5 * 0.2))
        self.assertAlmostEqual(td.activation_to_ctrl(model, position, 2.0), length0 * (1 - 0.2))
        self.assertEqual(td.activation_to_ctrl(model, {"name": "x", "actuator": {"type": "motor"}}, 0.3), -0.3)
        self.assertIsNone(td.activation_to_ctrl(model, {"name": "x", "actuator": {"type": "none"}}, 0.3))
        default_strain = {"name": "back", "actuator": {"type": "position"}}
        self.assertAlmostEqual(td.activation_to_ctrl(model, default_strain, 1.0), length0 * (1 - td.DEFAULT_MAX_STRAIN))

    def test_frame_angles(
            self
            ):
        theta = math.radians(12)
        c, s = math.cos(theta), math.sin(theta)
        about_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        about_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        about_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        eye = np.eye(3)
        self.assertAlmostEqual(run_protocol.frame_angles(eye, about_y)["ventral"], 12)
        self.assertAlmostEqual(run_protocol.frame_angles(eye, about_x)["lateral"], 12)
        self.assertAlmostEqual(run_protocol.frame_angles(eye, about_x)["bend"], 12)
        self.assertAlmostEqual(run_protocol.frame_angles(eye, about_z)["twist"], 12)
        self.assertAlmostEqual(run_protocol.frame_angles(about_y, about_y @ about_z)["twist"], 12)
        self.assertAlmostEqual(run_protocol.frame_angles(about_z, about_z)["twist"], 0)
        # beyond 90 deg of ventral bend, the lateral angle keeps its sign: asin(-v.y)
        big = math.radians(120)
        about_y_big = np.array([[math.cos(big), 0, math.sin(big)], [0, 1, 0], [-math.sin(big), 0, math.cos(big)]])
        angles = run_protocol.frame_angles(eye, about_y_big @ about_x)
        self.assertAlmostEqual(angles["ventral"], 120)
        self.assertAlmostEqual(angles["lateral"], 12)

    def test_measurement_consistency(
            self
            ):
        sim = run_protocol.TendonSimulation(td.load_config(FIXTURES_DIR / "fork.json"), catalog=CATALOG)
        rest = sim.measure()
        self.assertAlmostEqual(rest["tip"]["distance"], 0)
        self.assertTrue(all(abs(s["ventral"]) < 1e-9 for s in rest["segments"]))
        sim.set_activations({"hm_dextral": 0.4, "hm_sinistral": 0.2})
        sim.advance(0.1)
        m = sim.measure()
        self.assertAlmostEqual(m["time"], 0.1)
        forces = {t["name"]: t["force"] for t in m["tendons"]}
        self.assertAlmostEqual(forces["hm_dextral"], 4.0)
        self.assertAlmostEqual(forces["hm_sinistral"], 2.0)
        for segment in m["segments"]:
            for axis in td.AXES:
                summed = sum(t["torque"][segment["index"] - 1][axis] for t in m["tendons"])
                self.assertAlmostEqual(summed, segment["torque"][axis], places=10)
        self.assertGreater(m["tip"]["distance"], 0)
        self.assertGreater(m["tendons"][0]["work"], 0)
        for t in m["tendons"]:
            self.assertAlmostEqual(t["excursion"], t["length0"] - t["length"])
            sensor = sim.model.sensor(t["name"] + "_length")
            self.assertAlmostEqual(sim.data.sensordata[sensor.adr[0]], t["length"])
        self.assertAlmostEqual(m["cumulative_ventral"][-1], sum(s["ventral"] for s in m["segments"]))
        sim.reset()
        self.assertEqual(sim.activations["hm_dextral"], 0.4)  # activations kept
        self.assertAlmostEqual(sim.measure()["tip"]["distance"], 0)

    def test_run_protocol_csv(
            self
            ):
        out = Path(tempfile.mkdtemp()) / "run.csv"
        try:
            header, rows, sim = run_protocol.run_protocol(
                    td.load_config(FIXTURES_DIR / "passive.json"), groups=["ventral"], ramp=0.1, hold=0.05,
                    sample_dt=0.01, per_tendon_torques=True, out=out)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(len(row) == len(header) for row in rows))
            lines = out.read_text().splitlines()
            self.assertEqual(lines[0].split(","), header)
            column = header.index("act_driver")
            self.assertAlmostEqual(rows[5][column], 0.5)
            self.assertAlmostEqual(rows[-1][column], 1.0)
            self.assertIn("seg5_torque_pitch_Nm", header)
            self.assertIn("tendon_passive_dorsal_force_N", header)
            with self.assertRaises(ValueError):
                run_protocol.run_protocol(td.load_config(FIXTURES_DIR / "passive.json"), groups=["nope"],
                                          sim=sim)
        finally:
            shutil.rmtree(out.parent)


class ParityTest(unittest.TestCase):
    """Python vs JS builder (web/js/model_builder.js).

    Uses the XML files in TD_JS_XML_DIR (``<config stem>.xml``) if that variable is set.
    Otherwise it runs ``node tests/node/build_xml.mjs`` for every fixture/preset, if Node.js
    and the node test dependencies (``npm install`` in tests/node) are available.
    """

    def test_parity_with_js_builder(
            self
            ):
        xml_dir = os.environ.get("TD_JS_XML_DIR")
        temp_dir = None
        node_dir = TESTS_DIR / "node"
        if not xml_dir:
            node = shutil.which("node")
            if node is None or not (node_dir / "build_xml.mjs").exists() or not (node_dir / "node_modules").exists():
                self.skipTest("TD_JS_XML_DIR not set and node / tests/node/build_xml.mjs not available")
            temp_dir = tempfile.mkdtemp()
            xml_dir = temp_dir
            for path in config_files():
                subprocess.run([node, "build_xml.mjs", str(path), str(Path(xml_dir) / f"{path.stem}.xml")],
                               cwd=node_dir, check=True, capture_output=True)
        try:
            checked = 0
            for path in config_files():
                xml_path = Path(xml_dir) / f"{path.stem}.xml"
                if not xml_path.exists():
                    continue
                with self.subTest(path.name):
                    self.assertEqual(parity_check.check(path, xml_path), [])
                    checked += 1
            self.assertGreater(checked, 0)
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()

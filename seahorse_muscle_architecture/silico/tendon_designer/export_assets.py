"""Export the base model and site catalogue used by the Tendon Designer web app.

Writes into ``web/model/``:

* ``base.xml`` + meshes: the seahorse tail with all tendon attachment sites, but
  without any HM or MVM tendons, actuators or their sensors. The vertebral struts
  (passive) are kept.
* ``catalog.json``: per segment and plate, the clickable attachment points (taps)
  with their MuJoCo site names, their position in the segment frame, the 2D
  silhouettes of the plates and vertebra for the slice view, and the default
  joint/strut parameters. See ``DESIGN.md`` for the format.

Usage:
    python -m seahorse_muscle_architecture.silico.tendon_designer.export_assets [--num-segments 11]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np
from contourpy import contour_generator
from dm_control import mjcf
from PIL import Image, ImageDraw

from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.morphology import MJCFSeahorseMorphology
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import \
    default_seahorse_morphology_specification

WEB_MODEL_DIR = Path(__file__).parent / "web" / "model"
CORNERS = ["ventral_dextral", "ventral_sinistral", "dorsal_sinistral", "dorsal_dextral"]

# Raster resolution used to compute the 2D silhouettes (metres per pixel).
SILHOUETTE_RESOLUTION = 0.00025
# Douglas-Peucker tolerance for simplifying the silhouettes (metres).
SILHOUETTE_TOLERANCE = 0.0002

TAP_PATTERNS = [
        ("intermediate", re.compile(r"^(segment_(\d+)_plate_(\d+)_intermediate_hm_tap_(\d+))_(\d)$")),
        ("ghost", re.compile(r"^(segment_(\d+)_plate_(\d+)_ghost_hm_tap_([ab]))_(\d)$")),
        ("end", re.compile(r"^(segment_(\d+)_plate_(\d+)_end_hm_tap)_(\d)$")),
        ("mvm", re.compile(r"^(segment_(\d+)_plate_(\d+)_mvm_tap_(sinistral|dextral))_(\d)$")),
        ]


def build_base_mjcf(
        num_segments: int
        ) -> mjcf.RootElement:
    """Build the morphology with every attachment site but no HM/MVM tendons."""
    # MVMs are enabled only so that their attachment sites get created.
    specification = default_seahorse_morphology_specification(
            num_segments=num_segments, hm_segment_span=0, p_control=False, mvm_enabled=True, mvm_strain=1.0
            )
    morphology = MJCFSeahorseMorphology(specification=specification)
    mjcf_model = morphology.mjcf_model
    mjcf_model.model = "seahorse_tendon_designer_base"

    # Sensors first: they hold references to the actuators/tendons removed below.
    for sensor in list(mjcf_model.find_all("sensor")):
        if getattr(sensor, "actuator", None) is not None or getattr(sensor, "tendon", None) is not None:
            sensor.remove()
    for actuator in list(mjcf_model.find_all("actuator")):
        actuator.remove()
    for tendon in list(mjcf_model.find_all("tendon")):
        if not tendon.name.startswith("segment_"):  # keep the vertebral struts
            tendon.remove()
    return mjcf_model


def _simplify(
        points: np.ndarray,
        tolerance: float
        ) -> np.ndarray:
    """Douglas-Peucker simplification of an (open or closed) polyline."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    segment = end - start
    norm = np.linalg.norm(segment)
    if norm == 0:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        offsets = points - start
        distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / norm
    index = int(np.argmax(distances))
    if distances[index] > tolerance:
        left = _simplify(points[:index + 1], tolerance)
        right = _simplify(points[index:], tolerance)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def silhouette(
        triangles_xy: np.ndarray
        ) -> List[List[List[float]]]:
    """Outline polygons (outer boundaries and holes) of a set of projected triangles."""
    lower = triangles_xy.reshape(-1, 2).min(axis=0) - 4 * SILHOUETTE_RESOLUTION
    upper = triangles_xy.reshape(-1, 2).max(axis=0) + 4 * SILHOUETTE_RESOLUTION
    size = np.ceil((upper - lower) / SILHOUETTE_RESOLUTION).astype(int) + 1

    image = Image.new("L", (int(size[0]), int(size[1])), 0)
    draw = ImageDraw.Draw(image)
    pixels = (triangles_xy - lower) / SILHOUETTE_RESOLUTION
    for triangle in pixels:
        draw.polygon([tuple(p) for p in triangle], fill=1)
    mask = np.asarray(image, dtype=float)

    polygons = []
    for line in contour_generator(z=mask).lines(0.5):
        if len(line) < 4:
            continue
        line_xy = line * SILHOUETTE_RESOLUTION + lower  # contourpy returns (column, row) = (x, y)
        simplified = _simplify(line_xy, SILHOUETTE_TOLERANCE)
        if len(simplified) >= 3:
            polygons.append(np.round(simplified, 5).tolist())
    return polygons


def _body_subtree(
        model: mujoco.MjModel,
        root: int
        ) -> List[int]:
    return [b for b in range(model.nbody) if _is_descendant(model, b, root)]


def _is_descendant(
        model: mujoco.MjModel,
        body: int,
        root: int
        ) -> bool:
    while True:
        if body == root:
            return True
        if body == 0:
            return False
        body = model.body_parentid[body]


def geoms_outline(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geoms: List[int],
        frame_pos: np.ndarray,
        frame_mat: np.ndarray
        ) -> List[List[List[float]]]:
    """Silhouette of the given mesh geoms, projected on the xy plane of a frame."""
    triangles = []
    for geom in geoms:
        mesh = model.geom_dataid[geom]
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or mesh < 0:
            continue
        vert_adr, num_vert = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        face_adr, num_face = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = model.mesh_vert[vert_adr:vert_adr + num_vert]
        faces = model.mesh_face[face_adr:face_adr + num_face]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - frame_pos) @ frame_mat
        triangles.append(local[faces][:, :, :2])
    if not triangles:
        return []
    return silhouette(np.concatenate(triangles))


def tap_label(
        kind: str,
        key: str,
        xy: np.ndarray
        ) -> str:
    side = "sinistral" if xy[1] > 0 else "dextral"
    if kind == "intermediate":
        return f"hole {key}"
    if kind == "ghost":
        return f"midline hole {key}"
    if kind == "end":
        return "corner anchor"
    return f"MVM hole ({side})"


def build_catalog(
        xml_path: Path,
        num_segments: int,
        mesh_files: List[str]
        ) -> Dict:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Group sites into taps: {(segment, plate): {tap_id: {"kind", "key", "sites": [(sub, site_id)]}}}
    taps: Dict = {}
    for site in range(model.nsite):
        name = model.site(site).name
        for kind, pattern in TAP_PATTERNS:
            match = pattern.match(name)
            if match is None:
                continue
            groups = match.groups()
            tap_id, segment, plate = groups[0], int(groups[1]), int(groups[2])
            key = groups[3] if kind != "end" else ""
            entry = taps.setdefault((segment, plate), {}).setdefault(
                    tap_id, {"kind": kind, "key": key, "sites": []}
                    )
            entry["sites"].append(site)
            break

    segments = []
    for segment in range(num_segments):
        frame = model.body(f"segment_{segment}_vertebrae").id
        frame_pos = data.xpos[frame].copy()
        frame_mat = data.xmat[frame].reshape(3, 3).copy()

        plate_bodies = {plate: model.body(f"segment_{segment}_plate_{plate}").id for plate in range(4)}
        plate_subtrees = {plate: set(_body_subtree(model, body)) for plate, body in plate_bodies.items()}
        all_plate_bodies = set().union(*plate_subtrees.values())
        vertebra_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in _body_subtree(model, frame)
                          and model.geom_bodyid[g] not in all_plate_bodies
                          and not _belongs_to_later_segment(model, model.geom_bodyid[g], segment)]

        plates = []
        for plate, body in plate_bodies.items():
            # Plates must have an identity rest pose in the segment frame (DESIGN.md relies on it).
            relative_pos = (data.xpos[body] - frame_pos) @ frame_mat
            relative_mat = frame_mat.T @ data.xmat[body].reshape(3, 3)
            assert np.allclose(relative_pos, 0, atol=1e-9) and np.allclose(relative_mat, np.eye(3), atol=1e-9), \
                f"plate body segment_{segment}_plate_{plate} is not at identity in its segment frame"

            plate_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in plate_subtrees[plate]
                           and "plate_body" in model.body(model.geom_bodyid[g]).name]

            plate_taps = []
            free_site_z = None
            for tap_id, entry in sorted(taps.get((segment, plate), {}).items(), key=_tap_sort_key):
                site_ids = entry["sites"]
                local = np.array([(data.site_xpos[s] - frame_pos) @ frame_mat for s in site_ids])
                order = np.argsort(local[:, 2])
                xy = local[:, :2].mean(axis=0)
                if entry["kind"] == "intermediate" and free_site_z is None:
                    assert all(model.site_bodyid[s] == body for s in site_ids)
                    free_site_z = sorted(float(model.site_pos[s][2]) for s in site_ids)
                plate_taps.append(
                        {
                                "id": tap_id,
                                "kind": entry["kind"],
                                "label": tap_label(entry["kind"], entry["key"], xy),
                                "xy": np.round(xy, 6).tolist(),
                                "sites": [model.site(site_ids[i]).name for i in order]
                                }
                        )

            plates.append(
                    {
                            "index": plate,
                            "corner": CORNERS[plate],
                            "body": f"segment_{segment}_plate_{plate}",
                            "outline": geoms_outline(model, data, plate_geoms, frame_pos, frame_mat),
                            "free_site_z": [round(z, 6) for z in free_site_z],
                            "taps": plate_taps
                            }
                    )

        segments.append(
                {
                        "index": segment,
                        "frame_body": f"segment_{segment}_vertebrae",
                        "vertebra_outline": geoms_outline(model, data, vertebra_geoms, frame_pos, frame_mat),
                        "plates": plates
                        }
                )

    def joint_defaults(
            axis: str
            ) -> Dict[str, float]:
        joint = model.joint(f"segment_1_vertebrae_vertebrae_joint_{axis}")
        return {
                f"{axis}_stiffness": float(model.jnt_stiffness[joint.id]),
                f"{axis}_damping": float(model.dof_damping[model.jnt_dofadr[joint.id]]),
                f"{axis}_range_deg": float(np.degrees(model.jnt_range[joint.id][1]))
                }

    glide = model.joint("segment_1_plate_0_x_axis")
    strut = model.tendon("segment_0_vertebral_vertebral_strut_ventral")
    defaults = {
            "vertebra": {**joint_defaults("pitch"), **joint_defaults("roll"), **joint_defaults("yaw")},
            "plate_glide": {
                    "stiffness": float(model.jnt_stiffness[glide.id]),
                    "damping": float(model.dof_damping[model.jnt_dofadr[glide.id]])
                    },
            "strut": {"stiffness": float(model.tendon_stiffness[strut.id]),
                      "damping": float(model.tendon_damping[strut.id])},
            "timestep": float(model.opt.timestep)
            }

    return {
            "format": "seahorse-catalog",
            "version": 1,
            "num_segments": num_segments,
            "segment_spacing": float(model.body("segment_1").pos[2]),
            "corners": CORNERS,
            "axes": {"ventral": [1, 0], "dorsal": [-1, 0], "sinistral": [0, 1], "dextral": [0, -1]},
            "mesh_files": mesh_files,
            "defaults": defaults,
            "segments": segments
            }


def _belongs_to_later_segment(
        model: mujoco.MjModel,
        body: int,
        segment: int
        ) -> bool:
    """True if the body is (inside) a deeper segment than ``segment``."""
    while body != 0:
        match = re.match(r"^segment_(\d+)$", model.body(body).name)
        if match is not None:
            return int(match.group(1)) != segment
        body = model.body_parentid[body]
    return False


def _tap_sort_key(
        item
        ):
    tap_id, entry = item
    order = {"end": 0, "ghost": 1, "mvm": 2, "intermediate": 3}[entry["kind"]]
    key = entry["key"]
    return order, int(key) if key.isdigit() else 0, key


def deduplicate_meshes(
        xml_path: Path
        ) -> int:
    """Merge <mesh> assets that load the same file with the same attributes.

    dm_control writes one mesh asset per geom (121) for only 14 files; MuJoCo then
    compiles every copy separately, which costs ~250 MB in the browser. Returns
    the number of remaining mesh assets.
    """
    from lxml import etree

    tree = etree.parse(str(xml_path))
    canonical: Dict[tuple, str] = {}
    renamed: Dict[str, str] = {}
    for mesh in tree.iter("mesh"):
        key = tuple(sorted((k, v) for k, v in mesh.attrib.items() if k != "name"))
        name = mesh.get("name")
        if key in canonical:
            renamed[name] = canonical[key]
            mesh.getparent().remove(mesh)
        else:
            canonical[key] = name
    for geom in tree.iter("geom"):
        if geom.get("mesh") in renamed:
            geom.set("mesh", renamed[geom.get("mesh")])
    tree.write(str(xml_path), xml_declaration=False, encoding="utf-8")
    return len(canonical)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Tendon Designer base model and catalogue.")
    parser.add_argument("--num-segments", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=WEB_MODEL_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mjcf_model = build_base_mjcf(num_segments=args.num_segments)
    mjcf.export_with_assets(mjcf_model, str(args.out_dir), out_file_name="base.xml")

    xml_path = args.out_dir / "base.xml"
    deduplicate_meshes(xml_path)
    mesh_files = sorted(set(re.findall(r'file="([^"]+)"', xml_path.read_text())))
    catalog = build_catalog(xml_path=xml_path, num_segments=args.num_segments, mesh_files=mesh_files)
    (args.out_dir / "catalog.json").write_text(json.dumps(catalog, indent=1))

    num_taps = sum(len(p["taps"]) for s in catalog["segments"] for p in s["plates"])
    print(f"Exported {xml_path} ({len(mesh_files)} mesh files) and catalog.json ({num_taps} taps)")


if __name__ == "__main__":
    main()

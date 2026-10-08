"""Seahorse muscle playground configurations in Python: load, validate and apply them to the MJCF model.

This is the Python half of the "config -> MJCF" contract in ``DESIGN.md``. It produces the
same sites, pulleys, actuators, sensors and parameter overrides as ``web/js/model_builder.js``
(``buildModelXml``), so a configuration JSON saved by a student in the browser can be
simulated here with exactly the same model.

Typical use::

    from seahorse_muscle_architecture.silico.tendon_designer import config as td

    cfg = td.load_config("student_config.json")
    mjcf_model = td.build_model(cfg)                 # dm_control mjcf.RootElement
    model = td.compile_model(mjcf_model)             # mujoco.MjModel
    data = mujoco.MjData(model)
    for tendon in cfg["tendons"]:
        td.set_activation(model, data, tendon, 0.5)  # a in [0, 1] -> ctrl
"""
from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import mujoco
import numpy as np
from dm_control import mjcf

TENDON_DESIGNER_DIR = Path(__file__).parent
WEB_MODEL_DIR = TENDON_DESIGNER_DIR / "web" / "model"
CATALOG_PATH = WEB_MODEL_DIR / "catalog.json"

CONFIG_FORMAT = "seahorse-tendon-config"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
ACTUATOR_TYPES = ("motor", "position", "none")
AXES = ("pitch", "roll", "yaw")

# Defaults used when a field is missing. They match buildModelXml() in model_builder.js
# (not the UI defaults of config_tools.js; the UI always writes these fields explicitly).
DEFAULT_WIDTH = 0.001
DEFAULT_DAMPING = 0.0
DEFAULT_STIFFNESS = 0.0
DEFAULT_MOTOR_MAX_FORCE = 10.0
DEFAULT_POSITION_MAX_FORCE = 1000.0
DEFAULT_POSITION_KP = 1000.0
DEFAULT_MAX_STRAIN = 0.26

FREE_SITE_SIZE = 0.0005
FREE_SITE_RGBA = (0.95, 0.6, 0.1, 1.0)
HANGING_EULER = (math.pi, 0.0, 0.0)

_GLIDE_JOINT = re.compile(r"^segment_\d+_plate_\d+_[xy]_axis$")
_STRUT_TENDON = re.compile(r"^segment_\d+_vertebral_vertebral_strut_")

Point = Dict[str, Any]
SiteItem = Dict[str, Any]  # {"site": name} or {"pulley": 1}


class ConfigError(ValueError):
    """Raised when a configuration cannot be turned into a model. ``errors`` lists all problems."""

    def __init__(
            self,
            errors: Union[str, Sequence[str]]
            ) -> None:
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("Invalid tendon configuration:\n  - " + "\n  - ".join(self.errors))


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def load_config(
        path: Union[str, Path]
        ) -> Dict:
    """Read a configuration JSON file (as saved by the web app)."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: not a JSON object")
    if config.get("format", CONFIG_FORMAT) != CONFIG_FORMAT:
        raise ConfigError(f"{path}: format is {config.get('format')!r}, expected {CONFIG_FORMAT!r}")
    return config


_catalog_cache: Dict[int, Dict] = {}


def load_catalog(
        num_segments: int = 11,
        path: Optional[Union[str, Path]] = None
        ) -> Dict:
    """The site catalogue (``catalog.json``).

    Uses ``web/model/catalog.json`` (or ``path``) when its segment count matches; otherwise
    builds a catalogue for ``num_segments`` with ``export_assets`` in a temporary directory.
    """
    if path is not None:
        return json.loads(Path(path).read_text())
    if num_segments in _catalog_cache:
        return _catalog_cache[num_segments]
    catalog = None
    if CATALOG_PATH.exists():
        catalog = json.loads(CATALOG_PATH.read_text())
        if catalog.get("num_segments") != num_segments:
            catalog = None
    if catalog is None:
        from seahorse_muscle_architecture.silico.tendon_designer import export_assets
        with tempfile.TemporaryDirectory() as tmp:
            base = export_assets.build_base_mjcf(num_segments=num_segments)
            mjcf.export_with_assets(base, tmp, out_file_name="base.xml")
            xml_path = Path(tmp) / "base.xml"
            mesh_files = sorted(set(re.findall(r'file="([^"]+)"', xml_path.read_text())))
            catalog = export_assets.build_catalog(xml_path=xml_path, num_segments=num_segments,
                                                  mesh_files=mesh_files)
    _catalog_cache[num_segments] = catalog
    return catalog


# ---------------------------------------------------------------------------------------------
# Catalogue lookups
# ---------------------------------------------------------------------------------------------

_tap_index_cache: Dict[int, Any] = {}


def catalog_tap_index(
        catalog: Dict
        ) -> Dict[str, Point]:
    """Map tap id -> {id, segment, plate, kind, sites, xy, body}."""
    cached = _tap_index_cache.get(id(catalog))
    if cached is not None and cached[0] is catalog:
        return cached[1]
    index = {}
    for segment in catalog["segments"]:
        for plate in segment["plates"]:
            for tap in plate["taps"]:
                index[tap["id"]] = {
                        "id": tap["id"], "segment": segment["index"], "plate": plate["index"],
                        "kind": tap["kind"], "sites": list(tap["sites"]), "xy": tap["xy"], "body": plate["body"]
                        }
    _tap_index_cache[id(catalog)] = (catalog, index)
    return index


def find_plate(
        catalog: Dict,
        segment: int,
        plate: int
        ) -> Optional[Dict]:
    for seg in catalog["segments"]:
        if seg["index"] == segment:
            for p in seg["plates"]:
                if p["index"] == plate:
                    return p
    return None


def resolve_point(
        point_id: str,
        catalog: Dict,
        free_points: Optional[Sequence[Dict]],
        tendon_name: str = "?"
        ) -> Point:
    """Point id (catalog tap id or free point id) -> {id, segment, sites}."""
    tap = catalog_tap_index(catalog).get(point_id)
    if tap is not None:
        return {"id": point_id, "segment": tap["segment"], "sites": list(tap["sites"])}
    for fp in free_points or []:
        if fp.get("id") == point_id:
            return {"id": point_id, "segment": int(fp["segment"]), "sites": [f"{point_id}_0", f"{point_id}_1"]}
    raise ConfigError(f'Tendon "{tendon_name}": unknown point "{point_id}" (not a catalog tap and not a free point)')


def _sign_or_plus(
        x: float
        ) -> int:
    return -1 if x < 0 else 1


def _is_index(
        value: Any
        ) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------------------------
# Site expansion (DESIGN.md "Site expansion rules")
# ---------------------------------------------------------------------------------------------

def expand_tendon_sites(
        tendon: Dict,
        catalog: Dict,
        free_points: Optional[Sequence[Dict]] = None
        ) -> List[SiteItem]:
    """Ordered spatial-tendon path of a tendon: a list of ``{"site": name}`` and ``{"pulley": 1}``.

    1. ``S(P)`` = ordered sites of point P (catalog order = proximal -> distal).
    2. Trunk direction ``D = sign(seg(last) - seg(first))`` (+1 if 0); each trunk point gives
       ``S(P)`` (D > 0) or ``reverse(S(P))``.
    3. Branch ``{from: k, path: [Q1..Qm]}``: ``Db = sign(seg(Qm) - seg(P_k))`` (+1 if 0); emit a
       pulley, the exit site of ``P_k`` (last of ``S(P_k)`` ordered by Db), then every ``S(Qj)``
       ordered by Db.
    4. A site equal to the immediately preceding emitted site (same branch) is dropped.
    5. Branches start from trunk points only.
    """
    name = tendon.get("name", "?")
    path = tendon.get("path") or []
    if len(path) < 2:
        raise ConfigError(f'Tendon "{name}": the path needs at least 2 points (has {len(path)})')
    trunk = [resolve_point(pid, catalog, free_points, name) for pid in path]

    out: List[SiteItem] = []
    previous: List[Optional[str]] = [None]

    def emit(
            site: str
            ) -> None:
        if site == previous[0]:
            return
        out.append({"site": site})
        previous[0] = site

    def ordered(
            sites: List[str],
            direction: int
            ) -> List[str]:
        return sites if direction > 0 else sites[::-1]

    direction = _sign_or_plus(trunk[-1]["segment"] - trunk[0]["segment"])
    for point in trunk:
        for site in ordered(point["sites"], direction):
            emit(site)

    for b, branch in enumerate(tendon.get("branches") or []):
        k = branch.get("from")
        if not _is_index(k) or k < 0 or k >= len(trunk):
            raise ConfigError(f'Tendon "{name}", branch {b + 1}: "from" must be a trunk point index '
                              f'0..{len(trunk) - 1} (got {k!r})')
        branch_path = branch.get("path") or []
        if len(branch_path) < 1:
            raise ConfigError(f'Tendon "{name}", branch {b + 1}: the branch path needs at least 1 point')
        points = [resolve_point(pid, catalog, free_points, name) for pid in branch_path]
        split = trunk[k]
        branch_direction = _sign_or_plus(points[-1]["segment"] - split["segment"])
        out.append({"pulley": 1})
        previous[0] = None  # rule 4 applies within a branch
        emit(ordered(split["sites"], branch_direction)[-1])
        for point in points:
            for site in ordered(point["sites"], branch_direction):
                emit(site)
    return out


def split_branches(
        items: Sequence[SiteItem]
        ) -> List[List[str]]:
    """Split an expanded site list at its pulleys: [[trunk sites], [branch 1 sites], ...]."""
    branches: List[List[str]] = [[]]
    for item in items:
        if "pulley" in item:
            branches.append([])
        else:
            branches[-1].append(item["site"])
    return branches


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

def _is_number(
        value: Any
        ) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_config(
        config: Dict,
        catalog: Dict,
        warnings: Optional[List[str]] = None
        ) -> List[str]:
    """All problems that prevent building a model (empty list = valid).

    Non-blocking remarks (e.g. a tendon that crosses no joint) are appended to ``warnings``
    when a list is given.
    """
    errors: List[str] = []
    warnings = warnings if warnings is not None else []
    if not isinstance(config, dict):
        return ["the configuration is not a JSON object"]
    if config.get("format", CONFIG_FORMAT) != CONFIG_FORMAT:
        errors.append(f'format is {config.get("format")!r}, expected {CONFIG_FORMAT!r}')

    num_segments = catalog["num_segments"]
    taps = catalog_tap_index(catalog)
    free_points = config.get("free_points") or []
    if not isinstance(free_points, list):
        errors.append("free_points must be a list")
        free_points = []

    free_segments: Dict[str, int] = {}
    for fp in free_points:
        fid = fp.get("id") if isinstance(fp, dict) else None
        if not isinstance(fid, str) or not NAME_PATTERN.match(fid):
            errors.append(f"free point id {fid!r} is not a valid name (letters, digits, _ and - only)")
            continue
        if fid in free_segments or fid in taps:
            errors.append(f'free point id "{fid}" is used twice')
            continue
        segment, plate = fp.get("segment"), fp.get("plate")
        if not _is_index(segment) or not _is_index(plate) or find_plate(catalog, segment, plate) is None:
            errors.append(f'free point "{fid}": no plate {plate!r} in segment {segment!r} '
                          f'(segments 0..{num_segments - 1}, plates 0..3)')
            continue
        xy = fp.get("xy")
        if not isinstance(xy, (list, tuple)) or len(xy) != 2 or not all(_is_number(v) for v in xy):
            errors.append(f'free point "{fid}": xy must be two numbers [x, y] in metres')
            continue
        free_segments[fid] = segment

    def segment_of(
            pid: Any
            ) -> Optional[int]:
        if isinstance(pid, str) and pid in taps:
            return taps[pid]["segment"]
        return free_segments.get(pid) if isinstance(pid, str) else None

    tendons = config.get("tendons") or []
    if not isinstance(tendons, list):
        errors.append("tendons must be a list")
        tendons = []
    seen = set()
    for t, tendon in enumerate(tendons):
        if not isinstance(tendon, dict):
            errors.append(f"tendon {t + 1} is not an object")
            continue
        name = tendon.get("name")
        label = f'tendon "{name}"'
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            errors.append(f'{label}: names may only contain letters, digits, "_" and "-"')
        elif name.startswith("segment_"):
            errors.append(f'{label}: names must not start with "segment_"')
        elif name in seen:
            errors.append(f"{label}: the name is used twice")
        seen.add(name)

        path = tendon.get("path") or []
        if not isinstance(path, list) or len(path) < 2:
            errors.append(f"{label}: the path needs at least 2 points (a start and an end)")
            path = path if isinstance(path, list) else []
        segments = []
        for i, pid in enumerate(path):
            segment = segment_of(pid)
            if segment is None:
                errors.append(f'{label}: point {i + 1} ("{pid}") does not exist')
            else:
                segments.append(segment)
            if i > 0 and path[i - 1] == pid:
                warnings.append(f"{label}: point {i + 1} is the same as the point before it")
        branches = tendon.get("branches") or []
        if not isinstance(branches, list):
            errors.append(f"{label}: branches must be a list")
            branches = []
        for b, branch in enumerate(branches):
            k = branch.get("from") if isinstance(branch, dict) else None
            if not _is_index(k) or not 0 <= k < len(path):
                errors.append(f'{label}, branch {b + 1}: "from" must be a trunk point index '
                              f'0..{len(path) - 1} (got {k!r})')
            branch_path = branch.get("path") if isinstance(branch, dict) else None
            if not isinstance(branch_path, list) or len(branch_path) < 1:
                errors.append(f"{label}, branch {b + 1}: the branch path needs at least 1 point")
                continue
            for pid in branch_path:
                segment = segment_of(pid)
                if segment is None:
                    errors.append(f'{label}, branch {b + 1}: point "{pid}" does not exist')
                else:
                    segments.append(segment)
        if len(path) >= 2 and segments and min(segments) == max(segments):
            warnings.append(f"{label} starts and ends on the same segment, so it crosses no joint")

        actuator = tendon.get("actuator") or {"type": "none"}
        kind = actuator.get("type")
        if kind not in ACTUATOR_TYPES:
            errors.append(f"{label}: unknown actuator type {kind!r} (motor, position or none)")
        for key in ("max_force", "kp"):
            if actuator.get(key) is not None and not (_is_number(actuator[key]) and actuator[key] > 0):
                errors.append(f"{label}: actuator {key} must be a number > 0")
        if kind == "position" and actuator.get("max_strain") is not None:
            strain = actuator["max_strain"]
            if not (_is_number(strain) and 0 < strain < 1):
                errors.append(f"{label}: actuator max_strain must be between 0 and 1")
        for key in ("width", "damping", "stiffness"):
            value = tendon.get(key)
            if value is not None and not (_is_number(value) and value >= 0):
                errors.append(f"{label}: {key} must be a number >= 0")

    errors.extend(_validate_params(config.get("params"), warnings))
    return errors


_PARAM_KEYS = {
        "vertebra": {f"{axis}_{q}" for axis in AXES for q in ("stiffness", "damping", "range_deg")},
        "plate_glide": {"stiffness", "damping"},
        "strut": {"stiffness", "damping"},
        }


def _validate_params(
        params: Any,
        warnings: List[str]
        ) -> List[str]:
    errors: List[str] = []
    if params is None:
        return errors
    if not isinstance(params, dict):
        return ["params must be an object"]
    for key, value in params.items():
        if value is None:
            continue
        if key in _PARAM_KEYS:
            if not isinstance(value, dict):
                errors.append(f"params.{key} must be an object")
                continue
            for sub, sub_value in value.items():
                if sub not in _PARAM_KEYS[key]:
                    warnings.append(f"params.{key}.{sub} is not a known parameter (ignored)")
                elif sub_value is not None and not (_is_number(sub_value) and sub_value >= 0):
                    errors.append(f"params.{key}.{sub} must be a number >= 0")
        elif key == "stiffness_taper":
            if not (_is_number(value) and value >= 0):
                errors.append("params.stiffness_taper must be a number >= 0")
        elif key == "orientation":
            if value not in ("upright", "hanging"):
                errors.append(f'params.orientation must be "upright" or "hanging" (got {value!r})')
        elif key == "gravity":
            if not isinstance(value, bool):
                errors.append("params.gravity must be true or false")
        elif key == "timestep":
            if not (_is_number(value) and value > 0):
                errors.append("params.timestep must be a number > 0")
        else:
            warnings.append(f"params.{key} is not a known parameter (ignored)")
    return errors


# ---------------------------------------------------------------------------------------------
# MJCF generation
# ---------------------------------------------------------------------------------------------

def _num(
        x: Any
        ) -> float:
    """Same rounding as ``num()`` in model_builder.js (12 significant digits)."""
    return float(f"{float(x):.12g}")


def color_to_rgba(
        color: Any
        ) -> List[float]:
    """'#rgb' | '#rrggbb' | '#rrggbbaa' -> [r, g, b, a] in 0..1 (grey if unknown), like colorToRgba()."""
    hex_str = str(color or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{3}", hex_str):
        hex_str = "".join(c + c for c in hex_str)
    if not re.fullmatch(r"[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", hex_str):
        return [0.5, 0.5, 0.5, 1.0]
    values = [int(hex_str[i:i + 2], 16) / 255 for i in range(0, len(hex_str), 2)]
    if len(values) == 3:
        values.append(1.0)
    return [round(v, 4) for v in values]


def _actuator_settings(
        tendon: Dict
        ) -> Dict:
    actuator = tendon.get("actuator") or {"type": "none"}
    kind = actuator.get("type", "none")
    if kind == "motor":
        max_force = actuator.get("max_force")
        return {"type": "motor", "max_force": float(DEFAULT_MOTOR_MAX_FORCE if max_force is None else max_force)}
    if kind == "position":
        max_force, kp, strain = actuator.get("max_force"), actuator.get("kp"), actuator.get("max_strain")
        return {"type": "position",
                "max_force": float(DEFAULT_POSITION_MAX_FORCE if max_force is None else max_force),
                "kp": float(DEFAULT_POSITION_KP if kp is None else kp),
                "max_strain": float(DEFAULT_MAX_STRAIN if strain is None else strain)}
    return {"type": kind}


def apply_tendon_config(
        mjcf_model: mjcf.RootElement,
        config: Dict,
        catalog: Dict
        ) -> mjcf.RootElement:
    """Add the configuration's free points, tendons, actuators and sensors to ``mjcf_model`` and
    apply its ``params`` overrides. Modifies ``mjcf_model`` in place and returns it.

    ``mjcf_model`` must be the base morphology as built by ``export_assets.build_base_mjcf``
    (or ``mjcf.from_path(web/model/base.xml)``). Raises ``ConfigError`` on invalid configs.
    """
    errors = validate_config(config, catalog)
    if errors:
        raise ConfigError(errors)
    free_points = config.get("free_points") or []

    # --- free points: two sites on the plate body -----------------------------------------
    for fp in free_points:
        plate = find_plate(catalog, fp["segment"], fp["plate"])
        body = mjcf_model.find("body", plate["body"])
        if body is None:
            raise ConfigError(f'free point "{fp["id"]}": body "{plate["body"]}" is not in the model')
        x, y = fp["xy"]
        for k, z in enumerate(plate["free_site_z"]):
            site_name = f'{fp["id"]}_{k}'
            if mjcf_model.find("site", site_name) is not None:
                raise ConfigError(f'free point "{fp["id"]}": site "{site_name}" already exists')
            body.add("site", name=site_name, type="sphere", size=[FREE_SITE_SIZE], rgba=list(FREE_SITE_RGBA),
                     pos=[_num(x), _num(y), _num(z)])

    # --- tendons, actuators, sensors --------------------------------------------------------
    for tendon in config.get("tendons") or []:
        name = tendon["name"]
        if mjcf_model.find("tendon", name) is not None:
            raise ConfigError(f'tendon name "{name}" already exists in the model')
        items = expand_tendon_sites(tendon, catalog, free_points)
        for item in items:
            if "site" in item and mjcf_model.find("site", item["site"]) is None:
                raise ConfigError(f'tendon "{name}": site "{item["site"]}" does not exist in the model')

        # springlength is left at MuJoCo's default "-1 -1" (= length at qpos0); dm_control's schema
        # only knows a scalar springlength, the JS builder writes "-1 -1" explicitly.
        spatial = mjcf_model.tendon.add(
                "spatial", name=name,
                width=_num(DEFAULT_WIDTH if tendon.get("width") is None else tendon["width"]),
                rgba=color_to_rgba(tendon.get("color")),
                damping=_num(DEFAULT_DAMPING if tendon.get("damping") is None else tendon["damping"]),
                stiffness=_num(DEFAULT_STIFFNESS if tendon.get("stiffness") is None else tendon["stiffness"]),
                )
        for item in items:
            if "site" in item:
                spatial.add("site", site=item["site"])
            else:
                spatial.add("pulley", divisor=1)

        settings = _actuator_settings(tendon)
        force = _num(settings.get("max_force", 0.0))
        if settings["type"] == "motor":
            mjcf_model.actuator.add("motor", name=name, tendon=name, gear=[force], ctrllimited="true",
                                    ctrlrange=[-1, 0], forcelimited="true", forcerange=[-force, 0])
        elif settings["type"] == "position":
            mjcf_model.actuator.add("position", name=name, tendon=name, kp=_num(settings["kp"]),
                                    ctrllimited="false", forcelimited="true", forcerange=[-force, 0])

        mjcf_model.sensor.add("tendonpos", name=f"{name}_length", tendon=name)
        if settings["type"] in ("motor", "position"):
            mjcf_model.sensor.add("actuatorfrc", name=f"{name}_force", actuator=name)

    apply_params(mjcf_model, config.get("params"), catalog)
    return mjcf_model


def apply_params(
        mjcf_model: mjcf.RootElement,
        params: Optional[Dict],
        catalog: Dict
        ) -> None:
    """Apply ``params`` overrides as explicit attributes (DESIGN.md "params overrides")."""
    if not params:
        return
    num_segments = catalog["num_segments"]
    defaults = catalog.get("defaults", {}).get("vertebra", {})

    # --- vertebral joints ---------------------------------------------------------------------
    vertebra = params.get("vertebra") or {}
    taper = params.get("stiffness_taper")
    taper = 1.0 if taper is None else float(taper)
    for i in range(1, num_segments):
        factor = 1 + (taper - 1) * (i - 1) / (num_segments - 2) if num_segments > 2 else taper
        for axis in AXES:
            joint = mjcf_model.find("joint", f"segment_{i}_vertebrae_vertebrae_joint_{axis}")
            if joint is None:
                continue
            stiffness = vertebra.get(f"{axis}_stiffness")
            if stiffness is not None or taper != 1:
                if stiffness is not None:
                    base = float(stiffness)
                elif joint.stiffness is not None:
                    base = float(joint.stiffness)
                else:
                    base = float(defaults.get(f"{axis}_stiffness", 0.0))
                joint.stiffness = _num(base * factor)
            damping = vertebra.get(f"{axis}_damping")
            if damping is not None:
                joint.damping = _num(damping)
            range_deg = vertebra.get(f"{axis}_range_deg")
            if range_deg is not None:
                rad = abs(float(range_deg)) * math.pi / 180
                joint.range = [_num(-rad), _num(rad)]
                joint.limited = "true"

    # --- plate glide joints & struts ---------------------------------------------------------
    glide = params.get("plate_glide") or {}
    strut = params.get("strut") or {}
    for joint in mjcf_model.find_all("joint"):
        if joint.name and _GLIDE_JOINT.match(joint.name):
            if glide.get("stiffness") is not None:
                joint.stiffness = _num(glide["stiffness"])
            if glide.get("damping") is not None:
                joint.damping = _num(glide["damping"])
    for tendon in mjcf_model.find_all("tendon"):
        if tendon.tag == "spatial" and tendon.name and _STRUT_TENDON.match(tendon.name):
            if strut.get("stiffness") is not None:
                tendon.stiffness = _num(strut["stiffness"])
            if strut.get("damping") is not None:
                tendon.damping = _num(strut["damping"])

    # --- orientation --------------------------------------------------------------------------
    orientation = params.get("orientation")
    if orientation is not None:
        if orientation not in ("upright", "hanging"):
            raise ConfigError(f'params.orientation must be "upright" or "hanging" (got {orientation!r})')
        body = mjcf_model.find("body", "segment_0")
        if body is None:
            raise ConfigError('params.orientation: body "segment_0" is not in the model')
        for attribute in ("quat", "axisangle", "xyaxes", "zaxis"):
            setattr(body, attribute, None)
        body.euler = list(HANGING_EULER) if orientation == "hanging" else [0.0, 0.0, 0.0]

    # --- option -------------------------------------------------------------------------------
    if params.get("timestep") is not None:
        mjcf_model.option.timestep = _num(params["timestep"])
    if params.get("gravity") is not None:
        mjcf_model.option.flag.gravity = "enable" if params["gravity"] else "disable"


# ---------------------------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------------------------

def build_model(
        config: Dict,
        num_segments: int = 11,
        catalog: Optional[Dict] = None
        ) -> mjcf.RootElement:
    """Base morphology (``export_assets.build_base_mjcf``) + ``apply_tendon_config``."""
    from seahorse_muscle_architecture.silico.tendon_designer.export_assets import build_base_mjcf
    catalog = catalog if catalog is not None else load_catalog(num_segments)
    mjcf_model = build_base_mjcf(num_segments=num_segments)
    if config.get("name"):
        mjcf_model.model = re.sub(r"[^A-Za-z0-9_-]+", "_", str(config["name"])).strip("_") or mjcf_model.model
    return apply_tendon_config(mjcf_model, config, catalog)


def compile_model(
        mjcf_model: mjcf.RootElement
        ) -> mujoco.MjModel:
    """Compile a dm_control MJCF model with plain MuJoCo (no name prefixes)."""
    return mujoco.MjModel.from_xml_string(mjcf_model.to_xml_string(), mjcf_model.get_assets())


def tendon_ids(
        model: mujoco.MjModel,
        tendon_cfg: Dict
        ) -> Dict[str, Optional[int]]:
    """{"tendon": id, "actuator": id | None} of a config tendon in a compiled model."""
    name = tendon_cfg["name"]
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return {"tendon": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name),
            "actuator": actuator if actuator >= 0 else None}


def activation_to_ctrl(
        model: mujoco.MjModel,
        tendon_cfg: Dict,
        activation: float
        ) -> Optional[float]:
    """Activation ``a`` in [0, 1] -> actuator ctrl (None for a passive tendon).

    motor: ``ctrl = -a``; position: ``ctrl = length0 * (1 - a * max_strain)``.
    """
    a = float(np.clip(activation, 0.0, 1.0))
    settings = _actuator_settings(tendon_cfg)
    if settings["type"] == "motor":
        return -a
    if settings["type"] == "position":
        tendon = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_cfg["name"])
        return float(model.tendon_length0[tendon]) * (1.0 - a * settings["max_strain"])
    return None


def set_activation(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tendon_cfg: Dict,
        activation: float
        ) -> None:
    """Write ``activation_to_ctrl`` into ``data.ctrl`` (no-op for passive tendons)."""
    ctrl = activation_to_ctrl(model, tendon_cfg, activation)
    if ctrl is not None:
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, tendon_cfg["name"])] = ctrl


def set_activations(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Dict,
        activations: Dict[str, float]
        ) -> None:
    """Set every actuated tendon; tendons missing from ``activations`` get a = 0.

    Note that a = 0 is *not* ctrl = 0 for position actuators (it is ctrl = length0).
    """
    for tendon in config.get("tendons") or []:
        set_activation(model, data, tendon, activations.get(tendon["name"], 0.0))


def make_environment(
        config: Dict,
        num_segments: int = 11,
        environment_configuration=None
        ):
    """A ``SeahorseMJCEnvironment`` (moojoco) for the configuration.

    The model is attached to the ``EmptyArena`` (cameras, lights) at identity, so the tail
    orientation is the one set by ``params.orientation`` (the arena used by the existing
    experiments hangs the tail; use ``"orientation": "hanging"`` to match it). Because of the
    attachment, every element name in the compiled env model gets a prefix
    (``"<model name>/"``); the env's action vector follows the actuator order of the config
    (motor ctrl in [-1, 0]; position ctrl in metres, use ``activation_to_ctrl``).
    """
    from seahorse_muscle_architecture.silico.seahorse.environment.mjc_env import \
        SeahorseEnvironmentConfiguration, SeahorseMJCEnvironment
    from seahorse_muscle_architecture.silico.seahorse.mjcf.arena.empty_arena import EmptyArena, \
        EmptyArenaConfiguration

    mjcf_model = build_model(config, num_segments=num_segments)
    arena = EmptyArena(EmptyArenaConfiguration("tendon_designer_arena"))
    arena.mjcf_model.compiler.angle = "radian"
    arena.mjcf_model.option.flag.contact = "disable"
    arena.mjcf_model.option.flag.gravity = mjcf_model.option.flag.gravity or "disable"
    if mjcf_model.option.timestep is not None:
        arena.mjcf_model.option.timestep = mjcf_model.option.timestep
    arena.mjcf_body.add("site", name="tendon_designer_attachment", pos=[0, 0, 0]).attach(mjcf_model)
    if environment_configuration is None:
        environment_configuration = SeahorseEnvironmentConfiguration(render_mode="rgb_array")
    return SeahorseMJCEnvironment(mjcf_str=arena.get_mjcf_str(), mjcf_assets=arena.get_mjcf_assets(),
                                  configuration=environment_configuration)

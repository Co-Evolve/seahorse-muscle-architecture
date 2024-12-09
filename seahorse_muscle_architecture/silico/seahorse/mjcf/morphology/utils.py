from typing import List, Tuple

import numpy as np
from dm_control import mjcf
from moojoco.mjcf.morphology import MJCFMorphologyPart

from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import PLATE_INDEX_TO_SIDE
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import (MeshSpecification, \
                                                                                                      SeahorsePlateSpecification)


def is_inner_plate_x_axis(
        segment_index: int,
        plate_index: int
        ) -> bool:
    if plate_index == 0:
        return False
    if plate_index == 1:
        return True
    if segment_index % 2 == 0:
        if plate_index == 2:
            return True
        else:
            return False
    else:
        if plate_index == 2:
            return False
        else:
            return True


def is_inner_plate_y_axis(
        segment_index: int,
        plate_index: int
        ) -> bool:
    if plate_index < 2:
        return False
    else:
        return True


def get_plate_indices_per_side(
        side: str
        ) -> Tuple[int, int]:
    if side == "ventral":
        return (0, 1)
    elif side == "dextral":
        return (0, 3)
    elif side == "sinistral":
        return (1, 2)
    elif side == "dorsal":
        return (2, 3)


def get_inner_and_outer_plate_indices_per_side(
        segment_index: int,
        side: str
        ) -> Tuple[int, int]:
    plates = get_plate_indices_per_side(side=side)

    if side == "ventral" or side == "dorsal":
        is_inner_check = is_inner_plate_x_axis
    else:
        is_inner_check = is_inner_plate_y_axis

    inner_plate, outer_plate = sorted(
            plates,
            key=lambda
                plate_index: not is_inner_check(
                    segment_index=segment_index, plate_index=plate_index
                    )
            )
    return inner_plate, outer_plate


def get_actuator_beam_plate_indices(
        side: str,
        segment_index: str
        ) -> Tuple[int, int]:
    if side == "ventral":
        return 1, 0
    elif side == "dorsal":
        if segment_index % 2 == 0:
            return 2, 3
        else:
            return 3, 2


def plate_index_to_sides(
        plate_index: int
        ) -> Tuple[str, str]:
    return tuple(PLATE_INDEX_TO_SIDE[plate_index].split("_"))


def get_actuator_ghost_taps_index(
        x_side: str,
        y_side: str,
        segment_index: str
        ) -> int:
    if x_side == "ventral":
        if y_side == "dextral":
            return 1
        else:
            return 0
    else:
        if segment_index % 2 == 0:
            if y_side == "dextral":
                return 1
            else:
                return 0
        else:
            if y_side == "dextral":
                return 0
            else:
                return 1


def get_plate_position(
        plate_index: int,
        plate_specification: SeahorsePlateSpecification
        ) -> np.ndarray:
    angles = [5 * np.pi / 4, 7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4]
    angle = angles[plate_index]
    offset_from_vertebrae = plate_specification.offset_from_vertebrae.value
    position = offset_from_vertebrae * np.array([-np.sin(angle), np.cos(angle), 0.0])
    position[2] = plate_specification.depth.value / 2
    return position


def get_all_beam_start_and_stop_segment_indices(
        total_num_segments: int,
        segment_span: int
        ) -> List[Tuple[int, int]]:
    start_and_stop_indices = []

    for segment_index in range(1, total_num_segments):
        start_and_stop_indices.append((segment_index - segment_span, segment_index))

    return start_and_stop_indices


def calculate_relaxed_beam_length(
        morphology_parts: List[MJCFMorphologyPart],
        attachment_points: List[mjcf.Element]
        ) -> float:
    relaxed_beam_length = 0
    for current_index in range(len(attachment_points) - 1):
        next_index = current_index + 1
        current_part = morphology_parts[current_index]
        next_part = morphology_parts[next_index]

        current_attachment_point = attachment_points[current_index]
        next_attachment_point = attachment_points[next_index]

        current_position = current_part.coordinates_of_point_in_root_frame(current_attachment_point.pos)
        next_position = next_part.coordinates_of_point_in_root_frame(next_attachment_point.pos)

        distance_between_points = np.linalg.norm(next_position - current_position)
        relaxed_beam_length += distance_between_points

    return relaxed_beam_length


def add_mesh_to_body(
        body: mjcf.Element,
        name: str,
        mesh_name: str,
        position: np.ndarray,
        euler: np.ndarray,
        rgba: np.ndarray,
        group: int,
        mesh_specification: MeshSpecification
        ) -> mjcf.Element:
    sub_body = body.add(
            'body', name=f"{name}_body", pos=position, euler=euler
            )

    sub_body.add(
            'inertial',
            pos=mesh_specification.center_of_mass.value,
            mass=mesh_specification.mass.value,
            fullinertia=mesh_specification.fullinertia.value
            )

    sub_body.add(
            'geom', name=f"{name}_geom", type="mesh", mesh=mesh_name, rgba=rgba, group=group
            )
    return sub_body

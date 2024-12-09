from __future__ import annotations

from typing import List

import numpy as np
from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class MeshSpecification(Specification):
    def __init__(
            self,
            *,
            mesh_path: str,
            scale: np.ndarray,
            mass: float,
            center_of_mass: np.ndarray,
            fullinertia: np.ndarray
            ) -> None:
        super().__init__()
        self.mesh_path = FixedParameter(mesh_path)
        self.scale_ratio = FixedParameter(scale)
        self.mass = FixedParameter(mass)
        self.center_of_mass = FixedParameter(center_of_mass)
        self.fullinertia = FixedParameter(fullinertia)


class JointSpecification(Specification):
    def __init__(
            self,
            *,
            stiffness: float,
            damping: float,
            friction_loss: float,
            range: float,
            armature: float
            ) -> None:
        super().__init__()
        self.stiffness = FixedParameter(stiffness)
        self.damping = FixedParameter(damping)
        self.friction_loss = FixedParameter(friction_loss)
        self.range = FixedParameter(range)
        self.armature = FixedParameter(armature)


class SeahorsePlateSpecification(Specification):
    def __init__(
            self,
            *,
            plate_mesh_specification: MeshSpecification,
            connector_mesh_specification: MeshSpecification,
            offset_from_vertebrae: float,
            depth: float,
            connector_offset_from_vertebrae: float,
            s_tap_x_offset_from_vertebrae: float,
            s_tap_y_offset_from_vertebrae: float,
            hm_ghost_tap_x_offset_from_plate_origin: float,
            hm_ghost_tap_y_offset_from_plate_origin: float,
            hm_num_intermediate_taps: int,
            hm_y_offset_between_intermediate_taps: float,
            hm_intermediate_first_tap_x_offset_from_plate_origin: float,
            hm_intermediate_first_tap_y_offset_from_plate_origin: float,
            mvm_tap_x_offset: float,
            mvm_tap_z_offset: float,
            x_axis_gliding_joint_specification: JointSpecification,
            y_axis_gliding_joint_specification: JointSpecification, ) -> None:
        super().__init__()
        self.plate_mesh_specification = plate_mesh_specification
        self.connector_mesh_specification = connector_mesh_specification
        self.offset_from_vertebrae = FixedParameter(offset_from_vertebrae)
        self.hm_ghost_tap_x_offset_from_plate_origin = FixedParameter(hm_ghost_tap_x_offset_from_plate_origin)
        self.hm_ghost_tap_y_offset_from_plate_origin = FixedParameter(hm_ghost_tap_y_offset_from_plate_origin)
        self.hm_num_intermediate_taps = FixedParameter(hm_num_intermediate_taps)
        self.hm_y_offset_between_intermediate_taps = FixedParameter(hm_y_offset_between_intermediate_taps)
        self.hm_intermediate_first_tap_x_offset_from_plate_origin = FixedParameter(
                hm_intermediate_first_tap_x_offset_from_plate_origin
                )
        self.hm_intermediate_first_tap_y_offset_from_plate_origin = FixedParameter(
                hm_intermediate_first_tap_y_offset_from_plate_origin
                )
        self.mvm_tap_x_offset = FixedParameter(mvm_tap_x_offset)
        self.mvm_tap_z_offset = FixedParameter(mvm_tap_z_offset)
        self.connector_offset_from_vertebrae = FixedParameter(connector_offset_from_vertebrae)
        self.s_tap_x_offset_from_vertebrae = FixedParameter(s_tap_x_offset_from_vertebrae)
        self.s_tap_y_offset_from_vertebrae = FixedParameter(s_tap_y_offset_from_vertebrae)
        self.x_axis_gliding_joint_specification = x_axis_gliding_joint_specification
        self.y_axis_gliding_joint_specification = y_axis_gliding_joint_specification
        self.depth = FixedParameter(depth)


class SeahorseVertebraeSpecification(Specification):
    def __init__(
            self,
            *,
            z_offset_to_ball_bearing: float,
            offset_to_vertebral_strut_attachment_point: float,
            connector_length: float,
            offset_to_bar_end: float,
            vertebral_mesh_specification: MeshSpecification,
            ball_bearing_mesh_specification: MeshSpecification,
            connector_mesh_specification: MeshSpecification,
            yaw_joint_specification: JointSpecification,
            pitch_joint_specification: JointSpecification,
            roll_joint_specification: JointSpecification
            ) -> None:
        super().__init__()
        self.z_offset_to_ball_bearing = FixedParameter(z_offset_to_ball_bearing)
        self.offset_to_vertebral_strut_attachment_point = FixedParameter(offset_to_vertebral_strut_attachment_point)
        self.offset_to_bar_end = FixedParameter(offset_to_bar_end)
        self.connector_length = FixedParameter(connector_length)
        self.vertebral_mesh_specification = vertebral_mesh_specification
        self.ball_bearing_mesh_specification = ball_bearing_mesh_specification
        self.connector_mesh_specification = connector_mesh_specification
        self.yaw_joint_specification = yaw_joint_specification
        self.pitch_joint_specification = pitch_joint_specification
        self.roll_joint_specification = roll_joint_specification


class SeahorseVertebralStrutSpecification(Specification):
    def __init__(
            self,
            *,
            stiffness: float,
            damping: float,
            beam_width: float
            ) -> None:
        super().__init__()
        self.stiffness = stiffness
        self.damping = damping
        self.beam_width = beam_width


class SeahorseSegmentSpecification(Specification):
    def __init__(
            self,
            *,
            z_offset_from_previous_segment: float,
            vertebrae_specification: SeahorseVertebraeSpecification,
            vertebral_strut_specification: SeahorseVertebralStrutSpecification,
            plate_specifications: List[SeahorsePlateSpecification]
            ) -> None:
        super().__init__()
        self.z_offset_from_previous_segment = FixedParameter(z_offset_from_previous_segment)
        self.vertebrae_specification = vertebrae_specification
        self.vertebral_strut_specification = vertebral_strut_specification
        self.plate_specifications = plate_specifications


class HMMTendonRoutingSpecification(Specification):
    def __init__(
            self,
            start_segment: int,
            end_segment: int,
            intermediate_points: List[int],
            sagittal_sides: List[str],
            coronal_sides: List[str]
            ) -> None:
        super().__init__()
        self.start_segment = FixedParameter(start_segment)
        self.end_segment = FixedParameter(end_segment)
        self.intermediate_points = FixedParameter(intermediate_points)
        self.sagittal_sides = FixedParameter(sagittal_sides)
        self.coronal_sides = FixedParameter(coronal_sides)


class SeahorseHMMTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            p_control: bool,
            p_control_kp: float,
            f_control_gear: float,
            beam_width: float,
            segment_span: int,
            damping: float,
            routing_specifications: List[HMMTendonRoutingSpecification]
            ) -> None:
        super().__init__()
        self.p_control = FixedParameter(p_control)
        self.p_control_kp = FixedParameter(p_control_kp)
        self.f_control_gear = FixedParameter(f_control_gear)
        self.beam_width = FixedParameter(beam_width)
        self.segment_span = FixedParameter(segment_span)
        self.damping = FixedParameter(damping)
        self.routing_specifications = routing_specifications


class SeahorseMVMTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            enabled: bool,
            contraction_factor: float,
            relaxation_factor: float,
            p_control_kp: float,
            beam_width: float,
            damping: float
            ) -> None:
        super().__init__()
        self.enabled = FixedParameter(enabled)
        self.contraction_factor = FixedParameter(contraction_factor)
        self.relaxation_factor = FixedParameter(relaxation_factor)
        self.p_control_kp = FixedParameter(p_control_kp)
        self.beam_width = FixedParameter(beam_width)
        self.damping = FixedParameter(damping)


class SeahorseTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            hm_beam_actuation_specification: SeahorseHMMTendonActuationSpecification,
            mvm_beam_actuation_specification: SeahorseMVMTendonActuationSpecification
            ) -> None:
        super().__init__()
        self.hm_beam_actuation_specification = hm_beam_actuation_specification
        self.mvm_beam_actuation_specification = mvm_beam_actuation_specification


class SeahorseMorphologySpecification(MorphologySpecification):
    sides = ["ventral", "sinistral", "dorsal", "dextral"]
    corners = ["ventral_dextral", "ventral_sinistral", "dorsal_sinistral", "dorsal_dextral"]

    def __init__(
            self,
            *,
            segment_specifications: List[SeahorseSegmentSpecification],
            beam_actuation_specification: SeahorseTendonActuationSpecification
            ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications
        self.beam_actuation_specification = beam_actuation_specification

    @property
    def num_segments(
            self
            ) -> int:
        return len(self.segment_specifications)

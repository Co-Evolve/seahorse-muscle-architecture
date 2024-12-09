from __future__ import annotations

import mujoco
import numpy as np
from biorobot.utils import colors
from moojoco.mjcf.morphology import MJCFMorphology

from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.parts.segment import MJCFSeahorseSegment
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    SeahorseMorphologySpecification, SeahorseTendonActuationSpecification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.utils import calculate_relaxed_beam_length, \
    get_actuator_beam_plate_indices, get_actuator_ghost_taps_index


class MJCFSeahorseMorphology(MJCFMorphology):
    def __init__(
            self,
            specification: SeahorseMorphologySpecification
            ) -> None:
        super().__init__(specification)

    @property
    def morphology_specification(
            self
            ) -> SeahorseMorphologySpecification:
        return super().morphology_specification

    @property
    def beam_actuation_specification(
            self
            ) -> SeahorseTendonActuationSpecification:
        return self.morphology_specification.beam_actuation_specification

    def _build(
            self
            ) -> None:
        self._configure_compiler()
        self._build_tail()
        self._configure_gliding_joint_equality_constraints()
        self._build_hm_beams()
        self._configure_hm_beam_ranges()
        self._configure_mvm_beams()
        self._configure_actuators()
        self._configure_sensors()

    def _configure_compiler(
            self
            ) -> None:
        self.mjcf_model.compiler.angle = 'radian'
        self.mjcf_model.option.flag.contact = 'disable'
        self.mjcf_model.option.flag.gravity = 'disable'


    def _build_tail(
            self
            ) -> None:
        self._segments = []
        num_segments = self.morphology_specification.num_segments
        next_parent = self
        for segment_index in range(num_segments):
            position = np.array(
                    [0.0, 0.0, self.morphology_specification.segment_specifications[
                        segment_index].z_offset_from_previous_segment.value]
                    )
            segment = MJCFSeahorseSegment(
                    parent=next_parent,
                    name=f"segment_{segment_index}",
                    pos=position,
                    euler=np.zeros(3),
                    segment_index=segment_index
                    )
            self._segments.append(segment)

            next_parent = segment.vertebrae

    def _configure_gliding_joint_equality_constraints(
            self
            ) -> None:
        for segment, next_segment in zip(self._segments, self._segments[1:]):
            for plate, next_plate in zip(segment.plates, next_segment.plates):
                try:
                    self.mjcf_model.equality.add(
                            'joint',
                            joint1=plate.x_axis_gliding_joint,
                            joint2=next_plate.x_axis_gliding_joint,
                            polycoef=[0, 1, 0, 0, 0]
                            )
                except AttributeError:
                    pass
                try:
                    self.mjcf_model.equality.add(
                            'joint',
                            joint1=plate.y_axis_gliding_joint,
                            joint2=next_plate.y_axis_gliding_joint,
                            polycoef=[0, 1, 0, 0, 0]
                            )
                except AttributeError:
                    pass

    def _build_hm_beams(
            self
            ) -> None:
        self._hm_beams = []

        hm_beam_actuation_specification = self.beam_actuation_specification.hm_beam_actuation_specification

        for routing_specification in hm_beam_actuation_specification.routing_specifications:
            for x_side in ["ventral", "dorsal"]:
                if x_side not in routing_specification.sagittal_sides.value:
                    continue
                for y_side in ["dextral", "sinistral"]:
                    if y_side not in routing_specification.coronal_sides.value:
                        continue

                    start_index = routing_specification.start_segment.value
                    stop_index = routing_specification.end_segment.value

                    taps = []

                    if start_index >= 0:
                        plate_index, _ = get_actuator_beam_plate_indices(side=x_side, segment_index=start_index)
                        plate = self._segments[start_index].plates[plate_index]

                        ghost_taps_index = get_actuator_ghost_taps_index(
                                x_side=x_side, y_side=y_side, segment_index=start_index
                                )
                        taps += plate.lower_ghost_hm_taps[ghost_taps_index]

                    end_plate_index = self.morphology_specification.corners.index(f"{x_side}_{y_side}")
                    end_plate = self._segments[stop_index].plates[end_plate_index]

                    intermediate_points = iter(routing_specification.intermediate_points.value)
                    for segment_index in range(start_index + 1, stop_index):
                        if segment_index < 0:
                            continue

                        plate = self._segments[segment_index].plates[end_plate_index]

                        tap_index = next(intermediate_points)  # routing_specification.intermediate_points.value[i]
                        taps += plate.intermediate_hm_taps[tap_index]

                    taps += end_plate.hm_tap_end

                    beam = self.mjcf_model.tendon.add(
                            'spatial',
                            name=f"hm_beam_{x_side}_{y_side}_{stop_index}",
                            width=hm_beam_actuation_specification.beam_width.value,
                            rgba=colors.rgba_blue,
                            damping=hm_beam_actuation_specification.damping.value
                            )

                    for i, tap in enumerate(taps):
                        beam.add('site', site=tap)

                    self._hm_beams.append(beam)

    def _configure_hm_beam_ranges(
            self
            ) -> None:
        # Set beam ranges based on maximum curvature

        xml_str, assets = self.get_mjcf_str(), self.get_mjcf_assets()
        model = mujoco.MjModel.from_xml_string(xml=xml_str, assets=assets)
        data = mujoco.MjData(model)
        vertebrae_pitch_joints = [model.joint(joint_id) for joint_id in range(model.njnt) if
                                  "vertebrae_joint_pitch" in model.joint(joint_id).name]
        vertebrae_pitch_joints_qpos_adr = np.array(
                [joint.qposadr[0] for joint in vertebrae_pitch_joints]
                )
        vertebrae_pitch_joints_range = np.array(
                [joint.range[1] for joint in vertebrae_pitch_joints]
                )
        data.qpos[vertebrae_pitch_joints_qpos_adr] = vertebrae_pitch_joints_range
        mujoco.mj_forward(model, data)

        for mjcf_beam in self._hm_beams:
            straight_length = model.tendon(mjcf_beam.full_identifier)._length0[0]
            curved_length = data.tendon(mjcf_beam.full_identifier).length[0]
            if "ventral" in mjcf_beam.full_identifier:
                mjcf_beam.range = (curved_length, straight_length)
                mjcf_beam.limited = True
            if "dorsal" in mjcf_beam.full_identifier:
                mjcf_beam.range = (straight_length, curved_length)
                mjcf_beam.limited = True

    def _configure_mvm_beams(
            self
            ) -> None:
        self._mvm_beams = []
        if not self.beam_actuation_specification.mvm_beam_actuation_specification.enabled.value:
            return

        mvm_beam_actuation_specification = self.beam_actuation_specification.mvm_beam_actuation_specification
        for segment, next_segment in zip(self._segments, self._segments[1:]):
            start_plate = segment.plates[1]
            end_plate = next_segment.plates[1]

            for side in ["sinistral", "dextral"]:
                start_tap = start_plate.mvm_taps[side][1]
                end_tap = end_plate.mvm_taps[side][0]
                taps = [start_tap, end_tap]

                base_length = calculate_relaxed_beam_length(
                        morphology_parts=[start_plate, end_plate], attachment_points=taps
                        )

                beam = self.mjcf_model.tendon.add(
                        'spatial',
                        name=f"mvm_beam_{side}_{segment.segment_index, next_segment.segment_index}",
                        width=mvm_beam_actuation_specification.beam_width.value,
                        rgba=colors.rgba_red,
                        limited=True,
                        range=[base_length * mvm_beam_actuation_specification.contraction_factor.value,
                               base_length * mvm_beam_actuation_specification.relaxation_factor.value],
                        damping=mvm_beam_actuation_specification.damping.value
                        )
                for tap in taps:
                    beam.add('site', site=tap)

                self._mvm_beams.append(beam)

    def _configure_actuators(
            self
            ) -> None:
        self._beam_actuators = []
        for beam in self._hm_beams:
            if not beam.limited:
                continue
            if self.beam_actuation_specification.hm_beam_actuation_specification.p_control.value:
                kp = self.beam_actuation_specification.hm_beam_actuation_specification.p_control_kp.value

                self._beam_actuators.append(
                        self.mjcf_model.actuator.add(
                                'position',
                                tendon=beam,
                                name=beam.name,
                                forcelimited=True,
                                # only allow contraction forces
                                forcerange=[-kp, 0],
                                ctrllimited=True,
                                ctrlrange=beam.range,
                                kp=kp
                                )
                        )
            else:
                gear = self.beam_actuation_specification.hm_beam_actuation_specification.f_control_gear.value
                self._beam_actuators.append(
                        self.mjcf_model.actuator.add(
                                'motor',
                                tendon=beam,
                                name=beam.name,
                                forcelimited=True,
                                # only allow contraction forces
                                forcerange=[-gear, 0],
                                ctrllimited=True,
                                ctrlrange=[-1, 0],
                                gear=[gear]
                                )
                        )
        for beam in self._mvm_beams:
            if not beam.limited:
                continue
            kp = self.beam_actuation_specification.mvm_beam_actuation_specification.p_control_kp.value
            self._beam_actuators.append(
                    self.mjcf_model.actuator.add(
                            'position', tendon=beam, name=beam.name, forcelimited=True,  # only allow contraction forces
                            forcerange=[-kp, 0], ctrllimited=True, ctrlrange=beam.range, kp=kp
                            )
                    )

    def _configure_sensors(
            self
            ) -> None:
        for beam in self._hm_beams + self._mvm_beams:
            self.mjcf_model.sensor.add("tendonpos", name=f"{beam.name}_position_sensor", tendon=beam)

        vertebral_joints = [joint for joint in self.mjcf_model.find_all("joint") if "vertebrae" in joint.name]
        for joint in vertebral_joints:
            self.mjcf_model.sensor.add("jointactuatorfrc", name=f"{joint.name}_actuatorfrc_sensor", joint=joint)

        for actuator in self._beam_actuators:
            self.mjcf_model.sensor.add("actuatorfrc", name=f"{actuator.name}_actuatorfrc_sensor", actuator=actuator)

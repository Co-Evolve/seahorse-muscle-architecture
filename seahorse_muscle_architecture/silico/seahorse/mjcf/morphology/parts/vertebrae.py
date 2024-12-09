from typing import Union

import numpy as np
from biorobot.utils import colors
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    SeahorseMorphologySpecification, SeahorseVertebraeSpecification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.utils import add_mesh_to_body

SEAHORSE_VERTEBRAE_COLOR = np.array([158, 38, 212, 255]) / 255


class MJCFSeahorseVertebrae(MJCFMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCFMorphology, MJCFMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs
            ) -> None:
        super().__init__(parent=parent, name=name, position=pos, euler=euler, *args, **kwargs)

    @property
    def morphology_specification(
            self
            ) -> SeahorseMorphologySpecification:
        return super().morphology_specification

    @property
    def vertebrae_specification(
            self
            ) -> SeahorseVertebraeSpecification:
        return self.morphology_specification.segment_specifications[self.segment_index].vertebrae_specification

    @property
    def is_first_segment(
            self
            ) -> bool:
        return self.segment_index == 0

    @property
    def is_last_segment(
            self
            ) -> bool:
        return self.segment_index == (self.morphology_specification.num_segments - 1)

    def _build(
            self,
            segment_index: int
            ) -> None:
        self.segment_index = segment_index

        self._configure_mesh_assets()
        self._build_vertebrae()
        self._build_ball_bearing()
        self._build_connectors()
        self._configure_vertebral_strut_attachment_points()
        self._configure_vertebral_joints()
        self._configure_sensors()

    def _configure_mesh_assets(
            self
            ) -> None:
        vertebrae_mesh_specification = self.vertebrae_specification.vertebral_mesh_specification
        ball_bearing_mesh_specification = self.vertebrae_specification.ball_bearing_mesh_specification
        connector_mesh_specification = self.vertebrae_specification.connector_mesh_specification

        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_vertebrae",
                file=vertebrae_mesh_specification.mesh_path.value,
                scale=vertebrae_mesh_specification.scale_ratio.value
                )
        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_ball_bearing",
                file=ball_bearing_mesh_specification.mesh_path.value,
                scale=ball_bearing_mesh_specification.scale_ratio.value
                )
        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_connector",
                file=connector_mesh_specification.mesh_path.value,
                scale=connector_mesh_specification.scale_ratio.value
                )

    def _build_vertebrae(
            self
            ) -> None:
        self.vertebrae = add_mesh_to_body(
                body=self.mjcf_body,
                name=f"{self.base_name}_vertebrae",
                mesh_name=f"{self.base_name}_vertebrae",
                position=np.zeros(3),
                euler=np.zeros(3),
                rgba=SEAHORSE_VERTEBRAE_COLOR,
                group=1,
                mesh_specification=self.vertebrae_specification.vertebral_mesh_specification, )

    def _build_ball_bearing(
            self
            ) -> None:
        self.ball_bearing = add_mesh_to_body(
                body=self.mjcf_body,
                name=f"{self.base_name}_ball_bearing",
                mesh_name=f"{self.base_name}_ball_bearing",
                position=np.array([0, 0, self.vertebrae_specification.z_offset_to_ball_bearing.value]),
                euler=np.zeros(3),
                rgba=colors.rgba_gray,
                group=1,
                mesh_specification=self.vertebrae_specification.ball_bearing_mesh_specification
                )

    def _build_connectors(
            self
            ) -> None:
        sides = ["ventral", "dextral", "dorsal", "sinistral"]
        radius = (
                self.vertebrae_specification.offset_to_bar_end.value +
                self.vertebrae_specification.connector_length.value / 2)

        angles = np.array([np.pi / 2 * side_index for side_index in range(4)])
        positions = radius * np.array([[np.cos(angle), np.sin(angle), 0.0] for angle in angles])
        eulers = np.array([[0, 0, angle] for angle in angles])

        self.connectors = []
        for side, position, euler in zip(sides, positions, eulers):
            connector = add_mesh_to_body(
                    body=self.mjcf_body,
                    name=f"{self.base_name}_connector_{side}",
                    mesh_name=f"{self.base_name}_connector",
                    position=position,
                    euler=euler,
                    rgba=colors.rgba_gray,
                    group=1,
                    mesh_specification=self.vertebrae_specification.connector_mesh_specification
                    )
            self.connectors.append(connector)

    def _configure_vertebral_strut_attachment_points(
            self
            ) -> None:
        sides = self.morphology_specification.sides
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        radius = self.vertebrae_specification.offset_to_vertebral_strut_attachment_point.value

        self.s_taps = []
        for side, angle in zip(sides, angles):
            pos = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
            s_tap = self.mjcf_body.add(
                    'site',
                    name=f"{self.base_name}_s_tap_{side}",
                    type="sphere",
                    rgba=colors.rgba_red,
                    pos=pos,
                    size=[0.001]
                    )
            self.s_taps.append(s_tap)

    def _configure_vertebral_joints(
            self
            ) -> None:
        if self.is_first_segment:
            return

        joint_pos = np.array([0.0, 0.0, -self.vertebrae_specification.z_offset_to_ball_bearing.value])

        yaw_joint_specification = self.vertebrae_specification.yaw_joint_specification
        pitch_joint_specification = self.vertebrae_specification.pitch_joint_specification
        roll_joint_specification = self.vertebrae_specification.roll_joint_specification

        self._pitch_joint = self._roll_joint = self._yaw_joint = None
        if pitch_joint_specification.range.value != 0:
            self._pitch_joint = self.mjcf_body.add(
                    'joint',
                    name=f'{self.base_name}_vertebrae_joint_pitch',
                    type='hinge',
                    pos=joint_pos,
                    limited=True,
                    axis=[0, 1, 0],
                    range=(-pitch_joint_specification.range.value, pitch_joint_specification.range.value),
                    damping=pitch_joint_specification.damping.value,
                    stiffness=pitch_joint_specification.stiffness.value,
                    frictionloss=pitch_joint_specification.friction_loss.value,
                    armature=pitch_joint_specification.armature.value
                    )
        if roll_joint_specification.range.value != 0:
            self._roll_joint = self.mjcf_body.add(
                    'joint',
                    name=f'{self.base_name}_vertebrae_joint_roll',
                    type='hinge',
                    pos=joint_pos,
                    limited=True,
                    axis=[1, 0, 0],
                    range=(-roll_joint_specification.range.value, roll_joint_specification.range.value),
                    damping=roll_joint_specification.damping.value,
                    stiffness=roll_joint_specification.stiffness.value,
                    frictionloss=roll_joint_specification.friction_loss.value,
                    armature=roll_joint_specification.armature.value
                    )
        if yaw_joint_specification.range.value != 0:
            self.mjcf_body.add(
                    'joint',
                    name=f'{self.base_name}_vertebrae_joint_yaw',
                    type='hinge',
                    pos=joint_pos,
                    limited=True,
                    axis=[0, 0, 1],
                    range=(-yaw_joint_specification.range.value, yaw_joint_specification.range.value),
                    damping=yaw_joint_specification.damping.value,
                    stiffness=yaw_joint_specification.stiffness.value,
                    frictionloss=yaw_joint_specification.friction_loss.value
                    )

    def _configure_sensors(
            self
            ) -> None:
        if self.is_first_segment:
            return

        if self._pitch_joint:
            self.mjcf_model.sensor.add("jointpos", name=f"{self._pitch_joint.name}_sensor", joint=self._pitch_joint)
        if self._roll_joint:
            self.mjcf_model.sensor.add(
                    "jointpos", name=f"{self._roll_joint.name}_sensor", joint=self._roll_joint
                    )
        if self._yaw_joint:
            self.mjcf_model.sensor.add(
                    "jointpos", name=f"{self._yaw_joint.name}_sensor", joint=self._yaw_joint
                    )

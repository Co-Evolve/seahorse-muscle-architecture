from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import mujoco
import numpy as np
from moojoco.environment.base import BaseEnvState, MuJoCoEnvironmentConfiguration
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable
from moojoco.environment.renderer import MujocoRenderer

from seahorse_muscle_architecture.silico.seahorse.mjcf.arena.empty_arena import EmptyArena
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.morphology import MJCFSeahorseMorphology


class SeahorseEnvironmentConfiguration(MuJoCoEnvironmentConfiguration):
    def __init__(
            self,
            *args,
            **kwargs
            ) -> None:
        super().__init__(
                time_scale=2.0, num_physics_steps_per_control_step=10, simulation_time=5, *args, **kwargs
                )


class SeahorseMJCEnvironment(MJCEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: SeahorseEnvironmentConfiguration
            ) -> None:
        super().__init__(mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration)

    @property
    def environment_configuration(
            self
            ) -> SeahorseEnvironmentConfiguration:
        return super().environment_configuration

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFSeahorseMorphology,
            arena: EmptyArena,
            configuration: SeahorseEnvironmentConfiguration
            ) -> SeahorseMJCEnvironment:
        arena.attach(
                other=morphology,
                position=arena.seahorse_attachment_site.pos,
                euler=arena.seahorse_attachment_site.euler,
                free_joint=False
                )
        mjcf_str, mjcf_assets = arena.get_mjcf_str(), arena.get_mjcf_assets()
        return cls(
                mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
                )

    def _create_observables(
            self
            ) -> List[MJCObservable]:
        sensors = [self.frozen_mj_model.sensor(i) for i in range(self.frozen_mj_model.nsensor)]
        joint_pos_sensors = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS]

        vertebrae_pitch_sensors = [sensor for sensor in joint_pos_sensors if
                                   "pitch" in sensor.name and "vertebrae" in sensor.name]
        vertebrae_roll_sensors = [sensor for sensor in joint_pos_sensors if
                                  "roll" in sensor.name and "vertebrae" in sensor.name]
        vertebrae_yaw_sensors = [sensor for sensor in joint_pos_sensors if
                                 "yaw" in sensor.name and "vertebrae" in sensor.name]

        vertebrae_pitch_joints = [self.frozen_mj_model.joint(sensor.objid[0]) for sensor in vertebrae_pitch_sensors]
        vertebrae_roll_joints = [self.frozen_mj_model.joint(sensor.objid[0]) for sensor in vertebrae_roll_sensors]
        vertebrae_yaw_joints = [self.frozen_mj_model.joint(sensor.objid[0]) for sensor in vertebrae_yaw_sensors]

        vertebrae_pitch_joint_pos_observable = MJCObservable(
                name=f"vertebrae_pitch_joint_pos",
                low=np.array([joint.range[0] for joint in vertebrae_pitch_joints]),
                high=np.array([joint.range[1] for joint in vertebrae_pitch_joints]),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         vertebrae_pitch_sensors]
                        )
                )
        vertebrae_roll_joint_pos_observable = MJCObservable(
                name=f"vertebrae_roll_joint_pos",
                low=np.array([joint.range[0] for joint in vertebrae_roll_joints]),
                high=np.array([joint.range[1] for joint in vertebrae_roll_joints]),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         vertebrae_roll_sensors]
                        )
                )
        vertebrae_yaw_joint_pos_observable = MJCObservable(
                name=f"vertebrae_yaw_joint_pos",
                low=np.array([joint.range[0] for joint in vertebrae_yaw_joints]),
                high=np.array([joint.range[1] for joint in vertebrae_yaw_joints]),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         vertebrae_yaw_sensors]
                        )
                )

        hm_beam_pos_sensors = [sensor for sensor in sensors if
                               sensor.type[0] == mujoco.mjtSensor.mjSENS_TENDONPOS and "hm" in sensor.name]
        hm_beams = [self.frozen_mj_model.tendon(sensor.objid[0]) for sensor in hm_beam_pos_sensors]
        hm_beam_pos_observable = MJCObservable(
                name="hm_beam_pos",
                low=np.array([beam._range[0] if beam._limited else 0 for beam in hm_beams]),
                high=np.array([beam._range[1] if beam._limited else np.inf for beam in hm_beams]),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         hm_beam_pos_sensors]
                        )
                )
        mvm_beam_pos_sensors = [sensor for sensor in sensors if
                                sensor.type[0] == mujoco.mjtSensor.mjSENS_TENDONPOS and "mvm" in sensor.name]
        mvm_beams = [self.frozen_mj_model.tendon(sensor.objid[0]) for sensor in mvm_beam_pos_sensors]
        mvm_beam_pos_observable = MJCObservable(
                name="mvm_beam_pos",
                low=np.array([beam._range[0] if beam._limited else 0 for beam in mvm_beams]),
                high=np.array([beam._range[1] if beam._limited else np.inf for beam in mvm_beams]),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         mvm_beam_pos_sensors]
                        )
                )

        actuator_frc_sensors = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_ACTUATORFRC]
        actuator_frc_observable = MJCObservable(
                name="actuator_force",
                low=-np.inf * np.ones_like(actuator_frc_sensors),
                high=np.zeros_like(actuator_frc_sensors),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         actuator_frc_sensors]
                        )
                )

        joint_actuator_frc_sensors = [sensor for sensor in sensors if
                                      sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTACTFRC]
        joint_actuator_frc_observable = MJCObservable(
                name="joint_actuator_force",
                low=-np.inf * np.ones_like(joint_actuator_frc_sensors),
                high=np.inf * np.ones_like(joint_actuator_frc_sensors),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]][0] for sensor in
                         joint_actuator_frc_sensors]
                        )
                )

        segment_torque_sensors = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_TORQUE]
        segment_torque_observables = MJCObservable(
                name="segment_torque",
                low=-np.inf * np.ones(3 * len(segment_torque_sensors)),
                high=np.inf * np.ones(3 * len(segment_torque_sensors)),
                retriever=lambda
                    state: np.array(
                        [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                         segment_torque_sensors]
                        )
                )

        return [vertebrae_pitch_joint_pos_observable, vertebrae_roll_joint_pos_observable,
                vertebrae_yaw_joint_pos_observable, hm_beam_pos_observable, mvm_beam_pos_observable,
                actuator_frc_observable, joint_actuator_frc_observable, segment_torque_observables]

    def get_renderer(
            self,
            identifier: int,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: BaseEnvState
            ) -> Union[MujocoRenderer, mujoco.Renderer]:
        renderer = super().get_renderer(identifier=identifier, model=model, data=data, state=state)
        if self.environment_configuration.render_mode == "human":
            renderer.max_geom = 5000
        return renderer

    def _get_mj_models_and_datas_to_render(
            self,
            state: MJCEnvState
            ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        return mj_models, mj_datas

    @staticmethod
    def _get_time(
            state: MJCEnvState
            ) -> MJCEnvState:
        return state.mj_data.time

    def reset(
            self,
            rng: np.random.RandomState,
            *args,
            **kwargs
            ) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()
        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state

    def _update_reward(
            self,
            state: MJCEnvState,
            previous_state: MJCEnvState
            ) -> MJCEnvState:
        return state

    def _update_terminated(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        return state

    def _update_truncated(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        truncated = self._get_time(state=state) > self.environment_configuration.simulation_time
        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    def _update_info(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        return state

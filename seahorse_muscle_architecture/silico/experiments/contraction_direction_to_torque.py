from typing import Tuple

import numpy as np
import pandas as pd
from fprs.parameters import FixedParameter
from matplotlib import pyplot as plt

from seahorse_muscle_architecture.silico.experiments.common import create_env
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import \
    default_seahorse_morphology_specification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    HMMTendonRoutingSpecification, SeahorseMorphologySpecification


def get_specification(
        contraction_direction: int
        ) -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
            num_segments=2, hm_segment_span=0, p_control=False, mvm_enabled=False
            )

    specification.beam_actuation_specification.hm_beam_actuation_specification.f_control_gear = FixedParameter(
            2.5
            )
    # Turn off gliding joints for stability
    for segment in specification.segment_specifications:
        for plate_specification in segment.plate_specifications:
            plate_specification.x_axis_gliding_joint_specification.range = FixedParameter(0)
            plate_specification.y_axis_gliding_joint_specification.range = FixedParameter(0)

    specification.beam_actuation_specification.hm_beam_actuation_specification.routing_specifications = [
            HMMTendonRoutingSpecification(
                    start_segment=-1,
                    end_segment=1,
                    intermediate_points=[contraction_direction],
                    sagittal_sides=["ventral"],
                    coronal_sides=["dextral", "sinistral"]
                    )]
    return specification


def do_contraction_direction_episode(
        contraction_direction: int
        ) -> float:
    morphology_specification = get_specification(contraction_direction=contraction_direction)

    env = create_env(morphology_specification=morphology_specification)
    state = env.reset(rng=np.random.RandomState(seed=42))
    while not (state.terminated | state.truncated):
        action = env.action_space.low
        state = env.step(state=state, action=action)  # env.render(state=state)
    env.close()

    vertebral_torque = state.observations["joint_actuator_force"][0]

    return vertebral_torque


def do_contraction_direction_experiment() -> Tuple[np.ndarray, np.ndarray]:
    contraction_directions = np.arange(0, 10)
    vertebral_torques = np.array(
            [do_contraction_direction_episode(contraction_direction) for contraction_direction in
             contraction_directions]
            )

    vertebral_torques = vertebral_torques / np.max(vertebral_torques)

    return contraction_directions, vertebral_torques


def get_rw_data() -> Tuple[np.ndarray, np.ndarray]:
    contraction_directions = []
    vertebral_torques = []

    for contraction_direction, contraction_direction_name in zip([0, 4, 9], ["innermost", "middle", "outermost"]):
        force_samples = []
        for sample_index in range(1, 11):
            log_file = (f"./seahorse_muscle_architecture/vivo/experiments/contraction_direction_to_torque/data"
                        f"/{contraction_direction_name}/force_"
                        f"{sample_index}.csv")
            force_df = pd.read_csv(log_file, sep='\t')
            times = force_df["time"].to_numpy()
            forces = force_df["force"].to_numpy()
            times = times - times[0]

            final_forces = forces[(np.max(times) - 3) < times]
            avg_final_force = np.mean(final_forces)
            force_samples.append(avg_final_force)
        vertebral_torques.append(np.mean(force_samples))
        contraction_directions.append(contraction_direction)

    return np.array(contraction_directions), np.array(vertebral_torques) / np.max(vertebral_torques)


if __name__ == '__main__':
    sim_contraction_directions, sim_vertebral_torques = do_contraction_direction_experiment()
    rw_contraction_directions, rw_vertebral_torques = get_rw_data()

    plt.plot(sim_contraction_directions, sim_vertebral_torques[::-1], marker='o', zorder=0)
    plt.scatter(rw_contraction_directions, rw_vertebral_torques, color="red", marker="x", zorder=10)
    plt.xlabel("contraction direction")
    plt.ylabel("normalised vertebral torque")
    plt.savefig("contraction_direction_to_torque.png")
    plt.show()
    plt.close()

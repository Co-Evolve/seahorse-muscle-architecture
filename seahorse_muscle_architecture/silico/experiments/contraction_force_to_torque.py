from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from fprs.parameters import FixedParameter
from matplotlib import pyplot as plt
from scipy import interpolate

from seahorse_muscle_architecture.silico.experiments.common import create_env
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import \
    default_seahorse_morphology_specification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    HMMTendonRoutingSpecification, SeahorseMorphologySpecification


def get_specification(
        force: float
        ) -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
            num_segments=2, hm_segment_span=0, p_control=False, mvm_enabled=False
            )

    specification.beam_actuation_specification.hm_beam_actuation_specification.f_control_gear = FixedParameter(
            force
            )

    specification.beam_actuation_specification.hm_beam_actuation_specification.routing_specifications = [
            HMMTendonRoutingSpecification(
                    start_segment=-1,
                    end_segment=1,
                    intermediate_points=[0],
                    sagittal_sides=["ventral"],
                    coronal_sides=["dextral", "sinistral"]
                    )]
    return specification


def do_force_experiment() -> Tuple[np.ndarray, np.ndarray]:
    MIN_FORCE, MAX_FORCE = 0, 5

    morphology_specification = get_specification(force=MAX_FORCE)
    env = create_env(morphology_specification=morphology_specification)
    state = env.reset(rng=np.random.RandomState(seed=42))

    beam_forces, vertebral_torques = [], []
    for step in range(env.environment_configuration.total_num_control_steps):
        alpha = step / env.environment_configuration.total_num_control_steps

        force = MIN_FORCE + alpha * (MAX_FORCE - MIN_FORCE)

        state = env.step(state=state, action=-alpha * np.ones(2))

        beam_forces.append(force)
        vertebral_torques.append(state.observations["joint_actuator_force"][0])

    return np.array(beam_forces), np.array(vertebral_torques)


def get_rw_data() -> Tuple[np.ndarray, np.ndarray]:
    currents_to_forces = defaultdict(list)
    all_currents = []
    all_forces = []

    for sample_index in range(1, 4):
        force_data = (f"./seahorse_muscle_architecture/vivo/experiments/contraction_force_to_torque/data/sampl"
                      f"e{sample_index}/force.csv")
        current_data = (f"./seahorse_muscle_architecture/vivo/experiments/contraction_force_to_torque/data/sample"
                        f"{sample_index}/current.csv")

        force_df = pd.read_csv(force_data, sep='\t')
        current_df = pd.read_csv(current_data, sep='\t')
        times = current_df["time"].to_numpy()

        currents = current_df["present_current"].to_numpy()
        force_interp_func = interpolate.interp1d(
                force_df['time'], force_df['force'], kind='cubic', fill_value='extrapolate'
                )
        forces = force_interp_func(times)

        positions = current_df["position_shift"].to_numpy()
        position_mask = (5 < positions) & (positions < 105)

        currents = currents[position_mask]
        forces = forces[position_mask]

        all_currents.append(currents)
        all_forces.append(forces)

        for current, force in zip(currents, forces):
            currents_to_forces[current].append(force)

    filtered_currents_to_forces = {k: v for k, v in currents_to_forces.items() if len(v) > 1}
    filtered_currents = np.array(sorted(list(filtered_currents_to_forces.keys())))
    filtered_forces = np.array([np.average(filtered_currents_to_forces[current]) for current in filtered_currents])

    return (all_currents, all_forces), (filtered_currents, filtered_forces)


if __name__ == '__main__':
    sim_beam_forces, sim_vertebral_torques = do_force_experiment()
    (all_currents, all_forces), (filtered_rw_motor_currents, filtered_rw_vertebral_torques) = get_rw_data()

    # normalize
    norm_sim_beam_forces = sim_beam_forces / np.max(sim_beam_forces)
    norm_sim_vertebral_torques = sim_vertebral_torques / np.max(sim_vertebral_torques)
    norm_rw_motor_currents = filtered_rw_motor_currents / np.max(filtered_rw_motor_currents)
    norm_rw_vertebral_torques = filtered_rw_vertebral_torques / np.max(filtered_rw_vertebral_torques)

    plt.plot(norm_sim_beam_forces, norm_sim_vertebral_torques, zorder=0)
    plt.scatter(norm_rw_motor_currents, norm_rw_vertebral_torques, color="red", marker="x", zorder=10)
    plt.xlabel("normalised contraction force")
    plt.ylabel("normalised vertebral torque")
    plt.savefig("contraction_force_to_torque.png")
    plt.show()
    plt.close()

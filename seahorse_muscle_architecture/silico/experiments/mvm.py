from typing import List

import numpy as np
from moojoco.environment.mjc_env import MJCEnvState

from seahorse_muscle_architecture.silico.experiments.common import create_env
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import \
    default_seahorse_morphology_specification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    SeahorseMorphologySpecification


def get_specification() -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
            num_segments=11, hm_segment_span=0, p_control=True, mvm_enabled=True
            )
    return specification


def run_simulation_with_rendering(
        num_to_contract: int
        ) -> List[MJCEnvState]:
    morphology_specification = get_specification()
    env = create_env(morphology_specification=morphology_specification)

    state = env.reset(rng=np.random.RandomState(42))
    base_length = state.observations["mvm_beam_pos"]
    for step in range(env.environment_configuration.total_num_control_steps):
        alpha = step / env.environment_configuration.total_num_control_steps
        action = base_length + alpha * (env.action_space.low - base_length)
        action[:-num_to_contract] = base_length[:-num_to_contract]
        state = env.step(state=state, action=action)
        env.render(state=state)

    env.close()


if __name__ == '__main__':
    NUM_MVMS_TO_CONTRACT = [3, 7, 10]
    for num_mvms in NUM_MVMS_TO_CONTRACT:
        run_simulation_with_rendering(num_to_contract=num_mvms)

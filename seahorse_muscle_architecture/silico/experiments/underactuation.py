import numpy as np
from fprs.parameters import FixedParameter

from seahorse_muscle_architecture.silico.experiments.common import run_simulation_with_rendering
from seahorse_muscle_architecture.silico.seahorse.environment.mjc_env import SeahorseEnvironmentConfiguration, \
    SeahorseMJCEnvironment
from seahorse_muscle_architecture.silico.seahorse.mjcf.arena.empty_arena import EmptyArena, EmptyArenaConfiguration
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.morphology import MJCFSeahorseMorphology
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import \
    default_seahorse_morphology_specification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    HMMTendonRoutingSpecification, SeahorseMorphologySpecification


def get_specification() -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
            num_segments=11, hm_segment_span=10, p_control=False, mvm_enabled=False
            )

    end_segment = specification.num_segments - 1
    start_segment = (
            end_segment - specification.beam_actuation_specification.hm_beam_actuation_specification.segment_span.value)

    num_intermediate_segments = end_segment
    if start_segment >= 0:
        num_intermediate_segments -= start_segment + 1

    specification.beam_actuation_specification.hm_beam_actuation_specification.routing_specifications = [
            HMMTendonRoutingSpecification(
                    start_segment=start_segment,
                    end_segment=end_segment,
                    intermediate_points=list(range(num_intermediate_segments - 1, -1, -1)),
                    sagittal_sides=["ventral"],
                    coronal_sides=["sinistral", "dextral"]
                    )]

    return specification


def create_env_with_cylinder() -> SeahorseMJCEnvironment:
    arena_configuration = EmptyArenaConfiguration("hm")
    arena = EmptyArena(arena_configuration)
    arena.mjcf_body.add(
            "geom",
            type="cylinder",
            pos=[0.132, 0, -0.08],
            size=[0.075, 0.08],
            contype=1,
            conaffinity=1,
            euler=[np.pi / 2, 0, 0],
            friction=[3, 1, 1]
            )
    arena.mjcf_body.find_all("camera")[0].pos[2] = -0.12

    morphology_specification = get_specification()
    morphology = MJCFSeahorseMorphology(specification=morphology_specification)
    morphology.mjcf_model.option.flag.contact = 'enable'

    environment_configuration = SeahorseEnvironmentConfiguration(
            render_mode="human", render_size=(960, 1280), camera_ids=[0]
            )
    env = SeahorseMJCEnvironment.from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=environment_configuration
            )
    return env


if __name__ == '__main__':
    env = create_env_with_cylinder()
    run_simulation_with_rendering(env=env)

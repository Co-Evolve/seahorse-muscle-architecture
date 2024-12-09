from seahorse_muscle_architecture.silico.experiments.common import run_simulation_with_rendering
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.default import \
    default_seahorse_morphology_specification
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    HMMTendonRoutingSpecification, SeahorseMorphologySpecification


def get_specification(
        segment_span: int
        ) -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
            num_segments=11, hm_segment_span=segment_span, p_control=True, mvm_enabled=False
            )

    end_segment = specification.num_segments - 1
    start_segment = (
            end_segment - specification.beam_actuation_specification.hm_beam_actuation_specification.segment_span.value)
    num_intermediate_segments = end_segment

    specification.beam_actuation_specification.hm_beam_actuation_specification.routing_specifications = [
            HMMTendonRoutingSpecification(
                    start_segment=start_segment,
                    end_segment=end_segment,
                    intermediate_points=list(range(num_intermediate_segments - 1, -1, -1)),
                    sagittal_sides=["ventral"],
                    coronal_sides=["sinistral", "dextral"]
                    )]

    return specification


if __name__ == '__main__':
    SEGMENT_SPANS = [2, 6, 10]
    for segment_span in SEGMENT_SPANS:
        morphology_specification = get_specification(segment_span=segment_span)
        run_simulation_with_rendering(morphology_specification=morphology_specification)

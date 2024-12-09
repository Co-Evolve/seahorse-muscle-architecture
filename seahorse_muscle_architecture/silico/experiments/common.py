from typing import List

import imageio
import numpy as np
from PIL import Image
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.environment.mjc_env import MJCEnv

from seahorse_muscle_architecture.silico.seahorse.environment.mjc_env import SeahorseEnvironmentConfiguration, \
    SeahorseMJCEnvironment
from seahorse_muscle_architecture.silico.seahorse.mjcf.arena.empty_arena import EmptyArena, EmptyArenaConfiguration
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.morphology import MJCFSeahorseMorphology
from seahorse_muscle_architecture.silico.seahorse.mjcf.morphology.specification.specification import \
    SeahorseMorphologySpecification


def create_env(
        morphology_specification: SeahorseMorphologySpecification
        ) -> SeahorseMJCEnvironment:
    arena_configuration = EmptyArenaConfiguration("")
    arena = EmptyArena(arena_configuration)

    morphology = MJCFSeahorseMorphology(specification=morphology_specification)

    environment_configuration = SeahorseEnvironmentConfiguration(
            render_mode="human", render_size=(960, 1280)
            )
    env = SeahorseMJCEnvironment.from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=environment_configuration
            )
    return env


def run_simulation_with_rendering(
        morphology_specification: SeahorseMorphologySpecification | None = None,
        env: MJCEnv | None = None
        ) -> None:
    if env is None:
        assert morphology_specification is not None
        env = create_env(morphology_specification=morphology_specification)

    state = env.reset(rng=np.random.RandomState(42))

    for step in range(env.environment_configuration.total_num_control_steps):
        alpha = step / env.environment_configuration.total_num_control_steps

        action = env.action_space.high + alpha * (env.action_space.low - env.action_space.high)
        state = env.step(state=state, action=action)
        env.render(state=state)

    env.close()


def post_render(
        render_output: List[np.ndarray],
        environment_configuration: MuJoCoEnvironmentConfiguration
        ) -> np.ndarray:
    if render_output is None:
        return

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in frames_per_env]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR


def create_video(
        frames: List[np.ndarray],
        simulation_time: int,
        path: str
        ) -> None:
    fps = int(len(frames) / simulation_time)
    imgio_kargs = {
            'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
            'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    writer = imageio.get_writer(path, **imgio_kargs)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def make_white_transparent(
        image_path,
        output_path
        ):
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    data = np.array(image)

    # Identify white pixels
    white_areas = (data[:, :, 0:3] == [255, 255, 255]).all(axis=-1)
    # Make white pixels transparent by setting alpha to 0
    data[white_areas, 3] = 0

    # Convert back to an image
    transparent_image = Image.fromarray(data)

    # Save the image
    transparent_image.save(output_path)

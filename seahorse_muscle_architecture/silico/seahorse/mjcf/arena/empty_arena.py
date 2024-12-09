import numpy as np
from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena


class EmptyArenaConfiguration(ArenaConfiguration):
    def __init__(
            self,
            name: str
            ) -> None:
        super().__init__(name=name)


class EmptyArena(MJCFArena):
    @property
    def arena_configuration(
            self
            ) -> EmptyArenaConfiguration:
        return super().arena_configuration

    def _build(
            self
            ) -> None:
        self._build_tail_attachment()
        self._configure_lights()
        self._configure_cameras()
        self._configure_background()

    def _configure_background(
            self
            ) -> None:
        self.mjcf_model.asset.add(
                'texture', type='skybox', builtin='flat', rgb1='1.0 1.0 1.0', rgb2='1.0 1.0 1.0', width=200, height=200
                )

    def _build_tail_attachment(
            self
            ) -> None:
        self.seahorse_attachment_site = self.mjcf_body.add(
                'site', name="seahorse_attachment_site", pos=[0.0, 0.0, 0], euler=[np.pi, 0.0, 0.0]
                )

    def _configure_cameras(
            self
            ) -> None:
        self.mjcf_model.worldbody.add(
                'camera', name='side_camera', pos=[0.075, -0.6, -0.215], zaxis=[0, -1, 0]
                )
        self.mjcf_model.worldbody.add(
                'camera',
                name='corner',
                pos=[0.43011626335213127, -0.43011626335213127, -0.175],
                xyaxes=(0.5, 0.5, 0) + (0, 0, 1)
                )
        self.mjcf_model.worldbody.add(
                'camera', name='ventral', pos=[0.55, 0.0, -0.175], xyaxes=(0, 1, 0) + (0, 0, 1)
                )
        self.mjcf_model.worldbody.add(
                'camera', name='top-down', pos=[0, 0, 0.5], xyaxes=(0, 1, 0) + (-1, 0, 0)
                )
        self.mjcf_model.worldbody.add(
                'camera', name='ventral-closeup', pos=[0.25, 0, -0.02], xyaxes=(0, 1, 0) + (0, 0, 1)
                )

    def _configure_lights(
            self
            ) -> None:
        self.mjcf_model.worldbody.add(
                'light',
                name='corner',
                pos=[0.43011626335213127, -0.43011626335213127, -0.175],
                dir=[1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                directional=True,
                ambient=[0.1, 0.1, 0.1]
                )
        self.mjcf_model.worldbody.add(
                'light',
                name='ventral',
                pos=[0.6, 0, -0.175],
                dir=[-1, 0, 0],
                directional=True,
                ambient=[0.01, 0.01, 0.01],
                specular=[0, 0, 0]
                )

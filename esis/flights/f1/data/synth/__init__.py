"""Create synthetic solar scenes that can be observed with an instrument model."""

from ._scene_aia import scene_aia
from ._scene_iris import velocity_scale_default, scene_iris

__all__ = [
    "scene_aia",
    "velocity_scale_default",
    "scene_iris",
]

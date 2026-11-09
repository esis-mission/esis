"""Create synthetic solar scenes in the passband of this flight."""

from ._scene_dem import scene_dem
from ._scene_muram import path_muram, scene_muram

__all__ = [
    "path_muram",
    "scene_dem",
    "scene_muram",
]

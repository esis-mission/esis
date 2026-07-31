"""The solar data captured during this flight."""

from ._fits import path_fits
from ._level_0 import level_0
from ._level_1 import level_1
from ._level_4 import level_4, level_4_frame, level_4_parallel
from ._aia import aia_context
from . import synth

__all__ = [
    "path_fits",
    "level_0",
    "level_1",
    "level_4",
    "level_4_frame",
    "level_4_parallel",
    "aia_context",
    "synth",
]

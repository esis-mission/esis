"""
A generalized model of the ESIS optical system.
"""

from . import abc
from ._front_apertures import FrontAperture
from ._central_obscurations import CentralObscuration
from ._primary_mirrors import PrimaryMirror

__all__ = [
    "abc",
    "FrontAperture",
    "CentralObscuration",
    "PrimaryMirror",
]

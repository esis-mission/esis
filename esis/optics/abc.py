"""
Abstract classes used in this module.
"""

from ._front_apertures import AbstractFrontAperture
from ._central_obscurations import AbstractCentralObscuration
from ._primary_mirrors import AbstractPrimaryMirror
from ._field_stops import AbstractFieldStop
from ._gratings import AbstractGrating

__all__ = [
    "AbstractFrontAperture",
    "AbstractCentralObscuration",
    "AbstractPrimaryMirror",
    "AbstractFieldStop",
    "AbstractGrating",
]

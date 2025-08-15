"""A generalized model of the ESIS optical system."""

from . import abc
from . import mixins
from ._front_apertures import FrontAperture
from ._central_obscurations import CentralObscuration
from ._primary_mirrors import PrimaryMirror
from ._field_stops import FieldStop
from ._gratings import Grating
from ._filters import Filter
from ._sensors import Sensor
from ._cameras import Camera
from ._requirements import Requirements

__all__ = [
    "abc",
    "mixins",
    "FrontAperture",
    "CentralObscuration",
    "PrimaryMirror",
    "FieldStop",
    "Grating",
    "Filter",
    "Sensor",
    "Camera",
    "Requirements",
]

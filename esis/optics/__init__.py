"""A generalized model of the optical system."""

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
from ._instruments import Instrument
from ._distortions import (
    DistortionParameters,
    DistortionObjective,
    DistortionResidual,
    ConvergenceLogger,
    fit_distortion,
    fit_distortion_scan,
    fit_distortion_series,
)

num_interpolation = 32
"""
The number of nodes used to interpolate the response of each multilayer
coating over the angle of incidence.

Solving the transfer matrices of a coating costs far more than the raytrace
it belongs to, and the response is smooth in the angle of incidence, over
which each ESIS surface spans only a few degrees.  Interpolating it reproduces
the exact solve to better than a part in :math:`10^5` while making
:meth:`optika.systems.SequentialSystem.linearize` about two and a half times
faster.

See :attr:`optika.materials.AbstractMultilayerMaterial.num_interpolation`.
Set to :obj:`None` to solve every ray exactly, which is worth doing when
checking a change to the optical model.
"""

__all__ = [
    "num_interpolation",
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
    "Instrument",
    "DistortionParameters",
    "DistortionObjective",
    "DistortionResidual",
    "ConvergenceLogger",
    "fit_distortion",
    "fit_distortion_scan",
    "fit_distortion_series",
]

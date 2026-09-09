"""
Fit the distortion of the instrument to observed images.

The fit has three stages.  An absolute stage places each channel against a
proxy image of the Sun with :class:`LinearMerit` and :func:`fit_distortion`;
it is reproducible from the as-built model and reaches the committed
reference, but it constrains the mapping only to about a pixel.  The
quantities shared by every channel are then made shared.  An internal
stage, :func:`align_channels`, aligns the channels to one another on the
sky plane to a fifth of a pixel in the grating and camera terms, without
revising what the absolute stage set.
"""

from ._parameters import DistortionParameters, KAPPA_FOCUS
from ._merit import LinearMerit, correlation, correlation_raytraced
from ._fit import fit_distortion, polish
from ._alignment import align_channels, sky_grid, sample_on_sky, measure_shifts

__all__ = [
    "DistortionParameters",
    "KAPPA_FOCUS",
    "LinearMerit",
    "correlation",
    "correlation_raytraced",
    "fit_distortion",
    "polish",
    "align_channels",
    "sky_grid",
    "sample_on_sky",
    "measure_shifts",
]

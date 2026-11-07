"""
Fit the distortion of the instrument to observed images.

:class:`DistortionParameters` are the degrees of freedom.
:class:`LinearMerit` compares a channel's linearized image of a proxy scene
with a frame, and :func:`fit_distortion` searches it, to about a pixel.
:func:`fit_distortion_outline` places each channel's windows on the edges
measured in the frames by :func:`measure_edges`, which the merit cannot
resolve.  :func:`align_channels` aligns the channels to one another on the
sky in the grating and camera terms, to a tenth of a pixel;
:func:`measure_channel_shifts` is its measurement on its own.
"""

from ._parameters import DistortionParameters, KAPPA_FOCUS
from ._merit import LinearMerit, correlation, correlation_raytraced
from ._fit import fit_distortion, polish
from ._alignment import (
    align_channels,
    sky_grid,
    sample_on_sky,
    measure_shifts,
    window_mask,
    measure_channel_shifts,
    median_shifts,
    shift_modes,
)
from ._outline import (
    SIDES,
    measure_edges,
    predict_edges,
    outline_residual,
    width_residual,
    fit_distortion_outline,
)

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
    "window_mask",
    "measure_channel_shifts",
    "median_shifts",
    "shift_modes",
    "SIDES",
    "measure_edges",
    "predict_edges",
    "outline_residual",
    "width_residual",
    "fit_distortion_outline",
]

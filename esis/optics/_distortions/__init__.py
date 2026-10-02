"""
Fit the distortion of the instrument to observed images.

An absolute stage places each channel against a proxy image of the Sun
with :class:`LinearMerit` and :func:`fit_distortion`; it is reproducible
from the as-built model, but one frame constrains only some combinations
of the mapping's terms, to about a pixel.  An outline stage,
:func:`fit_distortion_outline`, places each channel's windows on the edges
measured in the frames by :func:`measure_edges`, which the merit itself
cannot resolve, and the quantities shared by every channel are made
shared.  An internal stage, :func:`align_channels`, aligns the channels to
one another on the sky to a tenth of a pixel in the grating and camera
terms, reading each frame inside its windows with :func:`window_mask`;
:func:`measure_channel_shifts` is its measurement on its own, which a
flight's frames can be put through one by one.
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

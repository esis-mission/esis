from typing import Self
import dataclasses
import numpy as np
import named_arrays as na
import esis
from .. import abc
from .. import Level_0

__all__ = [
    "Level_1",
]


@dataclasses.dataclass(eq=False, repr=False)
class Level_1(
    abc.AbstractChannelData,
):
    """
    ESIS images represented in terms of photoelectrons collected by the sensor.

    This class is intended to be created from an instance of :class:`Level_0`
    using the :meth:from_level_0` method.
    """

    instrument: None | esis.optics.Instrument = None
    """A model of the optical system associated with these images."""

    @classmethod
    def from_level_0(
        cls,
        a: Level_0,
        instrument: None | esis.optics.Instrument = None,
    ) -> Self:
        """
        Create a new instance of this class from an instance of :class:`Level_0`.

        This function applies the following operations to the :class:`Level_0` data:

        * Removes the bias (or pedestal) using :meth:`~esis.data.Level_0.unbiased`.
        * Removes the non-active pixels using  :meth:`~esis.data.Level_0.active`.
        * Converts the signal to electrons using :meth:`~esis.data.Level_0.electrons`.
        * Removes the dark signal using :attr:`~esis.data.Level_0.dark_subtracted`.
        * Removes the cosmic ray spikes using :meth:`~esis.data.Level_0.despiked`.

        Parameters
        ----------
        a
            An instance of :class:`Level_0` to convert.
        instrument
            A model of the ESIS instrument to associate with these observations.
        """
        taps = a.taps
        taps = taps.unbiased
        taps = taps.active
        taps = taps.electrons
        a = a.from_taps(taps)
        a = a.dark_subtracted
        a = a.lights
        a = a.despiked

        return cls(
            inputs=a.inputs,
            outputs=a.outputs[dict(detector_y=slice(None, None, -1))],
            instrument=instrument,
            axis_time=a.axis_time,
            axis_channel=a.axis_channel,
            axis_x=a.axis_x,
            axis_y=a.axis_y,
        )

    def where_shadow(
        self,
        num_search: int = 50,
        threshold: float = 0.8,
        margin: int = 8,
    ) -> na.AbstractScalarArray:
        """
        Estimate a boolean mask of the pixels shaded by the storage-region mask.

        The mask that shades the storage region of each sensor (so it can
        operate in frame-transfer mode) is slightly misaligned and shades a
        channel-dependent number of active columns on the short-wavelength
        edge of the detector.
        Those columns read close to zero electrons where the optical model
        predicts real signal, so they should be excluded from any comparison
        of the images to a forward model.

        The shadow is measured from the data:
        the median image over time and the central half of the detector rows
        gives a signal profile per column, and the columns among the first
        `num_search` whose profile is below `threshold` times the median
        profile of the interior are flagged, plus `margin` additional
        columns to cover the penumbra of the shadow.
        The result is :obj:`True` for shaded pixels, arranged along the
        channel and horizontal axes, and broadcasts against the observations.

        Parameters
        ----------
        num_search
            The number of columns, starting from the short-wavelength edge,
            in which to search for the shadow.
        threshold
            The fraction of the interior median signal below which a column
            is considered shaded.
        margin
            The number of extra columns to mask past the last shaded column,
            covering the penumbra of the shadow and its meander along the
            vertical axis.
        """
        axis_time = self.axis_time
        axis_channel = self.axis_channel
        axis_x = self.axis_x
        axis_y = self.axis_y

        shape = self.outputs.shape
        num_x = shape[axis_x]
        num_y = shape[axis_y]

        interior = self.outputs[{axis_y: slice(num_y // 4, 3 * num_y // 4)}]
        axis_median = tuple(ax for ax in (axis_time, axis_y) if ax in shape)
        profile = np.median(interior, axis=axis_median)

        profile = np.moveaxis(
            profile.ndarray,
            profile.axes.index(axis_channel),
            0,
        )
        plateau = np.median(profile[..., num_search : 5 * num_search], axis=~0)

        num_channel = shape[axis_channel]
        width = np.zeros(num_channel, dtype=int)
        for c in range(num_channel):
            below = np.nonzero(profile[c, :num_search] < threshold * plateau[c])[0]
            if below.size > 0:
                width[c] = below.max() + 1 + margin

        index = na.ScalarArray(np.arange(num_x), axes=(axis_x,))
        width = na.ScalarArray(width, axes=(axis_channel,))

        return index < width

from typing import Self
import dataclasses
import named_arrays as na
from .. import Level_0

__all__ = [
    "Level_1"
]


@dataclasses.dataclass
class Level_1(
    na.FunctionArray[
        na.TemporalSpectralPositionalVectorArray,
        na.ScalarArray,
    ],
):
    """
    ESIS images represented in terms of photoelectrons collected by the sensor.

    This class is intended to be created from an instance of :class:`Level_0`
    using the :meth:from_level_0` method.
    """

    axis_time: str = dataclasses.field(default="time", kw_only=True)
    """The name of the logical axis corresponding to changing time."""

    axis_channel: str = dataclasses.field(default="channel", kw_only=True)
    """The name of the logical axis corresponding to the different channels."""

    axis_x: str = dataclasses.field(default="detector_x", kw_only=True)
    """The name of the horizontal axis of the sensor."""

    axis_y: str = dataclasses.field(default="detector_y", kw_only=True)
    """The name of the vertical axis of the sensor."""

    @classmethod
    def from_level_0(cls, a: Level_0) -> Self:
        """
        Create a new instance of this class from an instance of :class:`Level_0`.

        This function applies the following operations to the :class:`Level_0` data:
        * Removes the bias (or pedestal) using the :meth:`~esis.data.Level_0.unbiased`
          property.
        * Removes the non-active pixels using the :meth:`~esis.data.Level_0.active`
          property.
        * Converts the signal from DN units to electron units using the
          :meth:`~esis.data.Level_0.electrons` property.
        * Removes the dark signal by applying the
          :attr:`~esis.data.Level_0.dark_subtracted` property.
        * Removes the cosmic ray spikes
        This function applies the :meth:`~esis.data.Level_0.unbiased

        Parameters
        ----------
        a
            An instance of :class:`Level_0` to convert.
        """
        taps = a.taps
        taps = taps.unbiased
        taps = taps.active
        taps = taps.electrons
        a = a.from_taps(taps)
        a = a.dark_subtracted
        a = a.lights
        a = a.despiked
        return a

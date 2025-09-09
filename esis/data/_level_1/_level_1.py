from typing import Self
import dataclasses
import esis
from .. import abc
from .. import Level_0

__all__ = [
    "Level_1",
]


@dataclasses.dataclass
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
            outputs=a.outputs,
            instrument=instrument,
            axis_time=a.axis_time,
            axis_channel=a.axis_channel,
            axis_x=a.axis_x,
            axis_y=a.axis_y,
        )

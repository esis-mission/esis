from typing import Self
import dataclasses
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
        * Removes the dark signal by subtracting :attr:`~esis.data.Level_0.dark`.
        * Keeps only the solar images, :attr:`~esis.data.Level_0.lights`.
        * Removes the cosmic ray spikes using :meth:`~esis.data.Level_0.despiked`.

        Each of the first three steps acts on every image on its own, so they
        are applied to :attr:`~esis.data.Level_0.darks` and to
        :attr:`~esis.data.Level_0.lights` separately instead of to the whole
        sequence. The result is the same, but the images between them are
        never converted, which keeps the memory this needs to a minimum.

        Parameters
        ----------
        a
            An instance of :class:`Level_0` to convert.
        instrument
            A model of the ESIS instrument to associate with these observations.
        """
        dark = _electrons(a.darks).despiked.mean(axis=a.axis_time)
        a = _electrons(a.lights)
        a = a - dark.outputs
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


def _electrons(a: Level_0) -> Level_0:
    """
    Remove the bias and the non-active pixels of `a` and convert it to electrons.

    Parameters
    ----------
    a
        The images to convert.
    """
    taps = a.taps
    taps = taps.unbiased
    taps = taps.active
    taps = taps.electrons
    return a.from_taps(taps)

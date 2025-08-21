from typing_extensions import Self
import dataclasses
import pathlib
import numpy as np
import numpy.typing as npt
import astropy.time
import named_arrays as na
import msfc_ccd
import esis

__all__ = [
    "Level_0",
]


@dataclasses.dataclass(eq=False, repr=False)
class Level_0(
    msfc_ccd.SensorData,
):
    """
    Representation of ESIS Level-0 images, the raw data gathered by the instrument.

    The Data Acquisition and Control System (DACS) reads out the cameras and
    saves the resulting images as FITS files.
    This represents those FITS files as a Python class.
    """

    timeline: None | esis.nsroc.Timeline = None
    """The sequence of NSROC events associated with these images."""

    @classmethod
    def from_fits(
        cls,
        path: str | pathlib.Path | na.AbstractScalarArray,
        camera: msfc_ccd.abc.AbstractCamera,
        axis_x: str = "detector_x",
        axis_y: str = "detector_y",
        timeline: None | esis.nsroc.Timeline = None,
    ) -> Self:

        self = super().from_fits(
            path=path,
            camera=camera,
            axis_x=axis_x,
            axis_y=axis_y,
        )

        self.timeline = timeline

        return self

    @property
    def channel(self) -> na.ScalarArray[npt.NDArray[str]]:
        """The name of each ESIS channel in a human-readable format."""
        sn = self.inputs.serial_number
        where_1 = sn == "6"
        where_2 = sn == "7"
        where_3 = sn == "9"
        where_4 = sn == "1"

        result = np.empty_like(sn, dtype=object)

        result[where_1] = "Channel 1"
        result[where_2] = "Channel 2"
        result[where_3] = "Channel 3"
        result[where_4] = "Channel 4"

        return result

    @property
    def time_mission_start(self) -> astropy.time.Time:
        """The :math:`T=0` time of the mission."""
        return self.inputs.time.ndarray.min() - self.timeline.timedelta_esis_start

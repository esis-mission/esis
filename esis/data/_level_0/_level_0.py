from typing_extensions import Self
import dataclasses
import pathlib
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import msfc_ccd
import esis
from .. import abc

__all__ = [
    "Level_0",
]


@dataclasses.dataclass(eq=False, repr=False)
class Level_0(
    msfc_ccd.SensorData,
    abc.AbstractChannelData,
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

        self.outputs = self.outputs.astype(np.float32)

        return self

    @property
    def time_mission_start(self) -> astropy.time.Time:
        """
        The :math:`T=0` time of the mission.

        This is :attr:`esis.nsroc.Timeline.time_mission_start` of
        :attr:`timeline`, rather than a time worked out from these images,
        which would move whenever the first image is not among them.
        """
        return self.timeline.time_mission_start

    def _index_after(self, timedelta: u.Quantity) -> int:
        """
        Return the index of the first image that starts after the given mission time.

        This is the number of images that start at or before that time, so it
        is the number of images if none start after it.
        The images must be in order of time.

        Parameters
        ----------
        timedelta
            The mission time.
        """
        time = self.inputs.time_start
        if self.axis_channel in time.shape:
            time = time.mean(self.axis_channel)
        t = self.time_mission_start + timedelta
        return int(np.sum(time <= t, axis=self.axis_time).ndarray)

    @property
    def _index_lights_start(self) -> int:
        """The index representing the first good image."""
        return self._index_after(self.timeline.timedelta_sparcs_rlg_enable)

    @property
    def _index_lights_stop(self) -> int:
        """One greater than the index representing the last good image."""
        return self._index_after(self.timeline.timedelta_sparcs_rlg_disable)

    @property
    def lights(self) -> Self:
        """
        The sequence of solar images taken during the flight.

        This uses only the images where the ring-laser gyroscope was enabled,
        so this should represent the images with the best-possible pointing stability.
        """
        index_start = self._index_lights_start
        index_stop = self._index_lights_stop
        index_lights = {self.axis_time: slice(index_start, index_stop)}
        return self[index_lights]

    @property
    def darks_up(self) -> Self:
        """
        The dark images collected on the upleg of the trajectory.

        This considers all the images up until the moment the shutter door
        is opened.

        Any images without an exposure time close to the median exposure
        time are ignored.
        This is intended to remove the first 1 or 2 images from the
        beginning of each exposure sequence since these images often have
        a different exposure time than the rest of the sequence.
        The median is that of these images, so a subset of only a few
        images, such as the first two, can keep images with a different
        exposure time.
        """
        axis_time = self.axis_time
        dt = self.inputs.timedelta_requested
        if self.axis_channel in dt.shape:
            dt = dt.mean(self.axis_channel)
        where = np.abs(dt - dt.median()) < (0.1 * u.s)
        index_start = np.argmax(where)[axis_time].ndarray
        index_stop = self._index_after(self.timeline.timedelta_shutter_open)
        index = {axis_time: slice(index_start, index_stop)}
        return self[index]

    @property
    def darks_down(self) -> Self:
        """
        The dark images collected on the downleg of the trajectory.

        This considers all the images after the parachute deployment since there
        is a transient, anomalous signal that occurs during atmospheric re-entry.
        """
        index_start = self._index_after(self.timeline.timedelta_parachute_deploy)
        index_stop = None
        index = {self.axis_time: slice(index_start, index_stop)}
        return self[index]

    @property
    def darks(self) -> Self:
        """
        The dark images used to construct the master dark image.

        This is a concatenation of :attr:`darks_up` and :attr:`darks_down`.

        Raises
        ------
        ValueError
            If none of these images are darks.
        """
        result = np.concatenate(
            arrays=[self.darks_up, self.darks_down],
            axis=self.axis_time,
        )

        if result.shape[self.axis_time] == 0:
            timeline = self.timeline
            raise ValueError(
                "None of these images are darks. Darks are taken before the "
                f"shutter opens, at T+{timeline.timedelta_shutter_open}, and "
                "after the parachute deploys, at "
                f"T+{timeline.timedelta_parachute_deploy}."
            )

        serial_number = result.inputs.serial_number
        for axis in serial_number.shape:
            sn0 = serial_number[{axis: 0}]
            if np.all(sn0 == serial_number):
                serial_number = sn0

        result.inputs.serial_number = serial_number

        return result

    @property
    def dark(self) -> Self:
        r"""
        The master dark image for each channel.

        Calculated by taking the mean of :attr:`darks`.\ :attr:`despiked`
        along :attr:`axis_time`.
        """
        return self.darks.despiked.mean(axis=self.axis_time)

    @property
    def dark_subtracted(self):
        """Subtract the master :attr:`dark` from each image in the sequence."""
        return self - self.dark.outputs

    @property
    def despiked(self) -> Self:
        """Remove cosmic rays using :func:`astroscrappy.detect_cosmics`."""
        return na.despike(
            array=self,
            sigclip=6,
            objlim=3,
            axis=(self.axis_x, self.axis_y),
        )

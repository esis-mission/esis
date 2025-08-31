from typing_extensions import Self
import dataclasses
import pathlib
import IPython.display
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.animation
import matplotlib.pyplot as plt
import astropy.units as u
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

    axis_time: str = dataclasses.field(default="time", kw_only=True)
    """The name of the logical axis corresponding to changing time."""

    axis_channel: str = dataclasses.field(default="channel", kw_only=True)
    """The name of the logical axis corresponding to the different channels."""

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

    def _index_after(self, timedelta: u.Quantity) -> dict[str, int]:
        """
        Return the index of the image after the given mission time.

        Parameters
        ----------
        timedelta
            The mission time.
        """
        time = self.inputs.time_start
        if self.axis_channel in time.shape:
            time = time.mean(self.axis_channel)
        t0 = self.time_mission_start
        t = t0 + timedelta
        where = time > t
        return np.argmax(where)

    @property
    def _index_lights_start(self) -> dict[str, int]:
        """The index representing the first good image."""
        return self._index_after(self.timeline.timedelta_sparcs_rlg_enable)

    @property
    def _index_lights_stop(self) -> dict[str, int]:
        """One greater than the index representing the last good image."""
        return self._index_after(self.timeline.timedelta_sparcs_rlg_disable)

    @property
    def lights(self) -> Self:
        """
        The sequence of solar images taken during the flight.

        This uses only the images where the ring-laser gyroscope was enabled,
        so this should represent the images with the best-possible pointing stability.
        """
        axis_time = self.axis_time
        index_start = self._index_lights_start[axis_time].ndarray
        index_stop = self._index_lights_stop[axis_time].ndarray
        index_lights = {axis_time: slice(index_start, index_stop)}
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

        """
        axis_time = self.axis_time
        dt = self.inputs.timedelta_requested.mean(self.axis_channel)
        where = np.abs(dt - dt.median()) < (0.1 * u.s)
        index_start = np.argmax(where)[axis_time].ndarray
        index_stop = self._index_after(self.timeline.timedelta_shutter_open)[
            axis_time
        ].ndarray
        index = {axis_time: slice(index_start, index_stop)}
        return self[index]

    @property
    def darks_down(self) -> Self:
        """
        The dark images collected on the downleg of the trajectory.

        This considers all the images after the parachute deployment since there
        is a transient, anomalous signal that occurs during atmospheric re-entry.
        """
        axis_time = self.axis_time
        index_start = self._index_after(self.timeline.timedelta_parachute_deploy)[
            axis_time
        ].ndarray
        index_stop = None
        index = {axis_time: slice(index_start, index_stop)}
        return self[index]

    @property
    def darks(self) -> Self:
        """
        The dark images used to construct the master dark image.

        This is a concatenation of :attr:`darks_up` and :attr:`darks_down`.
        """
        return np.concatenate(
            arrays=[self.darks_up, self.darks_down],
            axis=self.axis_time,
        )

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

    def animate(
        self,
        ax: None | matplotlib.axes.Axes | na.AbstractArray = None,
        cmap: None | str | matplotlib.colors.Colormap = None,
        norm: None | str | matplotlib.colors.Normalize = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        cbar_fraction: float = 0.1,
    ) -> matplotlib.animation.FuncAnimation:
        """
        Create an animation using the frames in this dataset.

        Parameters
        ----------
        ax
            The :class:`~Axes` instance(s) to use.
            If :obj:`None`, a new set of axes will be created.
        cmap
            The colormap used to map scalar data to colors.
        norm
            The normalization method used to scale data into the range [0, 1] before
            mapping to colors.
        vmin
            The minimum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        vmax
            The maximum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        cbar_fraction
            The fraction of the space to use for the colorbar axes.
        """
        axis_time = self.axis_time
        axis_channel = self.axis_channel

        if ax is None:
            figwidth = plt.rcParams["figure.figsize"][0]
            figwidth_eff = figwidth * (1 - 2 * cbar_fraction)
            shape = self.shape
            num_channel = shape[axis_channel]
            num_x = shape[self.axis_x]
            num_y = shape[self.axis_y]
            figheight = num_channel * figwidth_eff * num_y / num_x
            fig, ax = na.plt.subplots(
                axis_rows=axis_channel,
                nrows=num_channel,
                sharex=True,
                sharey=True,
                constrained_layout=True,
                figsize=(figwidth, figheight),
                origin="upper",
            )
            na.plt.set_xlabel("detector $x$ (pix)", ax=ax[{axis_channel: ~0}])

        data = self.outputs
        unit = na.unit(data)

        if norm is None:
            if vmin is None:
                vmin = data.percentile(1).ndarray
            if vmax is None:
                vmax = data.percentile(99).ndarray

            if unit is not None:
                vmin = vmin.to(unit).value
                vmax = vmax.to(unit).value

            norm = plt.Normalize(
                vmin=vmin,
                vmax=vmax,
            )

        colorizer = plt.Colorizer(
            cmap=cmap,
            norm=norm,
        )

        pixel = self.inputs.pixel

        ani = na.plt.pcolormovie(
            self.inputs.time.mean(axis_channel),
            pixel.x,
            pixel.y,
            C=data,
            axis_time=axis_time,
            ax=ax,
            kwargs_pcolormesh=dict(
                colorizer=colorizer,
            ),
        )
        na.plt.text(
            x=0.5,
            y=1.01,
            s=self.channel,
            transform=na.plt.transAxes(ax),
            ax=ax,
            ha="center",
            va="bottom",
        )
        na.plt.set_aspect("equal", ax=ax)

        plt.colorbar(
            mappable=plt.cm.ScalarMappable(colorizer=colorizer),
            ax=ax.ndarray,
            label=f"signal ({unit:latex_inline})",
            fraction=cbar_fraction,
        )

        return ani

    def to_jshtml(
        self,
        ax: None | matplotlib.axes.Axes | na.AbstractArray = None,
        cmap: None | str | matplotlib.colors.Colormap = None,
        norm: None | str | matplotlib.colors.Normalize = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        cbar_fraction: float = 0.1,
    ) -> IPython.display.HTML:
        """
        Create a Javascript animation ready to be displayed in a Jupyter notebook.

        Converts the output of :method:`animate` to jshtml using
        :meth:`matplotlib.animation.FuncAnimation.to_jshtml`,
        and then wraps the html string in :class:`IPython.display.HTML`.

        Parameters
        ----------
        ax
            The :class:`~Axes` instance(s) to use.
            If :obj:`None`, a new set of axes will be created.
        cmap
            The colormap used to map scalar data to colors.
        norm
            The normalization method used to scale data into the range [0, 1] before
            mapping to colors.
        vmin
            The minimum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        vmax
            The maximum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        cbar_fraction
            The fraction of the space to use for the colorbar axes.
        """
        ani = self.movie(
            ax=ax,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cbar_fraction=cbar_fraction,
        )

        result = ani.to_jshtml()
        result = IPython.display.HTML(result)

        plt.close(ani._fig)

        return result

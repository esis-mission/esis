import dataclasses
import IPython.display
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.animation
import matplotlib.pyplot as plt
import named_arrays as na
import msfc_ccd

__all__ = [
    "AbstractChannelData",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractChannelData(
    msfc_ccd.abc.AbstractImageData,
):
    """An interface for representing images in the coordinate system of the sensor."""

    axis_time: str = dataclasses.field(default="time", kw_only=True)
    """The name of the logical axis corresponding to changing time."""

    axis_channel: str = dataclasses.field(default="channel", kw_only=True)
    """The name of the logical axis corresponding to the different channels."""

    axis_x: str = dataclasses.field(default="detector_x", kw_only=True)
    """The name of the horizontal axis of the sensor."""

    axis_y: str = dataclasses.field(default="detector_y", kw_only=True)
    """The name of the vertical axis of the sensor."""

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
            The :class:`~matplotlib.axes.Axes` instance(s) to use.
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
            na.plt.set_ylabel("detector $y$ (pix)", ax=ax)

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
        fps: None | float = None,
    ) -> IPython.display.HTML:
        """
        Create a Javascript animation ready to be displayed in a Jupyter notebook.

        Converts the output of :meth:`animate` to Javascript using
        :meth:`matplotlib.animation.Animation.to_jshtml`,
        and then wraps the html string in :class:`IPython.display.HTML`.

        Parameters
        ----------
        ax
            The :class:`~matplotlib.axes.Axes` instance(s) to use.
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
        fps
            The frames per second of the animation.
        """
        ani = self.animate(
            ax=ax,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cbar_fraction=cbar_fraction,
        )

        result = ani.to_jshtml(fps=fps)
        result = IPython.display.HTML(result)

        plt.close(ani._fig)

        return result

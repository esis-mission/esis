"""Event-movie rendering for the Level-4 data product."""

import numpy as np
import scipy.ndimage
import matplotlib.animation
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na

__all__ = [
    "locate_event",
    "animate_event",
]


def _centers(vertices: np.ndarray) -> np.ndarray:
    """
    Compute the midpoints of a vertex grid.

    Parameters
    ----------
    vertices
        The vertices to average.
    """
    return (vertices[:-1] + vertices[1:]) / 2


def locate_event(
    a,
    index_line: None | int = None,
    center: None | u.Quantity = None,
    radius: u.Quantity = 330 * u.arcsec,
    sigma: float = 2,
) -> u.Quantity:
    """
    Locate the strongest compact Doppler event in a Level-4 reconstruction.

    The event is found as the maximum over time of the smoothed product of
    the line intensity and the magnitude of the intensity-weighted mean
    Doppler velocity, restricted to the interior of the field of view.

    Parameters
    ----------
    a
        The :class:`~esis.data.Level_4` reconstruction to search.
    index_line
        The index of the spectral line to search along
        :attr:`~esis.data.Level_4.axis_line`.
        If :obj:`None`, the last line (O V 630 for the baseline product).
    center
        The center of the field of view.
        If :obj:`None`, the center of the scene grid.
    radius
        The radius around `center` to search; the default excludes the
        region outside the octagonal field stop of ESIS-I.
    sigma
        The width of the Gaussian smoothing, in scene cells.
    """
    if index_line is None:
        index_line = a.num_line - 1

    intensity = a.intensity[{a.axis_line: index_line}]
    velocity = a.velocity_mean[{a.axis_line: index_line}]

    x_center = _centers(a.inputs.position.x.ndarray.to_value(u.arcsec))
    y_center = _centers(a.inputs.position.y.ndarray.to_value(u.arcsec))

    if center is None:
        center = [x_center.mean(), y_center.mean()] * u.arcsec

    xx, yy = np.meshgrid(x_center, y_center, indexing="ij")
    distance = np.hypot(
        xx - center[0].to_value(u.arcsec),
        yy - center[1].to_value(u.arcsec),
    )

    signature = np.zeros(xx.shape)
    for t in range(a.shape[a.axis_time]):
        intensity_t = a._index_xy(intensity[{a.axis_time: t}])
        velocity_t = a._index_xy(velocity[{a.axis_time: t}].to(u.km / u.s))
        signature_t = scipy.ndimage.gaussian_filter(
            np.nan_to_num(intensity_t * np.abs(velocity_t)),
            sigma=sigma,
        )
        signature = np.maximum(signature, signature_t)

    signature[distance > radius.to_value(u.arcsec)] = 0
    i, j = np.unravel_index(np.argmax(signature), signature.shape)

    return [x_center[i], y_center[j]] * u.arcsec


def animate_event(
    a,
    position: u.Quantity,
    halfwidth: u.Quantity = 40 * u.arcsec,
    context: None | dict[str, na.FunctionArray] = None,
    cmaps_context: None | dict[str, str] = None,
    labels: None | list[str] = None,
    limit_velocity: None | u.Quantity = None,
    percentile_max: float = 99.5,
    percentile_alpha: float = 99,
    correct_transmission: bool = True,
    interval: int = 200,
) -> matplotlib.animation.FuncAnimation:
    """
    Animate an event: intensity and Doppler maps of every line, with context.

    The figure has one column per spectral line and three rows: the total
    intensity of each line, the Doppler map of each line (color encodes the
    intensity-weighted mean velocity, opacity the intensity), and — if
    `context` is given — co-temporal context images (e.g. AIA channels),
    each shown at the frame nearest in time to the current Level-4 frame.

    Parameters
    ----------
    a
        The :class:`~esis.data.Level_4` reconstruction to animate.
    position
        The center of the event, in scene coordinates.
    halfwidth
        The half-width of the field of view of the movie.
    context
        A mapping from label to a context-image function of
        ``(time, x, y)``, on the same coordinate frame as the scene.
    cmaps_context
        An optional mapping from context label to colormap name.
    labels
        The label of each spectral line.
        If :obj:`None`, the rest wavelength of each line is used.
    limit_velocity
        The Doppler velocity mapped to the ends of the colormap.
        If :obj:`None` (the default), a limit measured from the
        velocities of the brightest line that are actually visible.
    percentile_max
        The percentile of the in-frame intensity mapped to the top of the
        intensity colormap.
    percentile_alpha
        The percentile of the in-frame intensity mapped to fully opaque in
        the Doppler maps.
    correct_transmission
        Whether to divide the intensity by the relative atmospheric
        transmission of each frame.
    interval
        The delay between frames in milliseconds.
    """
    num_line = a.num_line
    num_time = a.shape[a.axis_time]

    if labels is None:
        labels = [a.label(i) for i in range(num_line)]

    x0 = position[0].to_value(u.arcsec)
    y0 = position[1].to_value(u.arcsec)
    hw = halfwidth.to_value(u.arcsec)

    x_vertices = a.inputs.position.x.ndarray.to_value(u.arcsec)
    y_vertices = a.inputs.position.y.ndarray.to_value(u.arcsec)
    x_center = _centers(x_vertices)
    y_center = _centers(y_vertices)

    slice_x = slice(
        int(np.searchsorted(x_center, x0 - hw)),
        int(np.searchsorted(x_center, x0 + hw)) + 1,
    )
    slice_y = slice(
        int(np.searchsorted(y_center, y0 - hw)),
        int(np.searchsorted(y_center, y0 + hw)) + 1,
    )
    extent = (
        x_vertices[slice_x.start],
        x_vertices[slice_x.stop],
        y_vertices[slice_y.start],
        y_vertices[slice_y.stop],
    )

    intensity = a.intensity
    velocity = a.velocity_mean
    if correct_transmission:
        intensity = intensity / a._transmission
    unit_intensity = na.unit(intensity)

    intensity_frames = [
        [
            a._index_xy(intensity[{a.axis_line: i, a.axis_time: t}])[slice_x, slice_y]
            for t in range(num_time)
        ]
        for i in range(num_line)
    ]
    velocity_frames = [
        [
            a._index_xy(velocity[{a.axis_line: i, a.axis_time: t}].to(u.km / u.s))[
                slice_x, slice_y
            ]
            for t in range(num_time)
        ]
        for i in range(num_line)
    ]

    vmax = [
        np.nanpercentile(np.stack(frames), percentile_max)
        for frames in intensity_frames
    ]
    alpha_reference = [
        np.nanpercentile(np.stack(frames), percentile_alpha)
        for frames in intensity_frames
    ]

    # one limit across every line, so the panels stay comparable; measured
    # from the brightest line, whose velocities are the ones worth reading
    from ._level_4 import _limit_velocity

    index_reference = int(np.argmax([np.nanmax(v) for v in vmax]))
    limit = _limit_velocity(
        limit_velocity=limit_velocity,
        intensity_frames=intensity_frames[index_reference],
        velocity_frames=velocity_frames[index_reference],
        alpha_reference=alpha_reference[index_reference],
    )
    norm_velocity = matplotlib.colors.Normalize(vmin=-limit, vmax=limit)
    cmap_velocity = matplotlib.colormaps["RdBu_r"]

    time_esis = a.inputs.time.ndarray

    context = context if context is not None else {}
    context_panels = []
    for label, function in context.items():
        axis_tc = function.inputs.time.axes[0]
        axis_cx = function.inputs.position.x.axes[0]
        axis_cy = function.inputs.position.y.axes[0]

        cx = _centers(function.inputs.position.x.ndarray.to_value(u.arcsec))
        cy = _centers(function.inputs.position.y.ndarray.to_value(u.arcsec))
        csx = slice(
            int(np.searchsorted(cx, x0 - hw)),
            int(np.searchsorted(cx, x0 + hw)) + 1,
        )
        csy = slice(
            int(np.searchsorted(cy, y0 - hw)),
            int(np.searchsorted(cy, y0 + hw)) + 1,
        )
        time_context = function.inputs.time.ndarray
        matches = [int(np.argmin(np.abs((time_context - t).sec))) for t in time_esis]

        def _frame_context(function, axis_tc, axis_cx, axis_cy, match):
            """
            Extract one context frame with ``(x, y)`` leading axes.

            Parameters
            ----------
            function
                The context-image function.
            axis_tc
                The time axis of the context images.
            axis_cx
                The horizontal axis of the context images.
            axis_cy
                The vertical axis of the context images.
            match
                The time index of the frame to extract.
            """
            o = function.outputs[{axis_tc: match}]
            source = (o.axes.index(axis_cx), o.axes.index(axis_cy))
            return np.moveaxis(
                np.asarray(o.ndarray, dtype=float),
                source,
                (0, 1),
            )

        frames = [
            _frame_context(function, axis_tc, axis_cx, axis_cy, matches[t])[csx, csy]
            for t in range(num_time)
        ]
        cmap = "gray"
        if cmaps_context is not None and label in cmaps_context:
            cmap = cmaps_context[label]
        context_panels.append(
            dict(
                label=label,
                frames=frames,
                extent=(cx[csx][0], cx[csx][-1], cy[csy][0], cy[csy][-1]),
                norm=matplotlib.colors.PowerNorm(
                    gamma=0.5,
                    vmin=0,
                    vmax=np.nanpercentile(np.stack(frames), percentile_max),
                ),
                cmap=cmap,
            )
        )

    num_context = len(context_panels)
    num_rows = 2 + (1 if num_context > 0 else 0)
    width = max(num_line, 1)

    fig = plt.figure(
        figsize=(2.1 * width, 2.4 * num_rows + 0.8),
        constrained_layout=True,
    )
    ncols_grid = width * max(num_context, 1)
    gs = fig.add_gridspec(nrows=num_rows, ncols=ncols_grid)

    span_line = ncols_grid // num_line
    images_intensity = []
    images_doppler = []
    axes_doppler = []

    def rgba(i: int, t: int) -> np.ndarray:
        """
        Build the Doppler RGBA image of one line at one time.

        Parameters
        ----------
        i
            The index of the spectral line.
        t
            The index of the frame.
        """
        result = cmap_velocity(norm_velocity(velocity_frames[i][t].T))
        alpha = intensity_frames[i][t].T / alpha_reference[i]
        result[..., 3] = np.clip(np.nan_to_num(alpha), 0, 1)
        return result

    label_unit = (
        f"({unit_intensity:latex_inline})" if unit_intensity is not None else ""
    )
    for i in range(num_line):
        ax = fig.add_subplot(gs[0, i * span_line : (i + 1) * span_line])
        images_intensity.append(
            ax.imshow(
                intensity_frames[i][0].T,
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=vmax[i],
            )
        )
        ax.set_title(labels[i], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        colorbar = fig.colorbar(
            images_intensity[i],
            ax=ax,
            location="bottom",
            fraction=0.06,
            pad=0.03,
        )
        colorbar.ax.tick_params(labelsize=6)
        if i == 0:
            colorbar.set_label(f"intensity {label_unit}", fontsize=7)

        ax = fig.add_subplot(gs[1, i * span_line : (i + 1) * span_line])
        images_doppler.append(ax.imshow(rgba(i, 0), extent=extent, origin="lower"))
        axes_doppler.append(ax)
        ax.set_xticks([])
        ax.set_yticks([])

    images_context = []
    if num_context > 0:
        span_context = ncols_grid // num_context
        for k, panel in enumerate(context_panels):
            ax = fig.add_subplot(gs[2, k * span_context : (k + 1) * span_context])
            images_context.append(
                ax.imshow(
                    panel["frames"][0].T,
                    extent=panel["extent"],
                    origin="lower",
                    norm=panel["norm"],
                    cmap=panel["cmap"],
                )
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_title(panel["label"], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    title = fig.suptitle(str(time_esis[0]), fontsize=10)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm_velocity, cmap=cmap_velocity),
        ax=axes_doppler,
        label="mean Doppler velocity (km/s)",
        shrink=0.9,
    )

    def update(t: int) -> tuple:
        """
        Draw the frame with the given time index.

        Parameters
        ----------
        t
            The index of the frame to draw.
        """
        for i in range(num_line):
            images_intensity[i].set_data(intensity_frames[i][t].T)
            images_doppler[i].set_data(rgba(i, t))
        for k, panel in enumerate(context_panels):
            images_context[k].set_data(panel["frames"][t].T)
        title.set_text(str(time_esis[t]))
        return tuple(images_intensity + images_doppler + images_context)

    return matplotlib.animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=num_time,
        interval=interval,
    )

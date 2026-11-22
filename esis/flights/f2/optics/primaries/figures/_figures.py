import named_arrays as na
from .._metrology import _serial_numbers, _map, _path_directory

__all__ = [
    "figure_measurement",
]


def figure_measurement(
    serial_number: str,
    axis_x: str = "primary_x",
    axis_y: str = "primary_y",
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    Load Optimax's measurement of the figure of a primary mirror for the second flight.

    Optimax Systems polished two primary mirrors for the second flight,
    serial numbers SN2 and SN3, and measured the figure of each with a
    Nanomefos profilometer in September 2026.
    Which of them will fly is not decided yet.

    The map is downloaded from the
    `esis-data-f2 <https://github.com/esis-mission/esis-data-f2/releases/tag/metrology-v1.0>`_
    repository the first time it is needed, as the file which Optimax's
    software saved.

    The result is the height of the surface of the mirror relative to a
    paraboloid, with piston, tilt, and power removed, as a function of the
    position of the center of each pixel.
    The map covers the whole clear aperture of the mirror, center included,
    and the pixels outside it are NaN.
    The positions are about the center of the map, with :math:`x` to the
    right and :math:`y` up as Zygo's software shows it.
    The flats of the octagonal edge of the map are along :math:`x` and
    :math:`y`, so it lines up with the clear aperture of
    :func:`esis.flights.f2.optics.design` turned by 22.5 degrees.
    The comment in the header of the file says that the mirror was measured
    with its fiducial at 6 o'clock.
    Which way the mirror will be turned about its axis in the instrument is
    not known yet.

    Parameters
    ----------
    serial_number
        The serial number of the mirror, ``"SN2"`` or ``"SN3"``.
    axis_x
        The name of the logical axis along the columns of the map.
    axis_y
        The name of the logical axis along the rows of the map.

    Raises
    ------
    ValueError
        If `serial_number` is not the serial number of a mirror which was
        measured.

    Examples
    --------
    Plot the maps of both mirrors.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        from esis.flights.f2.optics import primaries

        fig, axs = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=(8, 4),
            constrained_layout=True,
        )
        for ax, serial_number in zip(axs, ["SN2", "SN3"]):
            figure = primaries.figures.figure_measurement(serial_number)
            mappable = na.plt.pcolormesh(
                figure.inputs.x.to(u.mm).value,
                figure.inputs.y.to(u.mm).value,
                C=figure.outputs.to(u.nm).value,
                ax=ax,
                cmap="RdBu_r",
                vmin=-10,
                vmax=10,
            )
            ax.set_aspect("equal")
            ax.set_title(serial_number)
            ax.set_xlabel("$x$ (mm)")
        axs[0].set_ylabel("$y$ (mm)")
        fig.colorbar(mappable.ndarray.item(), ax=axs, label="height (nm)")
    """
    if serial_number not in _serial_numbers:
        raise ValueError(
            f"serial_number must be one of {_serial_numbers}, got {serial_number!r}"
        )

    path = _path_directory() / f"primary/{serial_number}_3219CC ASP_IRR_UC.dat"

    return _map(path, axis_x=axis_x, axis_y=axis_y)

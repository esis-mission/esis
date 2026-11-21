import named_arrays as na
from .._metrology import _serial_numbers, _map, _path_directory

__all__ = [
    "finish_measurement",
]

_objectives = ("1.4X", "20X")
"""The objectives of the interferometer through which the finish was measured."""


def finish_measurement(
    serial_number: str,
    objective: str,
    axis_x: str = "finish_x",
    axis_y: str = "finish_y",
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    Load Optimax's measurement of the finish of a primary mirror for the second flight.

    Optimax Systems measured a field of the finish of each of the primary
    mirrors for the second flight, serial numbers SN2 and SN3, through each of
    two objectives of a Zygo NexView white-light interferometer in September
    2026.
    The field of the 1.4X objective is 6.07 mm square, with pixels 5.92 µm
    apart, and the field of the 20X objective is 0.417 mm square, with pixels
    0.407 µm apart.

    The field is downloaded from the
    `esis-data-f2 <https://github.com/esis-mission/esis-data-f2/releases/tag/metrology-v1.0>`_
    repository the first time it is needed, as the file which Zygo's software
    saved.

    The result is the height of the surface of the mirror as it was measured,
    as a function of the position of the center of each pixel.
    Nothing is removed from it, so it still holds the piston and the tilt of
    the measurement, and the curvature of the mirror across the field.
    A pixel which was not measured is NaN.
    The positions are about the center of the field, with :math:`x` to the
    right and :math:`y` up as Zygo's software shows it.
    The files do not say where on the mirror each field is, or which way it
    is turned.

    Parameters
    ----------
    serial_number
        The serial number of the mirror, ``"SN2"`` or ``"SN3"``.
    objective
        The objective through which the field was measured, ``"1.4X"`` or
        ``"20X"``.
    axis_x
        The name of the logical axis along the columns of the field.
    axis_y
        The name of the logical axis along the rows of the field.

    Raises
    ------
    ValueError
        If `serial_number` is not the serial number of a mirror which was
        measured, or `objective` is not one of the objectives.

    Examples
    --------
    Plot the field of each mirror through the 20X objective, with a
    polynomial of the second degree, which takes out the piston, the tilt, and
    the curvature, removed from each.

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
            finish = primaries.finishes.finish_measurement(serial_number, "20X")
            fit = na.PolynomialFitFunctionArray.from_degree(
                inputs=finish.inputs,
                outputs=finish.outputs,
                degree=2,
                axis_polynomial=("finish_x", "finish_y"),
            )
            mappable = na.plt.pcolormesh(
                finish.inputs.x.to(u.um).value,
                finish.inputs.y.to(u.um).value,
                C=(finish.outputs - fit.predictions).to(u.nm).value,
                ax=ax,
                cmap="RdBu_r",
                vmin=-2,
                vmax=2,
            )
            ax.set_aspect("equal")
            ax.set_title(serial_number)
            ax.set_xlabel("$x$ (µm)")
        axs[0].set_ylabel("$y$ (µm)")
        fig.colorbar(mappable.ndarray.item(), ax=axs, label="height (nm)")
    """
    if serial_number not in _serial_numbers:
        raise ValueError(
            f"serial_number must be one of {_serial_numbers}, got {serial_number!r}"
        )
    if objective not in _objectives:
        raise ValueError(f"objective must be one of {_objectives}, got {objective!r}")

    name = f"{serial_number}_00a_MET01_NXV1_{objective}_1XZOOM.dat"
    path = _path_directory() / "primary" / name

    return _map(path, axis_x=axis_x, axis_y=axis_y)

"""A synthetic solar scene from a velocity-resolved DEM."""

import pathlib
from typing import TYPE_CHECKING
import astropy.units as u
import named_arrays as na
import numpy as np
from .._scene_dem import scene_emission_measure

if TYPE_CHECKING:
    import zarr

__all__ = [
    "scene_vdem",
]

_abundances = {
    "Coronal_Chianti_2021": "sun_coronal_2021_chianti",
}
"""
The abundances a velocity-resolved DEM may name, and the names :mod:`fiasco`
gives the same abundances.
"""


def scene_vdem(
    path: str | pathlib.Path,
    wavelength_min: u.Quantity,
    wavelength_max: u.Quantity,
    velocity: na.AbstractScalar,
    pressure: u.Quantity,
    plate_scale: (
        None | u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
    ) = None,
    abundance: None | str = None,
    width_nonthermal: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    fraction_min: float = 0.005,
    ions: None | list[str] = None,
    axis_x: str = "x",
    axis_y: str = "y",
    axis_line: str = "line",
    axis_velocity: str = "velocity",
) -> na.FunctionArray[na.DopplerPositionalVectorArray, na.ScalarArray]:
    r"""
    Create a synthetic solar scene from a velocity-resolved DEM.

    A velocity-resolved DEM is the emission measure of each pixel of a
    simulation in bins of temperature and of line-of-sight velocity, in the
    format used to synthesize observations for the MUSE mission. The
    spectrum of the plasma in each bin is computed with
    :func:`scene_emission_measure`, as a Gaussian centered on the bin of
    velocity.

    Parameters
    ----------
    path
        The path to the zarr store of the DEM. It holds ``vdem``, the
        emission measure along ``logT``, ``vdop``, ``x``, and ``y``;
        ``logT``, the logarithm of the temperature of each bin; ``vbins``,
        the vertices of the bins of line-of-sight velocity; ``x`` and ``y``,
        the positions of the pixels along each axis; and, optionally,
        ``Abundance``, the name of the abundances of the simulation. Each
        array has its unit in the attribute ``units``. Without one,
        ``vbins`` is taken to be in kilometers per second, and ``x`` and
        ``y`` in centimeters, the units of the format, but ``vdem`` has to
        have one. The names of the axes of ``vdem`` are those in its
        metadata, ``dimension_names``, in a zarr v3 store, or in its
        attribute ``_ARRAY_DIMENSIONS`` in a zarr v2 store, as xarray writes
        them.
    wavelength_min
        The shortest wavelength of the lines to include.
    wavelength_max
        The longest wavelength of the lines to include.
    velocity
        The vertices of the bins of line-of-sight velocity each line is
        spread over, along `axis_velocity`.
    pressure
        The constant pressure, :math:`n_e T`, the lines are computed at. The
        DEM does not record the density of the plasma.
    plate_scale
        The angle each pixel of the simulation spans, the same along both
        axes, or along each axis as a vector. If :obj:`None` (the default),
        the angle the pitch of the pixels along each axis spans seen from
        one astronomical unit away. A larger angle stretches the scene, as
        when each pixel of the simulation is to fall on one pixel of a
        detector.
    abundance
        The abundances the lines are computed with, by their name in
        :mod:`fiasco`. If :obj:`None` (the default), those named by the
        array ``Abundance`` of the DEM.
    width_nonthermal
        The standard deviation of the unresolved nonthermal velocities of the
        plasma in each bin of the DEM, beyond its spread across the bin,
        see :func:`utu.spectrum.synthesize`. It is added in quadrature to
        the width the bin gives, see the notes.
    fraction_min
        The least fraction of the radiance of all the lines in the passband,
        averaged over the pixels, for a line to be included, see
        :func:`scene_emission_measure`.
    ions
        The ions to compute the lines of, see :func:`scene_emission_measure`.
    axis_x
        The name of the axis of the pixels in :math:`x`.
    axis_y
        The name of the axis of the pixels in :math:`y`.
    axis_line
        The name of the axis along the lines of the result.
    axis_velocity
        The name of the axis of `velocity`.

    Returns
    -------
    :
        The spectral radiance of the scene, along `axis_line`,
        `axis_velocity`, `axis_x`, and `axis_y`. The positions are the
        vertices of the pixels, centered on the middle of the simulation,
        and increase along `axis_x` and `axis_y`, whichever way the
        positions of the store run.

    Raises
    ------
    ValueError
        If `abundance` is :obj:`None` and the DEM names no abundances, or
        names ones :mod:`fiasco` is not known to have; if the axes of
        ``vdem`` are not named, or are not ``logT``, ``vdop``, ``x``, and
        ``y``; if ``vdem`` has no unit; or if `plate_scale` is :obj:`None`
        and the DEM is a single pixel, whose angle cannot be found from the
        positions of its pixels.

    Notes
    -----
    The plasma in each bin of line-of-sight velocity is taken to be spread
    evenly across the bin, and is approximated by a Gaussian of the same
    variance: centered on the bin, with a nonthermal width of the width of
    the bin over :math:`\sqrt{12}`, added in quadrature to
    `width_nonthermal`.

    The pitch of the pixels along each axis is the distance between the
    first and the last of them over the number of steps between them, as a
    magnitude: the pixels are taken to be evenly spaced. A simulation one
    pixel wide along an axis is taken to have the pitch of the other axis
    along it too.

    The format records neither the sign of the velocity nor whether the
    emission measure is :math:`\int n_e n_H \, dh` or
    :math:`\int n_e^2 \, dh`. The velocity is taken to be positive away from
    the observer, and the emission measure to be :math:`\int n_e n_H \, dh`,
    which is what the contribution functions of :mod:`fiasco` go with.
    """
    # zarr takes nearly a fifth of a second to import, and only this function
    # reads it, so it is imported here rather than with esis.
    import zarr

    store = zarr.open_group(path, mode="r")

    if abundance is None:
        if "Abundance" not in store:
            raise ValueError(
                "The DEM does not name the abundances it was simulated with, so "
                "they have to be given as `abundance`."
            )
        name = str(store["Abundance"][()])
        if name not in _abundances:
            raise ValueError(
                f"The abundances of the DEM, {name!r}, are not ones fiasco is "
                f"known to have, so they have to be given as `abundance`."
            )
        abundance = _abundances[name]

    axis_temperature = "_temperature"
    axis_shift = "_shift"

    vdem = store["vdem"]
    axes = dict(
        logT=axis_temperature,
        vdop=axis_shift,
        x=axis_x,
        y=axis_y,
    )
    dimensions = _dimensions(vdem)
    if sorted(dimensions) != sorted(axes):
        raise ValueError(
            f"The axes of the DEM must be {tuple(axes)}, in any order, "
            f"got {dimensions}."
        )

    # The unit is attached without a copy, which keeps the precision of the
    # store, single for a simulation. utu.spectrum.synthesize sums it in that
    # precision, so it is never copied into double precision.
    emission_measure = na.ScalarArray(
        ndarray=_read(vdem, unit=None),
        axes=tuple(axes[a] for a in dimensions),
    )

    temperature = na.ScalarArray(
        ndarray=10 ** store["logT"][:].astype(float) * u.K,
        axes=axis_temperature,
    )

    vbins = na.ScalarArray(
        ndarray=_read(store["vbins"], unit=u.km / u.s),
        axes=axis_shift,
    )

    if plate_scale is None:
        plate_scale = _angle_pixel(
            x=na.ScalarArray(_read(store["x"], unit=u.cm), axes=axis_x),
            y=na.ScalarArray(_read(store["y"], unit=u.cm), axes=axis_y),
            axis_x=axis_x,
            axis_y=axis_y,
        )

    num_x = emission_measure.shape[axis_x]
    num_y = emission_measure.shape[axis_y]
    position = na.Cartesian2dVectorArray(
        x=na.arange(0, num_x + 1, axis=axis_x) - num_x / 2,
        y=na.arange(0, num_y + 1, axis=axis_y) - num_y / 2,
    )
    position = position * plate_scale

    # the spread of plasma evenly across each bin, and the nonthermal width
    # within it
    width_bin = vbins.volume_cell(axis_shift) / np.sqrt(12)
    width_nonthermal = np.sqrt(np.square(width_bin) + np.square(width_nonthermal))

    return scene_emission_measure(
        emission_measure=emission_measure,
        temperature=temperature,
        position=position,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        velocity=velocity,
        abundance=abundance,
        pressure=pressure,
        shift=vbins.cell_centers(axis_shift),
        width_nonthermal=width_nonthermal,
        fraction_min=fraction_min,
        ions=ions,
        axis_temperature=axis_temperature,
        axis_shift=axis_shift,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_line=axis_line,
        axis_velocity=axis_velocity,
    )


def _dimensions(array: "zarr.Array") -> tuple[str, ...]:
    """
    Return the names of the axes of an array of a zarr store.

    A zarr v3 store keeps them in the metadata of the array, and a zarr v2
    store, as xarray writes it, in the attribute ``_ARRAY_DIMENSIONS``.

    Parameters
    ----------
    array
        The array to name the axes of.

    Raises
    ------
    ValueError
        If the array names its axes in neither place.
    """
    names = getattr(array.metadata, "dimension_names", None)
    if names is None:
        names = array.attrs.get("_ARRAY_DIMENSIONS")
    if names is None:
        raise ValueError(
            f"The array {array.basename!r} of the DEM does not name its axes, "
            f"in its metadata or in the attribute '_ARRAY_DIMENSIONS'."
        )
    return tuple(names)


def _read(array: "zarr.Array", unit: None | u.UnitBase) -> u.Quantity:
    """
    Read an array of a zarr store, with the unit in its attribute ``units``.

    The unit is attached without a copy, so the array keeps the precision it
    is stored in.

    Parameters
    ----------
    array
        The array to read.
    unit
        The unit of the array if it has no attribute ``units``, or
        :obj:`None` if it must have one.

    Raises
    ------
    ValueError
        If the array has no unit and `unit` is :obj:`None`.
    """
    name = array.attrs.get("units")
    if name is not None:
        unit = u.Unit(name)
    elif unit is None:
        raise ValueError(
            f"The array {array.basename!r} of the DEM must have its unit in the "
            f"attribute 'units'."
        )
    return array[:] << unit


def _angle_pixel(
    x: na.AbstractScalar,
    y: na.AbstractScalar,
    axis_x: str,
    axis_y: str,
) -> na.Cartesian2dVectorArray:
    """
    Return the angle each pixel spans along each axis, from one AU away.

    The pitch along each axis is the distance between its first and last
    pixels over the number of steps between them, as a magnitude, so that
    positions which decrease along an axis do not turn the scene over. A
    simulation one pixel wide along an axis is taken to have the pitch of
    the other axis along it too.

    Parameters
    ----------
    x
        The position of each pixel in :math:`x`, along `axis_x`.
    y
        The position of each pixel in :math:`y`, along `axis_y`.
    axis_x
        The name of the axis of `x`.
    axis_y
        The name of the axis of `y`.

    Raises
    ------
    ValueError
        If the simulation is a single pixel.
    """
    angle = dict()
    for component, a, axis in [("x", x, axis_x), ("y", y, axis_y)]:
        num = a.shape[axis]
        if num > 1:
            pitch = np.abs(a[{axis: -1}] - a[{axis: 0}]) / (num - 1)
            angle[component] = (pitch / u.au).to(
                u.arcsec,
                equivalencies=u.dimensionless_angles(),
            )

    if not angle:
        raise ValueError(
            "The DEM is a single pixel, so the angle it spans cannot be found "
            "from the positions of its pixels, and has to be given as "
            "`plate_scale`."
        )

    return na.Cartesian2dVectorArray(
        x=angle.get("x", angle.get("y")),
        y=angle.get("y", angle.get("x")),
    )

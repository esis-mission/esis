"""A synthetic solar scene from a velocity-resolved DEM."""

import pathlib
import astropy.units as u
import named_arrays as na
import numpy as np
import zarr
from .._scene_dem import scene_emission_measure

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
    plate_scale: None | u.Quantity = None,
    abundance: None | str = None,
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
        emission measure along ``logT``, ``vdop``, ``x``, and ``y``, with its
        unit in the attribute ``units``; ``logT``, the logarithm of the
        temperature of each bin; ``vbins``, the vertices of the bins of
        line-of-sight velocity in kilometers per second; ``x``, the position
        of each pixel in centimeters; and ``Abundance``, the abundances of
        the simulation.
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
        The angle each pixel of the simulation spans. If :obj:`None` (the
        default), the angle it spans seen from one astronomical unit away.
        A larger angle stretches the scene, as when each pixel of the
        simulation is to fall on one pixel of a detector.
    abundance
        The abundances the lines are computed with, by their name in
        :mod:`fiasco`. If :obj:`None` (the default), those named by the
        attribute ``Abundance`` of the DEM.
    fraction_min
        The least fraction of the radiance of all the lines in the passband,
        averaged over the scene, for a line to be included, see
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
        vertices of the pixels, centered on the middle of the simulation.

    Raises
    ------
    ValueError
        If `abundance` is :obj:`None` and the abundances the DEM names are not
        ones :mod:`fiasco` is known to have.

    Notes
    -----
    The plasma in each bin of line-of-sight velocity is taken to be spread
    evenly across the bin, and is approximated by a Gaussian of the same
    variance: centered on the bin, with a nonthermal width of the width of
    the bin over :math:`\sqrt{12}`.

    The format records neither the sign of the velocity nor whether the
    emission measure is :math:`\int n_e n_H \, dh` or
    :math:`\int n_e^2 \, dh`. The velocity is taken to be positive away from
    the observer, and the emission measure to be :math:`\int n_e n_H \, dh`,
    which is what the contribution functions of :mod:`fiasco` go with.
    """
    store = zarr.open_group(path, mode="r")

    if abundance is None:
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
    axes = tuple(axes[a] for a in vdem.attrs["_ARRAY_DIMENSIONS"])

    # The unit is attached without a copy, which keeps the precision of the
    # store, single for a simulation, and so half the memory.
    emission_measure = na.ScalarArray(
        ndarray=vdem[:] << u.Unit(vdem.attrs["units"]),
        axes=axes,
    )

    temperature = na.ScalarArray(
        ndarray=10 ** store["logT"][:].astype(float) * u.K,
        axes=axis_temperature,
    )

    vbins = na.ScalarArray(
        ndarray=store["vbins"][:] * u.km / u.s,
        axes=axis_shift,
    )

    x = store["x"][:] * u.cm
    pitch = x[1] - x[0]
    if plate_scale is None:
        plate_scale = (pitch / u.au).to(
            u.arcsec, equivalencies=u.dimensionless_angles()
        )

    num_x = emission_measure.shape[axis_x]
    num_y = emission_measure.shape[axis_y]
    position = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.arange(num_x + 1) - num_x / 2, axes=axis_x),
        y=na.ScalarArray(np.arange(num_y + 1) - num_y / 2, axes=axis_y),
    )
    position = position * plate_scale

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
        width_nonthermal=vbins.volume_cell(axis_shift) / np.sqrt(12),
        fraction_min=fraction_min,
        ions=ions,
        axis_temperature=axis_temperature,
        axis_shift=axis_shift,
        axis_line=axis_line,
        axis_velocity=axis_velocity,
    )

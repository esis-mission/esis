"""A synthetic solar scene from the differential emission measure of AIA images."""

from typing import Literal
import astropy.units as u
import named_arrays as na
import numpy as np
import fiasco
import sdo
import utu
import esis
from esis._caching import versions

__all__ = [
    "scene_dem",
]

abundance = "sun_coronal_1992_feldman_ext"
"""
The abundances the lines are computed with, by their name in :mod:`fiasco`.

These are the abundances of both tables of :func:`sdo.aia.temperature_response`,
which the DEM is inverted against. A line computed with any other abundances
would be the line of some other plasma than the one the DEM describes.
"""

pressure = 1e15 * u.K / u.cm**3
"""
The constant pressure, :math:`n_e T`, the lines are computed at.

The pressure of both tables of :func:`sdo.aia.temperature_response`, for the
same reason as :data:`abundance`.
"""


def scene_dem(
    filtergram: sdo.aia.Filtergram,
    wavelength_min: u.Quantity,
    wavelength_max: u.Quantity,
    velocity: na.AbstractScalar,
    shift: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    width_nonthermal: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    fraction_min: float = 0.005,
    ions: None | list[str] = None,
    emissivity: Literal["SolarSoft", "CHIANTI 11"] = "CHIANTI 11",
    temperature_min: u.Quantity = 10**5.5 * u.K,
    temperature_max: u.Quantity = 10**7.5 * u.K,
    axis_line: str = "line",
    axis_velocity: str = "velocity",
) -> na.FunctionArray[na.TemporalDopplerPositionalVectorArray, na.ScalarArray]:
    r"""
    Create a synthetic solar scene from the DEM of a set of AIA images.

    The differential emission measure (DEM) of each pixel is inverted from
    the images, and the spectrum of the plasma it describes is computed from
    an atomic database.

    The DEM of each pixel is inverted from the six optically-thin EUV channels
    of AIA with :func:`utu.dem.plowman`, against the temperature response of
    :func:`sdo.aia.temperature_response`. The spectrum of the plasma it
    describes is then computed from the CHIANTI atomic database, through
    :func:`utu.spectrum.contribution_function`, for every line between
    `wavelength_min` and `wavelength_max` bright enough to matter, and each
    line is spread into a profile over bins of velocity with
    :func:`utu.spectrum.synthesize`.

    Unlike :func:`scene_filtergram`, which stands an AIA channel in for each
    spectral line, every line here is emitted by the plasma at the
    temperatures it forms at, so lines which form at different temperatures
    show different structures, and every line in the passband is included,
    not only the ones being observed.

    Parameters
    ----------
    filtergram
        The AIA images to invert, one in each of the 94, 131, 171, 193, 211,
        and 335 angstrom channels along ``filtergram.axis_wavelength``.
        The channels must share one grid of positions, as the images
        registered by :func:`sdo.aia.prep` do. Every other axis, such as
        that of time, is a separate scene.
    wavelength_min
        The shortest wavelength of the lines to include.
    wavelength_max
        The longest wavelength of the lines to include.
    velocity
        The vertices of the bins of line-of-sight velocity each line is
        spread over, along `axis_velocity`.
    shift
        The line-of-sight velocity of the plasma, see
        :func:`utu.spectrum.synthesize`.
    width_nonthermal
        The standard deviation of the unresolved nonthermal velocities of the
        plasma, see :func:`utu.spectrum.synthesize`.
    fraction_min
        The least fraction of the radiance of all the lines in the passband,
        averaged over the scene, for a line to be included. The default
        keeps the lines brighter than half a percent of the total.
    ions
        The ions to compute the lines of, named as :mod:`fiasco` names them.
        If :obj:`None` (the default), every ion with a line between
        `wavelength_min` and `wavelength_max`, found with
        :func:`utu.spectrum.ions`. There are dozens of them in a passband tens
        of angstroms wide, many with only faint lines, so naming the ones that
        matter saves most of the time the contribution functions take.
    emissivity
        The emissivity of the temperature response the DEM is inverted
        against, see :func:`sdo.aia.temperature_response`. The default is
        computed from the same version of CHIANTI as the lines, if
        :mod:`fiasco` reads that version.
    temperature_min
        The lowest temperature of the DEM.
    temperature_max
        The highest temperature of the DEM.
    axis_line
        The name of the axis along the lines of the result.
    axis_velocity
        The name of the axis of `velocity`.

    Returns
    -------
    :
        The spectral radiance of the scene, along `axis_line`,
        `axis_velocity`, and the axes of the images other than that of the
        channels. The positions are those of the AIA pixels, the time is that
        of the earliest image in each set, and the wavelengths are the
        vertices of the bins of velocity of each line, beside its rest
        wavelength.

    Notes
    -----
    The lines are computed with the abundances and the pressure the
    temperature response was computed with, the extended coronal abundances
    of Feldman (1992), ``"sun_coronal_1992_feldman_ext"`` in :mod:`fiasco`,
    and a constant pressure :math:`n_e T = 10^{15} \, \mathrm{K \, cm^{-3}}`,
    so that they come from the same plasma as the DEM.

    :func:`utu.dem.plowman` takes the DEM to be piecewise linear in
    :math:`\log_{10} T` between the temperatures of the response, and
    integrates it against the response exactly. The lines are integrated
    against the DEM in the same way, so the emission measure each temperature
    stands for is the DEM weighted by the mass matrix of those triangle
    functions. A pixel whose DEM could not be found emits nothing.

    The contribution functions take minutes to compute, so they are cached
    with ``esis.memory``.
    """
    axis_wavelength = filtergram.axis_wavelength
    axis_temperature = "_temperature"

    counts = filtergram.outputs
    intensity = counts / filtergram.timedelta / u.pix
    uncertainty = sdo.aia.uncertainty(
        counts=counts,
        wavelength=filtergram.inputs.wavelength,
    )
    uncertainty = uncertainty / filtergram.timedelta / u.pix

    time = filtergram.inputs.time

    response = sdo.aia.temperature_response(
        wavelength=filtergram.inputs.wavelength,
        time=time.ndarray.min(),
        eve=True,
        axis_wavelength=axis_wavelength,
        axis_temperature=axis_temperature,
        emissivity=emissivity,
    )
    t = response.inputs
    response = response[(t >= temperature_min) & (t <= temperature_max)]

    dem, chi2 = utu.dem.plowman(
        intensity=intensity,
        uncertainty=uncertainty,
        response=response,
        axis_channel=axis_wavelength,
        axis_temperature=axis_temperature,
    )

    temperature = dem.inputs

    emission_measure = _emission_measure(dem, axis_temperature)
    emission_measure = np.where(chi2 >= 0, emission_measure, 0 * emission_measure)

    contribution, mass = _lines(
        temperature=temperature.ndarray,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        ions=ions,
        abundance=abundance,
        pressure=pressure,
        versions=versions("fiasco", "utu") | dict(chianti=_chianti_version()),
        axis_temperature=axis_temperature,
        axis_line=axis_line,
    )

    # the brightness of each line from the average plasma in the scene, as a
    # fraction of all of them
    axis_pixel = tuple(set(na.shape(emission_measure)) - {axis_temperature})
    radiance = contribution.outputs * emission_measure.mean(axis_pixel)
    radiance = radiance.sum(axis_temperature)
    fraction = radiance / radiance.sum(axis_line)
    where = fraction > fraction_min

    result = utu.spectrum.synthesize(
        contribution=contribution[where],
        temperature=temperature,
        mass=mass[where],
        emission_measure=emission_measure,
        velocity=velocity,
        shift=shift,
        width_nonthermal=width_nonthermal,
        axis_temperature=axis_temperature,
        axis_velocity=axis_velocity,
        axis=axis_line,
    )

    # The earliest image of each set. :class:`astropy.time.Time` has no
    # minimum along a named axis, so it is found from the dates as numbers.
    time = time.explicit
    mjd = na.ScalarArray(time.ndarray.mjd, axes=time.axes)
    time = time[np.argmin(mjd, axis=axis_wavelength)]

    return na.FunctionArray(
        inputs=na.TemporalDopplerPositionalVectorArray(
            time=time,
            wavelength=result.inputs.wavelength,
            wavelength_rest=result.inputs.wavelength_rest,
            position=filtergram.inputs.position[{axis_wavelength: 0}],
        ),
        outputs=result.outputs,
    )


def _emission_measure(
    dem: na.FunctionArray[na.AbstractScalar, na.AbstractScalar],
    axis: str,
) -> na.AbstractScalar:
    r"""
    Weight a piecewise-linear DEM by the mass matrix of its temperatures.

    The result is the emission measure each temperature stands for.

    A DEM which is linear in :math:`\log_{10} T` between its temperatures is
    a sum of triangle functions, one at each temperature, and so is a
    contribution function sampled at the same temperatures. The integral of
    their product is :math:`G^T M D`, where :math:`M` is the mass matrix of
    the triangles, tridiagonal, and :math:`MD` is the emission measure each
    temperature stands for. At an interior temperature with neighbors
    :math:`\Delta` away on either side it is
    :math:`\Delta (D_{i-1} + 4 D_i + D_{i+1}) / 6`, and at either end it is
    half what it would be with a neighbor on both sides.

    This is the integral :func:`utu.dem.plowman` fits the intensities with,
    so a channel of AIA computed from these emission measures and its
    temperature response is the intensity the inversion found.

    Parameters
    ----------
    dem
        The DEM per unit :math:`\log_{10} T`, as :func:`utu.dem.plowman`
        returns it.
    axis
        The name of the axis of temperature.
    """
    logt = np.log10(dem.inputs.to(u.K).value)
    d = dem.outputs

    first = {axis: slice(None, -1)}
    last = {axis: slice(1, None)}

    # the width of the interval below and above each temperature, zero
    # beyond either end
    step = logt[last] - logt[first]
    below = na.concatenate([0 * step[{axis: slice(0, 1)}], step], axis=axis)
    above = na.concatenate([step, 0 * step[{axis: slice(0, 1)}]], axis=axis)

    # the neighbor below and above each temperature, any value beyond either
    # end, since its interval there is zero
    d_below = na.concatenate([d[{axis: slice(0, 1)}], d[first]], axis=axis)
    d_above = na.concatenate([d[last], d[{axis: slice(-1, None)}]], axis=axis)

    return (below * (2 * d + d_below) + above * (2 * d + d_above)) / 6


def _chianti_version() -> str:
    """Return the version of the CHIANTI database :mod:`fiasco` reads."""
    from fiasco.util import read_chianti_version

    return str(read_chianti_version(fiasco.defaults["ascii_dbase_root"]))


@esis.memory.cache
def _lines(
    temperature: u.Quantity,
    wavelength_min: u.Quantity,
    wavelength_max: u.Quantity,
    ions: None | list[str],
    abundance: str,
    pressure: u.Quantity,
    versions: dict[str, None | str],
    axis_temperature: str,
    axis_line: str,
) -> tuple[na.FunctionArray, na.ScalarArray]:
    """
    Compute and remember the contribution functions of the lines in a passband.

    They take minutes, so they are cached. Everything they depend on is an
    argument, so that it is part of the key the cache is looked up by.
    ``versions`` is not read here: it holds the versions of :mod:`fiasco`,
    :mod:`utu`, and the CHIANTI database, any of which can change the result.

    Returns the contribution functions along `axis_temperature` and
    `axis_line`, as :func:`utu.spectrum.contribution_function` returns them,
    and the mass of the ion of each line.

    Parameters
    ----------
    temperature
        The temperatures to compute the contribution functions at.
    wavelength_min
        The shortest wavelength of the lines to compute.
    wavelength_max
        The longest wavelength of the lines to compute.
    ions
        The ions to compute the lines of, or :obj:`None` for every ion with a
        line between `wavelength_min` and `wavelength_max`.
    abundance
        The abundances, by their name in :mod:`fiasco`.
    pressure
        The constant pressure of the plasma.
    versions
        The versions of the packages and the database the result depends on.
    axis_temperature
        The name of the axis of temperature.
    axis_line
        The name of the axis along the lines.
    """
    t = na.ScalarArray(temperature, axes=axis_temperature)

    # the ratio of protons to electrons depends on nothing but the
    # temperature, and computing it walks the whole database, so it is
    # computed once for every ion
    ratio = fiasco.proton_electron_ratio(temperature)

    contributions = []
    masses = []

    if ions is None:
        ions = utu.spectrum.ions(
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            abundance=abundance,
        )

    for name in ions:
        try:
            ion = fiasco.Ion(name, temperature, abundance=abundance)
            g = utu.spectrum.contribution_function(
                ion=ion,
                density=pressure / t,
                axis_temperature=axis_temperature,
                axis=axis_line,
                proton_electron_ratio=ratio,
            )
        except Exception:  # pragma: nocover
            # an ion whose atomic model the database cannot complete, as in
            # utu.spectrum.lines
            continue

        w = g.inputs
        g = g[(w > wavelength_min) & (w < wavelength_max)]

        contributions.append(g)
        masses.append(ion.mass * na.ScalarArray(np.ones(g.inputs.size), axis_line))

    contribution = na.FunctionArray(
        inputs=na.concatenate([g.inputs for g in contributions], axis=axis_line),
        outputs=na.concatenate([g.outputs for g in contributions], axis=axis_line),
    )

    mass = na.concatenate(masses, axis=axis_line)

    return contribution, mass

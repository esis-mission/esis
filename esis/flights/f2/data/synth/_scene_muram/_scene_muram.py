import pathlib
import astropy.units as u
import named_arrays as na
import esis
from esis.flights.f1.data._fits._fits import _path_cache, _retrieve

__all__ = [
    "path_muram",
    "scene_muram",
]

_name = "VDEM_Quadrupolar_x_1380000_M6.6_B1.6.zarr"
"""The name of the zarr store of the velocity-resolved DEM."""

_url = (
    "https://download.hao.ucar.edu/pub/rempel/data/MUSE/MUSE_Flare_Models"
    f"/Quadrupolar_M6.6/VDEM/{_name}.tar"
)
"""Where the archive of the velocity-resolved DEM is published."""

_hash = "sha256:03517fff80e486cbc3e6a8a05a7db20fd2f5cf19b882a4d967cb3cafe1a05d69"
"""The hash of the archive of the velocity-resolved DEM."""

pressure = 3e15 * u.K / u.cm**3
"""
The constant pressure, :math:`n_e T`, the lines of :func:`scene_muram` are
computed at.

The pressure of the active region DEM of CHIANTI. The velocity-resolved DEM
does not record the density of the plasma.
"""


def path_muram() -> pathlib.Path:
    """
    Return the path to the velocity-resolved DEM of :func:`scene_muram`.

    It is a zarr store.

    The store is downloaded the first time it is needed, and unpacked into
    :func:`esis.flights.f1.data._fits._fits._path_cache`.
    """
    _retrieve(
        url=_url,
        known_hash=_hash,
        fname=f"{_name}.tar",
        path=_path_cache(),
        extract_dir="muram",
    )
    return _path_cache() / "muram" / _name


def _plate_scale(instrument: esis.optics.Instrument) -> na.AbstractScalar:
    """
    Return the angle one pixel of the detector spans across the dispersion.

    The component across the dispersion, along the y axis of the sensor, of
    :meth:`optika.systems.SequentialSystem.plate_scale`, which is measured
    from a raytrace, in the middle of the passband of the instrument. It
    varies by a third of a percent across the passband.

    Along the dispersion, the anamorphic magnification of the grating and the
    tilt of the detector stretch the image further, so a pixel spans a
    smaller angle along it. The channels of ESIS-II disperse at different
    azimuths, so no one angle on the sky is one pixel of the detector along
    the dispersion in all of them.

    Parameters
    ----------
    instrument
        A single channel of ESIS-II, such as
        :func:`esis.flights.f2.optics.design_single`.
    """
    wavelength = (instrument.wavelength_min + instrument.wavelength_max) / 2
    result = instrument.system.plate_scale(wavelength).y * u.pix
    return result.to(u.arcsec)


def scene_muram(
    velocity: None | na.AbstractScalar = None,
    width_nonthermal: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    fraction_min: float = 0.005,
    ions: None | list[str] = None,
    axis_x: str = "x",
    axis_y: str = "y",
    axis_line: str = "line",
    axis_velocity: str = "velocity",
) -> na.FunctionArray[na.DopplerPositionalVectorArray, na.ScalarArray]:
    r"""
    Create a synthetic scene in the ESIS-II passband from a MURaM simulation.

    The scene is the velocity-resolved DEM of a radiative MHD simulation of
    an active region by Rempel et al. (2023), made with MURaM, 79,273
    seconds into the run, before its M6.6 flare, when it is as bright as a
    GOES B1.6 flare. It is 196.6 megameters across, in 1024 pixels on a side.

    This plugs the passband of :func:`esis.flights.f2.optics.design_single`
    into :func:`esis.data.synth.scene_vdem`, with the DEM downloaded from
    the High Altitude Observatory.

    Each pixel of the simulation is stretched, along both axes, to the angle
    on the sky one pixel of the detector of ESIS-II spans across the
    dispersion: the width of a pixel over the effective focal length of the
    instrument, the focal length of the primary mirror times
    :attr:`~esis.optics.Instrument.magnification`. That is about
    :math:`0.48''`, 1.8 times the angle a pixel of the simulation spans from
    one astronomical unit away, so the scene is :math:`491''` across and
    fills the field of view. Along the dispersion, the anamorphic
    magnification of the grating and the tilt of the detector stretch the
    image a few percent further, so there a pixel of the simulation spans a
    little more than one pixel of the detector. The six channels disperse at
    different azimuths, so no one angle could make it one pixel along the
    dispersion in all of them.

    Parameters
    ----------
    velocity
        The vertices of the bins of line-of-sight velocity each line is
        spread over. If :obj:`None` (the default), bins
        :math:`10\,\text{km}\,\text{s}^{-1}` wide from :math:`-400` to
        :math:`300\,\text{km}\,\text{s}^{-1}`, which hold every bin of the
        DEM and the thermal width of the hottest lines beyond them.
    width_nonthermal
        The standard deviation of the unresolved nonthermal velocities of the
        plasma in each bin of the DEM, beyond its spread across the bin, see
        :func:`esis.data.synth.scene_vdem`.
    fraction_min
        The least fraction of the radiance of the passband a line must carry
        to be included, see :func:`esis.data.synth.scene_emission_measure`.
    ions
        The ions to compute the lines of, see
        :func:`esis.data.synth.scene_emission_measure`.
    axis_x
        The name of the axis of the pixels in :math:`x`.
    axis_y
        The name of the axis of the pixels in :math:`y`.
    axis_line
        The name of the axis along the lines.
    axis_velocity
        The name of the axis along the bins of velocity.

    Notes
    -----
    The lines are computed with the abundances of the simulation, the
    coronal abundances of CHIANTI, ``"sun_coronal_2021_chianti"`` in
    :mod:`fiasco`, at the constant pressure of the active region DEM of
    CHIANTI, :math:`n_e T = 3 \times 10^{15} \, \mathrm{K \, cm^{-3}}`. The
    DEM does not record the density of the plasma.

    The DEM is a 330 megabyte download, and takes 3.3 gigabytes of memory in
    single precision, which it is summed in without a copy. The scene takes
    4 bytes for each line in each bin of velocity in each of the million
    pixels, so it can be computed for a few bins of velocity at a time, by
    passing their vertices as `velocity`.
    """
    if velocity is None:
        velocity = na.linspace(-400, 300, axis=axis_velocity, num=71) * u.km / u.s

    instrument = esis.flights.f2.optics.design_single(num_distribution=0)

    return esis.data.synth.scene_vdem(
        path=path_muram(),
        wavelength_min=instrument.wavelength_min,
        wavelength_max=instrument.wavelength_max,
        velocity=velocity,
        pressure=pressure,
        plate_scale=_plate_scale(instrument),
        width_nonthermal=width_nonthermal,
        fraction_min=fraction_min,
        ions=ions,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_line=axis_line,
        axis_velocity=axis_velocity,
    )

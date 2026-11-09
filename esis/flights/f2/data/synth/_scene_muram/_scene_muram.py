import pathlib
import astropy.units as u
import named_arrays as na
import numpy as np
import esis
from esis.flights.f1.data._fits._fits import _path_cache, _retrieve

__all__ = [
    "path_muram",
    "plate_scale",
    "pressure",
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
The constant pressure, :math:`n_e T`, the lines are computed at.

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


def plate_scale() -> u.Quantity:
    """
    Return the angle each pixel of the detector of ESIS-II spans on the sky.

    The width of a pixel over the effective focal length of the instrument,
    the focal length of the primary mirror times the magnification of the
    grating, of :func:`esis.flights.f2.optics.design_single`.
    """
    instrument = esis.flights.f2.optics.design_single(num_distribution=0)
    focal_length = instrument.primary_mirror.sag.focal_length
    focal_length = focal_length * instrument.magnification
    result = instrument.camera.sensor.width_pixel / np.abs(focal_length)
    result = result.to(u.arcsec, equivalencies=u.dimensionless_angles())
    if isinstance(result, na.AbstractArray):
        result = result.ndarray
    return result


def scene_muram(
    velocity: None | na.AbstractScalar = None,
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

    Each pixel of the simulation is stretched to span one pixel of the
    detector of ESIS-II, :func:`plate_scale`, so that the simulation fills
    the field of view. That is about 1.8 times its true size, and the scene
    is :math:`492''` across.

    Parameters
    ----------
    velocity
        The vertices of the bins of line-of-sight velocity each line is
        spread over. If :obj:`None` (the default), bins
        :math:`10\,\text{km}\,\text{s}^{-1}` wide from :math:`-400` to
        :math:`300\,\text{km}\,\text{s}^{-1}`, which hold every bin of the
        DEM and the thermal width of the hottest lines beyond them.
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
    :mod:`fiasco`, at the constant pressure :data:`pressure`.

    The DEM is 330 megabytes, and takes 3.1 gigabytes of memory, in single
    precision. The scene takes 8 bytes for each line in each bin of velocity
    in each of the million pixels, so it can be computed for a few bins of
    velocity at a time, by passing their vertices as `velocity`.
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
        plate_scale=plate_scale(),
        fraction_min=fraction_min,
        ions=ions,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_line=axis_line,
        axis_velocity=axis_velocity,
    )

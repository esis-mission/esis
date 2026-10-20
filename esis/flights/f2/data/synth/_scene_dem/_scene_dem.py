import astropy.units as u
import astropy.time
import named_arrays as na
import numpy as np
import sdo
import esis

__all__ = [
    "scene_dem",
]

_channels = [94, 131, 171, 193, 211, 335] * u.AA
"""The optically-thin EUV channels of AIA, which the DEM is inverted from."""


def scene_dem(
    time: str | astropy.time.Time = "2016-07-15T00:00:00",
    center: na.AbstractCartesian2dVectorArray = na.Cartesian2dVectorArray(-30, 70)
    * u.arcsec,
    width: u.Quantity = 500 * u.arcsec,
    velocity: None | na.AbstractScalar = None,
    shift: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    width_nonthermal: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    fraction_min: float = 0.005,
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    axis_line: str = "line",
    axis_velocity: str = "velocity",
) -> na.FunctionArray[na.TemporalDopplerPositionalVectorArray, na.ScalarArray]:
    r"""
    Create a synthetic scene in the ESIS-II passband from AIA images.

    The scene is made from the differential emission measure of the images.

    This plugs the passband of :func:`esis.flights.f2.optics.design_single`
    into :func:`esis.data.synth.scene_dem`, with AIA images downloaded from
    the JSOC.

    The default scene is the pair of active regions NOAA 12565 and 12567 on
    2016 July 15, which Jacob D. Parker chose for the first simulated ESIS-II
    images, in a box :math:`500''` across around the boresight he assumed for
    them. It is wider than the field of view of ESIS-II, so the field stop of
    the instrument model decides what is seen.

    Parameters
    ----------
    time
        The time of the AIA images. The images closest to it are used.
    center
        The helioprojective coordinates of the center of the scene, which is
        taken to be the boresight of ESIS-II: the positions of the result are
        offsets from it.
    width
        The width of the square of the Sun to include.
    velocity
        The vertices of the bins of line-of-sight velocity each line is
        spread over. If :obj:`None` (the default), bins
        :math:`10\,\text{km}\,\text{s}^{-1}` wide from
        :math:`-100` to :math:`100\,\text{km}\,\text{s}^{-1}`, about half the
        Doppler shift of one ESIS-II pixel.
    shift
        The line-of-sight velocity of the plasma, see
        :func:`utu.spectrum.synthesize`.
    width_nonthermal
        The standard deviation of the unresolved nonthermal velocities of the
        plasma, see :func:`utu.spectrum.synthesize`.
    fraction_min
        The least fraction of the radiance of the passband a line must carry
        to be included, see :func:`esis.data.synth.scene_dem`.
    axis_detector_x
        The name of the axis of the AIA pixels in :math:`x`.
    axis_detector_y
        The name of the axis of the AIA pixels in :math:`y`.
    axis_line
        The name of the axis along the lines.
    axis_velocity
        The name of the axis along the bins of velocity.

    Raises
    ------
    ValueError
        If no AIA pixel has its center within the square.
    """
    if velocity is None:
        velocity = na.linspace(-100, 100, axis=axis_velocity, num=21) * u.km / u.s

    images = sdo.aia.open(
        time_start=time,
        wavelength=na.ScalarArray(_channels, axes="wavelength"),
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        register=True,
    )
    images = images[{images.axis_time: 0}]

    # The pixels whose centers lie within the square. Registration puts every
    # channel on one grid of positions, so those of the first channel do for
    # all of them.
    position = images.inputs.position[{images.axis_wavelength: 0}]
    position = position.cell_centers()
    offset = position - center
    inside = (np.abs(offset.x) <= width / 2) & (np.abs(offset.y) <= width / 2)
    if not np.any(inside):
        raise ValueError(
            f"No AIA pixel has its center within {width} of {center}, so the "
            f"scene would be empty."
        )
    index_x = np.nonzero(inside.any(axis_detector_y).ndarray)[0]
    index_y = np.nonzero(inside.any(axis_detector_x).ndarray)[0]
    images = images[
        {
            axis_detector_x: slice(index_x[0], index_x[-1] + 1),
            axis_detector_y: slice(index_y[0], index_y[-1] + 1),
        }
    ]

    instrument = esis.flights.f2.optics.design_single(num_distribution=0)

    result = esis.data.synth.scene_dem(
        filtergram=images,
        wavelength_min=instrument.wavelength_min,
        wavelength_max=instrument.wavelength_max,
        velocity=velocity,
        shift=shift,
        width_nonthermal=width_nonthermal,
        fraction_min=fraction_min,
        axis_line=axis_line,
        axis_velocity=axis_velocity,
    )

    result.inputs.position = result.inputs.position - center

    return result

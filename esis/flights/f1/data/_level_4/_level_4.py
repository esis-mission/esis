import astropy.units as u
import named_arrays as na
import esis
from ... import optics
from ... import spectrum
from .. import level_1

__all__ = [
    "level_4",
]


@esis.memory.cache(ignore=["verbose"])
def level_4(
    pitch_scene: None | u.Quantity = None,
    num_velocity: None | int = None,
    num_iteration: int = 100,
    verbose: bool = False,
) -> esis.data.Level_4:
    """
    Invert the Level-1 images and process them to the Level-4 stage.

    This function takes the result of :func:`level_1`,
    creates a new instance of :class:`esis.data.Level_4` using the
    :meth:`~esis.data.Level_4.from_level_1` classmethod,
    and then caches the result for future use.

    The reconstruction targets the three brightest lines in the ESIS
    passband — He I 584 Å, Mg X 610 Å, and O V 630 Å
    (:mod:`esis.flights.f1.spectrum`) — each with a ±200 km/s Doppler
    window, on a scene grid matching the plate scale of the instrument.
    The forward model is the fitted instrument returned by
    :func:`esis.flights.f1.optics.distortion_fit`,
    and the ``time=15`` frame, the frame the distortion fit was optimized
    against, is the reference frame of the inversion.

    Parameters
    ----------
    pitch_scene
        The spatial pitch of the scene grid.
        If :obj:`None` (the default), the spatial plate scale of the
        instrument, measured from the linearized optical system
        (about 0.73 arcsec).
    num_velocity
        The number of velocity bins in the Doppler window of each line.
        If :obj:`None` (the default), the number of bins that matches the
        spectral plate scale of the instrument, measured from the linearized
        optical system (about 17 km/s per bin).
    num_iteration
        The maximum number of MART iterations per frame.
    verbose
        Whether to print the convergence statistics of each frame.
        Ignored by the cache.

    Notes
    -----
    With the default grid, this function is only affordable on a machine
    with hundreds of gigabytes of memory: building the regridding weights
    of the forward model peaks at roughly four times the memory of the
    1.5-arcsec runs the pipeline was developed with.
    """
    wavelength_center = na.ScalarArray(
        ndarray=u.Quantity(
            [
                spectrum.He_I.wavelength,
                spectrum.Mg_X.wavelength,
                spectrum.O_V.wavelength,
            ]
        ),
        axes="line",
    )

    width_doppler = na.ScalarArray(
        ndarray=u.Quantity(
            [
                spectrum.He_I.width_doppler,
                spectrum.Mg_X.width_doppler,
                spectrum.O_V.width_doppler,
            ]
        ),
        axes="line",
    )

    return esis.data.Level_4.from_level_1(
        a=level_1(),
        wavelength_center=wavelength_center,
        width_doppler=width_doppler,
        instrument=optics.distortion_fit(num_distribution=0),
        pitch_scene=pitch_scene,
        num_velocity=num_velocity,
        index_time_reference=15,
        num_iteration=num_iteration,
        verbose=verbose,
    )

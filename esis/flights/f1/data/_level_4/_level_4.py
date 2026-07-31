import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from ... import optics
from ... import spectrum
from .. import level_1

__all__ = [
    "level_4",
    "level_4_frame",
    "level_4_parallel",
]


def _lines() -> tuple[na.AbstractScalarArray, na.AbstractScalarArray]:
    """
    Gather the rest wavelength and Doppler width of the six baseline lines.

    The six brightest disjoint lines in the ESIS passband — He I 584,
    O III 600, O IV 608, Mg X 610, Mg X 625, and O V 630 Å
    (:mod:`esis.flights.f1.spectrum`) — in order of increasing wavelength.
    """
    lines = [
        spectrum.He_I,
        spectrum.O_III,
        spectrum.O_IV,
        spectrum.Mg_X,
        spectrum.Mg_X_625,
        spectrum.O_V,
    ]

    wavelength_center = na.ScalarArray(
        ndarray=u.Quantity([line.wavelength for line in lines]),
        axes="line",
    )

    width_doppler = na.ScalarArray(
        ndarray=u.Quantity([line.width_doppler for line in lines]),
        axes="line",
    )

    return wavelength_center, width_doppler


def _line_labels() -> list[str]:
    """
    Gather the display label of each baseline line, annotating blends.

    The Mg X 610 window also contains the O IV 609.83 line (roughly 30% of
    the blend), which is 20 km/s from Mg X 609.79 and has no window of its
    own.
    """
    return [
        "He I 584",
        "O III 600",
        "O IV 608",
        "Mg X 610 + O IV",
        "Mg X 625",
        "O V 630",
    ]


def _grid_velocity(
    pitch_velocity: None | u.Quantity,
    limit_velocity: u.Quantity,
) -> tuple[None | int, u.Quantity]:
    """
    Compute the number of velocity bins and the exact window half-width.

    The window is widened to the nearest whole number of bins covering
    ±`limit_velocity`, so the bin width is exactly `pitch_velocity`.

    Parameters
    ----------
    pitch_velocity
        The width of the velocity bins in the Doppler window of each line.
        If :obj:`None`, the number of bins is left to be measured from the
        linearized optical system.
    limit_velocity
        The minimum half-width of the Doppler window of each line.
    """
    if pitch_velocity is None:
        return None, limit_velocity
    ratio = (2 * limit_velocity / pitch_velocity).to_value(u.dimensionless_unscaled)
    num_velocity = int(np.ceil(ratio))
    return num_velocity, num_velocity * pitch_velocity / 2


@esis.memory.cache(ignore=["verbose"])
def level_4(
    pitch_scene: None | u.Quantity = 0.75 * u.arcsec,
    pitch_velocity: None | u.Quantity = 17.5 * u.km / u.s,
    limit_velocity: u.Quantity = 200 * u.km / u.s,
    num_iteration: int = 100,
    verbose: bool = False,
) -> esis.data.Level_4:
    """
    Invert the Level-1 images and process them to the Level-4 stage.

    This function takes the result of :func:`level_1`,
    creates a new instance of :class:`esis.data.Level_4` using the
    :meth:`~esis.data.Level_4.from_level_1` classmethod,
    and then caches the result for future use.

    The reconstruction targets the six brightest disjoint lines in the ESIS
    passband — He I 584, O III 600, O IV 608, Mg X 610, Mg X 625, and
    O V 630 Å (:mod:`esis.flights.f1.spectrum`) — each with a Doppler window
    of at least ±`limit_velocity` around its rest wavelength.
    The O IV 609.83 Å line, 20 km/s from Mg X 609.79 Å, is not given its own
    window: its emission (roughly 30% of the blend) lands in the Mg X 610
    window.
    The forward model is the fitted instrument returned by
    :func:`esis.flights.f1.optics.distortion_fit`,
    and the ``time=15`` frame, the frame the distortion fit was optimized
    against, is the reference frame of the inversion.

    The frames are inverted sequentially, each warm-started from the solution
    of its neighbor in a chain outward from the reference frame.
    For the embarrassingly-parallel alternative, where every frame starts
    from the same Gaussian seed, see :func:`level_4_frame` and
    :func:`level_4_parallel`.

    Parameters
    ----------
    pitch_scene
        The spatial pitch of the scene grid.
        If :obj:`None`, the spatial plate scale of the instrument, measured
        from the linearized optical system.
    pitch_velocity
        The width of the velocity bins in the Doppler window of each line.
        The window is widened to the nearest whole number of bins covering
        ±`limit_velocity`, so the bin width is exactly this value.
        If :obj:`None`, the spectral plate scale of the instrument, measured
        from the linearized optical system.
    limit_velocity
        The minimum half-width of the Doppler window of each line.
    num_iteration
        The maximum number of MART iterations per frame.
    verbose
        Whether to print the convergence statistics of each frame.
        Ignored by the cache.

    Notes
    -----
    With the default grid, this function is only affordable on a machine
    with hundreds of gigabytes of memory: building the regridding weights
    of the forward model peaks well beyond the capacity of a workstation.
    """
    wavelength_center, width_doppler = _lines()
    num_velocity, limit_velocity = _grid_velocity(pitch_velocity, limit_velocity)

    return esis.data.Level_4.from_level_1(
        a=level_1(),
        wavelength_center=wavelength_center,
        width_doppler=width_doppler,
        instrument=optics.distortion_fit(num_distribution=0),
        limit_velocity=limit_velocity,
        num_velocity=num_velocity,
        pitch_scene=pitch_scene,
        index_time_reference=15,
        num_iteration=num_iteration,
        verbose=verbose,
    )


@esis.memory.cache(ignore=["verbose"])
def level_4_frame(
    index_time: int,
    weights_shared: bool = True,
    pitch_scene: None | u.Quantity = 0.75 * u.arcsec,
    pitch_velocity: None | u.Quantity = 17.5 * u.km / u.s,
    limit_velocity: u.Quantity = 200 * u.km / u.s,
    num_iteration: int = 100,
    verbose: bool = False,
) -> esis.data.Level_4:
    """
    Invert a single frame of the Level-1 images to the Level-4 stage.

    The unit of work of the parallel inversion: each frame is inverted
    independently from the same spatially-uniform Gaussian seed (rescaled to
    the frame's own total signal), so every frame of the flight can run
    concurrently on a cluster.
    :func:`level_4_parallel` assembles the cached single-frame results into
    the full time-dependent product.

    The shadow mask is measured once from the full flight
    (:meth:`esis.data.Level_1.where_shadow`), so every frame is inverted
    with the same mask.

    Parameters
    ----------
    index_time
        The index of the frame to invert.
    weights_shared
        Whether to invert with the regridding weights of the ``time=15``
        reference frame (the default, and the validated approach: the
        payload pointing drift moves the scene but barely changes the
        field-stop-to-detector mapping) or to rebuild the weights from the
        fitted per-frame pointing of this frame
        (:func:`esis.flights.f1.optics.distortion_fit` with ``axis_time``).
    pitch_scene
        The spatial pitch of the scene grid.
        If :obj:`None`, the spatial plate scale of the instrument, measured
        from the linearized optical system.
    pitch_velocity
        The width of the velocity bins in the Doppler window of each line.
        The window is widened to the nearest whole number of bins covering
        ±`limit_velocity`, so the bin width is exactly this value.
        If :obj:`None`, the spectral plate scale of the instrument, measured
        from the linearized optical system.
    limit_velocity
        The minimum half-width of the Doppler window of each line.
    num_iteration
        The maximum number of MART iterations.
    verbose
        Whether to print the convergence statistics of the frame.
        Ignored by the cache.
    """
    wavelength_center, width_doppler = _lines()
    num_velocity, limit_velocity = _grid_velocity(pitch_velocity, limit_velocity)

    a = level_1()

    if weights_shared:
        instrument = optics.distortion_fit(num_distribution=0)
    else:
        instrument = optics.distortion_fit(
            num_distribution=0,
            axis_time=a.axis_time,
        )
        index = {a.axis_time: index_time}
        instrument.pitch = instrument.pitch[index]
        instrument.yaw = instrument.yaw[index]
        instrument.roll = instrument.roll[index]

    where_shadow = a.where_shadow()

    a = a[{a.axis_time: slice(index_time, index_time + 1)}]

    return esis.data.Level_4.from_level_1(
        a=a,
        wavelength_center=wavelength_center,
        width_doppler=width_doppler,
        instrument=instrument,
        limit_velocity=limit_velocity,
        num_velocity=num_velocity,
        pitch_scene=pitch_scene,
        where_shadow=where_shadow,
        index_time_reference=0,
        num_iteration=num_iteration,
        verbose=verbose,
    )


@esis.memory.cache(ignore=["verbose"])
def level_4_parallel(
    weights_shared: bool = True,
    pitch_scene: None | u.Quantity = 0.75 * u.arcsec,
    pitch_velocity: None | u.Quantity = 17.5 * u.km / u.s,
    limit_velocity: u.Quantity = 200 * u.km / u.s,
    num_iteration: int = 100,
    verbose: bool = False,
) -> esis.data.Level_4:
    """
    Assemble the single-frame inversions into the full Level-4 product.

    Every frame of the flight is inverted independently with
    :func:`level_4_frame` — from the same Gaussian seed, instead of the
    warm-start chain of :func:`level_4` — and the cached results are stacked
    along the time axis.
    On a cluster, run the frames concurrently (one job per frame, so each
    call is a cache hit here) and then call this function to gather them.

    Parameters
    ----------
    weights_shared
        Whether the frames were inverted with the shared reference-frame
        weights or with per-frame weights; see :func:`level_4_frame`.
    pitch_scene
        The spatial pitch of the scene grid.
        If :obj:`None`, the spatial plate scale of the instrument, measured
        from the linearized optical system.
    pitch_velocity
        The width of the velocity bins in the Doppler window of each line.
        The window is widened to the nearest whole number of bins covering
        ±`limit_velocity`, so the bin width is exactly this value.
        If :obj:`None`, the spectral plate scale of the instrument, measured
        from the linearized optical system.
    limit_velocity
        The minimum half-width of the Doppler window of each line.
    num_iteration
        The maximum number of MART iterations per frame.
    verbose
        Whether to print the convergence statistics of each frame.
        Ignored by the cache.
    """
    a = level_1()
    axis_time = a.axis_time
    num_time = a.shape[axis_time]

    frames = [
        level_4_frame(
            index_time=index,
            weights_shared=weights_shared,
            pitch_scene=pitch_scene,
            pitch_velocity=pitch_velocity,
            limit_velocity=limit_velocity,
            num_iteration=num_iteration,
            verbose=verbose,
        )
        for index in range(num_time)
    ]

    index = {axis_time: 0}
    reference = frames[0]

    return esis.data.Level_4(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=na.stack(
                [frame.inputs.time[index] for frame in frames],
                axis=axis_time,
            ),
            wavelength=reference.inputs.wavelength,
            position=reference.inputs.position,
        ),
        outputs=na.stack(
            [frame.outputs[index] for frame in frames],
            axis=axis_time,
        ),
        instrument=reference.instrument,
        wavelength_center=reference.wavelength_center,
        label_line=_line_labels(),
        num_velocity=reference.num_velocity,
        mean_chi_squared=na.stack(
            [frame.mean_chi_squared[index] for frame in frames],
            axis=axis_time,
        ),
        num_iteration=na.stack(
            [frame.num_iteration[index] for frame in frames],
            axis=axis_time,
        ),
        factor_norm=na.stack(
            [frame.factor_norm[index] for frame in frames],
            axis=axis_time,
        ),
        where_shadow=reference.where_shadow,
        axis_time=axis_time,
        axis_wavelength=reference.axis_wavelength,
        axis_x=reference.axis_x,
        axis_y=reference.axis_y,
        axis_line=reference.axis_line,
    )

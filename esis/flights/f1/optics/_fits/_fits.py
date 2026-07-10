"""Reproduce the ESIS-I distortion fits from the flight data."""

import pathlib
import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na
import optika
import esis

__all__ = [
    "fit_distortion_reference",
    "fit_distortion_pointing",
]


def _idealized(instrument: esis.optics.Instrument) -> esis.optics.Instrument:
    """
    Replace the material models of an instrument with ideal ones.

    The distortion fit measures geometry, not throughput, so the multilayer
    coatings, filter, and sensor quantum efficiency are replaced with ideal
    equivalents to save raytracing time. The uncertainty wrappers of the
    measured ruling coefficients are also stripped, so that the distortion
    parameters are plain scalars.

    Parameters
    ----------
    instrument
        The instrument model to idealize (modified in place and returned).
    """
    instrument.grating.material = optika.materials.Mirror()
    instrument.primary_mirror.material = optika.materials.Mirror()
    instrument.filter.material = None
    instrument.camera.sensor.material = optika.sensors.materials.IdealSensorMaterial()
    coefficients = instrument.grating.rulings.spacing.coefficients
    for k in list(coefficients):
        coefficients[k] = na.nominal(coefficients[k])
    return instrument


def _wavelength_lines() -> na.ScalarArray:  # pragma: nocover
    """Return the wavelengths of the three bright spectral lines seen by ESIS-I."""
    from esis.flights.f1.spectrum import He_I, Mg_X, O_V

    return na.ScalarArray(
        u.Quantity([He_I.wavelength, Mg_X.wavelength, O_V.wavelength]),
        axes="wavelength",
    )


def _pupil() -> na.AbstractCartesian2dVectorArray:  # pragma: nocover
    """Build the 2x2 pupil grid used by the production fits."""
    return na.Cartesian2dVectorLinearSpace(
        -0.25,
        0.25,
        axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
        num=2,
    )


def _frame(time: int, num_scene: int):  # pragma: nocover
    """
    Prepare the resampled AIA scene and the Level-1 observation of one frame.

    Parameters
    ----------
    time
        The frame index into :func:`esis.flights.f1.data.level_1`.
    num_scene
        The number of samples along each axis of the resampled scene.
    """
    scene_full = esis.flights.f1.data.synth.scene_aia()
    frame_l1 = esis.flights.f1.data.level_1()[dict(time=time)]

    time_exposure = frame_l1.inputs.time_start[dict(channel=0)].ndarray
    difference = (scene_full.inputs.time - time_exposure).mean("wavelength")
    frame_scene = scene_full[np.argmin(np.abs(difference))]

    inputs = na.TemporalSpectralPositionalVectorArray(
        time=frame_scene.inputs.time,
        wavelength=frame_scene.inputs.wavelength,
        position=na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(
                frame_scene.inputs.position.x.min(),
                frame_scene.inputs.position.x.max(),
                axis="detector_x",
                num=num_scene,
            ),
            y=na.ScalarLinearSpace(
                frame_scene.inputs.position.y.min(),
                frame_scene.inputs.position.y.max(),
                axis="detector_y",
                num=num_scene,
            ),
        ),
    )
    scene = frame_scene(
        inputs,
        axis=("detector_x", "detector_y"),
        method="conservative",
    )
    return scene, frame_l1.outputs.value


def _grids_reference() -> list[dict]:
    """
    Build the scan schedule of the from-scratch reference fit.

    Three repetitions of a wide capture round (with the strongly-coupled
    pairs scanned jointly), two shrinking refinement rounds, and a polish
    round which :func:`esis.optics.fit_distortion_scan` repeats until it
    stops improving.
    """
    capture = {
        ("pitch_grating", "pitch"): (
            np.linspace(-10, 10, 13) * u.arcmin,
            np.linspace(-60, 60, 13) * u.arcsec,
        ),
        ("yaw_grating", "yaw"): (
            np.linspace(-3, 3, 13) * u.arcmin,
            np.linspace(-60, 60, 13) * u.arcsec,
        ),
        "roll_grating": np.linspace(-2, 2, 17) * u.deg,
        "roll_field_stop": np.linspace(-2, 2, 17) * u.deg,
        "roll": np.linspace(-2, 2, 17) * u.deg,
        ("spacing_rulings", "displacement_primary"): (
            np.linspace(-2e-3, 2e-3, 13) * u.um,
            np.linspace(-20, 20, 13) * u.mm,
        ),
    }
    refine_1 = dict(
        pitch_grating=np.linspace(-2, 2, 13) * u.arcmin,
        yaw_grating=np.linspace(-0.6, 0.6, 13) * u.arcmin,
        pitch=np.linspace(-9, 9, 13) * u.arcsec,
        yaw=np.linspace(-9, 9, 13) * u.arcsec,
        roll_grating=np.linspace(-0.4, 0.4, 13) * u.deg,
        roll_field_stop=np.linspace(-0.4, 0.4, 13) * u.deg,
        roll=np.linspace(-0.4, 0.4, 13) * u.deg,
        spacing_rulings=np.linspace(-4e-4, 4e-4, 13) * u.um,
        displacement_primary=np.linspace(-4, 4, 13) * u.mm,
    )
    refine_2 = dict(
        pitch_grating=np.linspace(-0.4, 0.4, 11) * u.arcmin,
        yaw_grating=np.linspace(-0.12, 0.12, 11) * u.arcmin,
        pitch=np.linspace(-1.8, 1.8, 11) * u.arcsec,
        yaw=np.linspace(-1.8, 1.8, 11) * u.arcsec,
        roll_grating=np.linspace(-0.08, 0.08, 11) * u.deg,
        roll_field_stop=np.linspace(-0.08, 0.08, 11) * u.deg,
        roll=np.linspace(-0.08, 0.08, 11) * u.deg,
        spacing_rulings=np.linspace(-8e-5, 8e-5, 11) * u.um,
        displacement_primary=np.linspace(-0.8, 0.8, 11) * u.mm,
    )
    polish = dict(
        pitch_grating=np.linspace(-0.1, 0.1, 9) * u.arcmin,
        yaw_grating=np.linspace(-0.05, 0.05, 9) * u.arcmin,
        pitch=np.linspace(-0.6, 0.6, 9) * u.arcsec,
        yaw=np.linspace(-0.6, 0.6, 9) * u.arcsec,
        roll_grating=np.linspace(-0.02, 0.02, 9) * u.deg,
        roll_field_stop=np.linspace(-0.02, 0.02, 9) * u.deg,
        roll=np.linspace(-0.02, 0.02, 9) * u.deg,
        spacing_rulings=np.linspace(-3e-5, 3e-5, 9) * u.um,
        displacement_primary=np.linspace(-0.3, 0.3, 9) * u.mm,
    )
    return 3 * [capture] + [refine_1, refine_2, polish]


def _grids_pointing() -> list[dict]:
    """Build the per-frame scan schedule of the production pointing series."""
    return [
        dict(
            pitch=np.linspace(-10, 10, 21) * u.arcsec,
            yaw=np.linspace(-10, 10, 21) * u.arcsec,
            roll=np.linspace(-0.4, 0.4, 11) * u.deg,
        ),
        dict(
            pitch=np.linspace(-1, 1, 11) * u.arcsec,
            yaw=np.linspace(-1, 1, 11) * u.arcsec,
            roll=np.linspace(-0.05, 0.05, 11) * u.deg,
        ),
    ]


def fit_distortion_reference(
    instrument: None | esis.optics.Instrument = None,
    time: int = 15,
    num_scene: int = 401,
    grids: None | list[dict] = None,
    sigma_psf: None | float = 1.0,
    seed: int = 0,
    tolerance: None | float = 3e-4,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> esis.optics.DistortionParameters:  # pragma: nocover
    """
    Fit the per-channel reference distortion parameters from scratch.

    Reproduces (and lets anyone verify) the fit stored in
    ``_data/distortion_reference.ecsv``: all nine per-channel degrees of
    freedom of :class:`esis.optics.DistortionParameters` are fit to one
    Level-1 frame with :func:`esis.optics.fit_distortion_scan`, reading
    every channel's peak from a single pass of coherent scans.

    Expect several thousand merit evaluations at a few seconds each: hours
    of runtime on a workstation. Requires the ESIS Level-1 data and the AIA
    synthetic scene (downloaded and cached on first use).

    Parameters
    ----------
    instrument
        The starting instrument model. If :obj:`None`, the as-built model
        (:func:`esis.flights.f1.optics.as_built`, which carries the measured
        grating radii and ruling density) is used, idealized for raytracing
        speed.
    time
        The frame index of the reference frame.
    num_scene
        The number of samples along each axis of the resampled AIA scene.
    grids
        The scan schedule. If :obj:`None`, the from-scratch schedule (wide
        joint capture rounds, shrinking refinement, repeated polish) is used.
    sigma_psf
        The standard deviation, in detector pixels, of the Gaussian
        point-spread function convolved with the modeled image.
    seed
        The seed used to make each evaluation deterministic.
    tolerance
        The mean-correlation gain below which the repeated polish round
        stops.
    path
        If given, the fitted parameters are written to this path as an ECSV
        table, in the format of ``_data/distortion_reference.ecsv``.
    directory
        A directory where the scan curves and convergence history are
        logged. If :obj:`None`, the fit is not logged.

    Examples
    --------
    .. code-block:: python

        import esis

        parameters = esis.flights.f1.optics.fit_distortion_reference(
            path="distortion_reference.ecsv",
            directory="distortion_reference_scan",
        )
    """
    if instrument is None:
        instrument = _idealized(
            esis.flights.f1.optics.as_built(num_distribution=0),
        )
        instrument.wavelength = _wavelength_lines()

    if grids is None:
        grids = _grids_reference()

    scene, observation = _frame(time, num_scene)

    parameters = esis.optics.fit_distortion_scan(
        instrument=instrument,
        scene=scene,
        observation=observation,
        grids=grids,
        pupil=_pupil(),
        axis_wavelength="velocity",
        axis_field=("detector_x", "detector_y"),
        axis_channel="channel",
        sigma_psf=sigma_psf,
        seed=seed,
        tolerance=tolerance,
        directory=directory,
    )

    if path is not None:
        parameters.to_file(
            path,
            metadata=dict(
                description=(
                    "Best-fit ESIS-I distortion parameters, optimized "
                    f"against the time={time} frame of "
                    "esis.flights.f1.data.level_1()."
                ),
                provenance=(
                    "esis.flights.f1.optics.fit_distortion_reference("
                    f"time={time}, num_scene={num_scene}, "
                    f"sigma_psf={sigma_psf}, seed={seed})"
                ),
            ),
        )

    return parameters


def fit_distortion_pointing(
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    grids: None | list[dict] = None,
    sigma_psf: None | float = 1.0,
    seed: int = 0,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    workers: int = 1,
) -> astropy.table.QTable:  # pragma: nocover
    r"""
    Fit the per-frame payload pointing of the whole flight.

    Reproduces (and lets anyone verify) the time series stored in
    ``_data/distortion_pointing.ecsv``: for every frame of
    :func:`esis.flights.f1.data.level_1`, a coherent pitch/yaw/roll offset
    from the reference model is fit with
    :func:`esis.optics.fit_distortion_series` under a rigid-payload model
    (the same offset for all four channels).

    Expect roughly a hundred merit evaluations per frame at a few seconds
    each; the frames are independent, so `workers` parallelizes them
    perfectly. Requires the ESIS Level-1 data and the AIA synthetic scene
    (downloaded and cached on first use).

    Parameters
    ----------
    instrument
        The reference instrument model whose pointing is offset. If
        :obj:`None`, :func:`esis.flights.f1.optics.distortion_fit` is used,
        idealized for raytracing speed.
    num_scene
        The number of samples along each axis of the resampled AIA scenes.
    grids
        The per-frame scan schedule. If :obj:`None`, the production schedule
        (coarse :math:`\\pm 10''`, fine :math:`\\pm 1''`) is used.
    sigma_psf
        The standard deviation, in detector pixels, of the Gaussian
        point-spread function convolved with the modeled images.
    seed
        The seed used to make each evaluation deterministic.
    path
        If given, the fitted offsets are written to this path as an ECSV
        table, in the format of ``_data/distortion_pointing.ecsv``.
    directory
        A directory under which each frame's fit is logged.
        If :obj:`None`, the fits are not logged.
    workers
        The number of processes used to fit frames concurrently.

    Examples
    --------
    .. code-block:: python

        import esis

        pointing = esis.flights.f1.optics.fit_distortion_pointing(
            path="distortion_pointing.ecsv",
            workers=8,
        )
    """
    if instrument is None:
        instrument = _idealized(
            esis.flights.f1.optics.distortion_fit(num_distribution=0),
        )

    if grids is None:
        grids = _grids_pointing()

    observations = esis.flights.f1.data.level_1()
    num_time = na.shape(observations.outputs)["time"]

    frames = [_frame(t, num_scene) for t in range(num_time)]

    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    results = esis.optics.fit_distortion_series(
        instrument=instrument,
        scenes=[scene for scene, observation in frames],
        observations=[observation for scene, observation in frames],
        grids=grids,
        parameters=parameters,
        pupil=_pupil(),
        axis_wavelength="velocity",
        axis_field=("detector_x", "detector_y"),
        axis_channel="channel",
        coherent=True,
        sigma_psf=sigma_psf,
        seed=seed,
        directory=directory,
        workers=workers,
    )

    def offset(index: int, field: str, unit: u.Unit) -> float:
        delta = getattr(results[index], field) - getattr(parameters, field)
        delta = na.as_named_array(delta)
        if "channel" in na.shape(delta):
            delta = delta.mean("channel")
        return float(na.value(delta.to(unit)).ndarray)

    table = astropy.table.QTable(
        dict(
            frame=np.arange(num_time),
            pitch=[offset(t, "pitch", u.arcsec) for t in range(num_time)] * u.arcsec,
            yaw=[offset(t, "yaw", u.arcsec) for t in range(num_time)] * u.arcsec,
            roll=[offset(t, "roll", u.deg) for t in range(num_time)] * u.deg,
        )
    )
    table.meta.update(
        dict(
            description=(
                "Fitted per-frame payload pointing offsets of the ESIS-I "
                "flight, relative to the reference fit, one row per frame "
                "of esis.flights.f1.data.level_1(). The offsets are common "
                "to all four channels (a rigid-payload model)."
            ),
            provenance=(
                "esis.flights.f1.optics.fit_distortion_pointing("
                f"num_scene={num_scene}, sigma_psf={sigma_psf}, seed={seed})"
            ),
        )
    )

    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)

    return table

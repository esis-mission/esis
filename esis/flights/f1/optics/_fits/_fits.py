"""The reproducible distortion fit of the ESIS-I flight data."""

from __future__ import annotations
from typing import Callable
import copy
import dataclasses
import datetime
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

_SHARED = ("displacement_primary", "roll_field_stop", "pitch", "yaw")
"""
The parameters that belong to the instrument rather than to a channel.

One primary mirror, one field stop, one payload pointing: the channels must
agree on these, and a fit which lets them differ per channel is using them
as stand-ins for the camera placement.
"""

_LINES_ALIGNMENT = ("He I", "O V")
"""The bright, isolated lines the channels are aligned on."""

_SENSOR_PLACEMENT = (
    "roll_sensor",
    "pitch_sensor",
    "yaw_sensor",
    "x_sensor",
    "y_sensor",
)
"""
The orientation and in-plane position of the sensor.

A fit against the proxy scene cannot tell these from the pointing and the
grating, so the absolute stage holds them at their as-built values and they
move only once the shared optics are fixed, in the second stage and in the
internal alignment.  The focus-preserving distance is not among them: the
as-built gratings were focused with their measured radii, which changes the
magnification, and no other parameter can put it back.
"""


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
    # the as-built gratings were focused and aligned with uncertain measured
    # radii, which leaves their placement wrapped even without a distribution
    grating = instrument.grating
    grating.translation = na.nominal(grating.translation)
    grating.yaw = na.nominal(grating.yaw)
    grating.pitch = na.nominal(grating.pitch)
    grating.roll = na.nominal(grating.roll)
    return instrument


def _wavelength_lines() -> na.ScalarArray:
    """Return the wavelengths of the three bright spectral lines seen by ESIS-I."""
    from esis.flights.f1.spectrum import He_I, Mg_X, O_V

    return na.ScalarArray(
        u.Quantity([He_I.wavelength, Mg_X.wavelength, O_V.wavelength]),
        axes="wavelength",
    )


def _wavelengths_alignment() -> dict[str, u.Quantity]:
    """Return the lines the channels are aligned on, by name."""
    from esis.flights.f1.spectrum import He_I, O_V

    return {"He I": He_I.wavelength, "O V": O_V.wavelength}


def _pupil() -> na.AbstractCartesian2dVectorArray:
    """Build the 2x2 pupil grid used by the ray-traced cross-check."""
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


def _base(instrument: None | esis.optics.Instrument) -> esis.optics.Instrument:
    if instrument is None:
        instrument = _idealized(esis.flights.f1.optics.as_built(num_distribution=0))
        instrument.wavelength = _wavelength_lines()
    return instrument


def _logger(directory: None | str | pathlib.Path, name: str) -> Callable[[str], None]:
    path = None
    if directory is not None:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.log"
        path.write_text(f"--- {name} {datetime.datetime.now()} ---\n")

    def log(message: str) -> None:
        print(message, flush=True)
        if path is not None:
            with path.open("a") as f:
                f.write(message + "\n")

    return log


def _names_absolute(parameters: esis.optics.DistortionParameters) -> tuple[str, ...]:
    """Name the fields the absolute stage fits: everything but the sensor placement."""
    return tuple(
        f.name
        for f in dataclasses.fields(parameters)
        if f.name not in _SENSOR_PLACEMENT
    )


def _frames_by_axis(
    observation: na.AbstractScalar, num_channel: int
) -> list[np.ndarray]:
    """Each channel's frame as a plain ``[y, x]`` array, for the alignment."""
    return [
        np.asarray(
            na.value(observation[dict(channel=c)]).ndarray_aligned(
                ("detector_y", "detector_x")
            )
        )
        for c in range(num_channel)
    ]


def enforce_shared(
    parameters: list[esis.optics.DistortionParameters],
    shared: tuple[str, ...] = _SHARED,
) -> list[esis.optics.DistortionParameters]:
    """
    Set the shared parameters of every channel to their mean.

    Parameters
    ----------
    parameters
        The parameters of each channel.
    shared
        The names of the parameters that belong to the instrument.
    """
    result = [copy.deepcopy(p) for p in parameters]
    for name in shared:
        values = [getattr(p, name) for p in parameters]
        unit = na.unit(values[0])
        mean = np.mean(
            [float(na.value(na.as_named_array(v).to(unit)).ndarray) for v in values]
        )
        for p in result:
            setattr(p, name, mean * unit)
    return result


def _stack(
    parameters: list[esis.optics.DistortionParameters],
    axis: str,
) -> esis.optics.DistortionParameters:
    """Stack per-channel parameters along a logical axis."""
    fields = {}
    for field in dataclasses.fields(parameters[0]):
        values = [na.as_named_array(getattr(p, field.name)) for p in parameters]
        fields[field.name] = na.stack(values, axis=axis)
    return type(parameters[0])(**fields)


def fit_distortion_reference(
    instrument: None | esis.optics.Instrument = None,
    time: int = 15,
    num_scene: int = 401,
    device: None | str = None,
    workers: int = 1,
    channels: None | tuple[int, ...] = None,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    parameters: None | list[esis.optics.DistortionParameters] = None,
) -> esis.optics.DistortionParameters:  # pragma: nocover
    """
    Fit the reference distortion parameters of ESIS-I from the as-built model.

    Reproduces the values stored in ``_data/distortion_reference.ecsv`` from
    the instrument model and the flight data alone, in three stages.

    1.  **Absolute.**  Each channel is fit on its own to the AIA proxy scene
        with :class:`esis.optics.LinearMerit` and
        :func:`esis.optics.fit_distortion`: a seeded capture followed by a
        local polish.  About an hour per channel on one GPU.
    2.  **Shared.**  The primary displacement, the field-stop roll and the
        payload pitch and yaw belong to the instrument; they are set to
        their mean over the channels and each channel's own terms are
        polished again.  This costs a few thousandths in correlation and
        removes the degeneracy in which those terms stood in for the
        camera placement.
    3.  **Internal.**  The channels are aligned to one another on the sky
        plane with :func:`esis.optics.align_channels`, in the grating and
        camera terms only, to about a fifth of a pixel.  Two minutes.

    Requires the ESIS Level-1 data and the AIA synthetic scene (downloaded
    and cached on first use).

    Parameters
    ----------
    instrument
        The starting instrument model. If :obj:`None`, the as-built model
        (:func:`esis.flights.f1.optics.as_built`) is used, idealized for
        raytracing speed.
    time
        The frame index of the reference frame.
    num_scene
        The number of samples along each axis of the resampled AIA scene.
    device
        The device on which to build and apply the regridding weights, for
        example ``"cuda"``.
    workers
        The number of processes evaluating the population of the capture.
    channels
        The channels to fit in the absolute stage.  If :obj:`None`, every
        channel is fit; a subset lets the channels be fit by separate jobs,
        with `parameters` supplying the others.
    path
        If given, the fitted parameters are written to this path as an ECSV
        table, in the format of ``_data/distortion_reference.ecsv``.
    directory
        A directory where the progress of every stage is logged.
        If :obj:`None`, the fit is not logged.
    parameters
        The result of the absolute stage for every channel, if it was run
        by separate jobs; the absolute stage is then skipped for those
        channels.

    Raises
    ------
    ValueError
        If a channel has neither been fit here nor supplied in `parameters`.

    Examples
    --------
    .. code-block:: python

        import esis

        parameters = esis.flights.f1.optics.fit_distortion_reference(
            device="cuda",
            workers=6,
            path="distortion_reference.ecsv",
            directory="distortion_reference",
        )
    """
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    scene, observation = _frame(time, num_scene)
    log = _logger(directory, "reference")

    if channels is None:
        channels = tuple(range(num_channel))
    if parameters is None:
        parameters = [None] * num_channel
    parameters = list(parameters)

    # 1. absolute
    for c in channels:
        channel = instrument[dict(channel=c)]
        p0 = esis.optics.DistortionParameters.from_instrument(channel)
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=p0,
            scene=scene,
            observation=observation[dict(channel=c)],
            device=device,
        )
        log(f"channel {c}: absolute fit")
        parameters[c] = esis.optics.fit_distortion(
            objective=merit,
            parameters=p0,
            bounds=esis.flights.f1.optics.distortion_fit_bounds(p0),
            workers=workers,
            free=_names_absolute(p0),
            log=log,
        )
    if any(p is None for p in parameters):
        raise ValueError("every channel needs parameters before the shared stage")

    # 2. shared
    parameters = enforce_shared(parameters)
    names = [f.name for f in dataclasses.fields(parameters[0])]
    index_own = [i for i, n in enumerate(names) if n not in _SHARED]
    for c in range(num_channel):
        channel = instrument[dict(channel=c)]
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=parameters[c],
            scene=scene,
            observation=observation[dict(channel=c)],
            device=device,
        )
        lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters[c])
        lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
        x_full = na.pack(parameters[c]).ndarray

        def objective(y, x_full=x_full, merit=merit):
            x = x_full.copy()
            x[index_own] = y
            return merit(x)

        log(f"channel {c}: polish with the shared optics fixed")
        y, _, _ = esis.optics.polish(
            objective,
            x_full[index_own],
            lb[index_own],
            ub[index_own],
            scale=0.01,
            log=log,
        )
        x_full[index_own] = y
        parameters[c] = na.unpack(x_full, parameters[c])

    # 3. internal
    log("internal alignment")
    parameters, medians = esis.optics.align_channels(
        instruments=[instrument[dict(channel=c)] for c in range(num_channel)],
        parameters=parameters,
        frames=_frames_by_axis(observation, num_channel),
        scene_position=scene.inputs.position,
        wavelengths=_wavelengths_alignment(),
        log=log,
    )

    result = _stack(parameters, axis="channel")
    if path is not None:
        result.to_file(
            path,
            metadata=dict(
                description=(
                    "Best-fit ESIS-I distortion parameters, optimized "
                    f"against the time={time} frame of "
                    "esis.flights.f1.data.level_1()."
                ),
                provenance=(
                    "esis.flights.f1.optics.fit_distortion_reference("
                    f"time={time}, num_scene={num_scene}): a seeded capture and "
                    "polish of every channel against the AIA proxy scene, the "
                    "shared optics set to their mean, and the channels aligned to "
                    f"one another on the sky at {', '.join(_LINES_ALIGNMENT)} "
                    "(median shift vs channel 1: "
                    f"{', '.join(f'{m:.2f}' for m in medians)} px)"
                ),
            ),
        )
    return result


def fit_distortion_pointing(
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    device: None | str = None,
    frames: None | tuple[int, ...] = None,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> astropy.table.QTable:  # pragma: nocover
    r"""
    Fit the per-frame payload pointing of the whole flight.

    Reproduces the time series stored in ``_data/distortion_pointing.ecsv``:
    for every frame of :func:`esis.flights.f1.data.level_1`, a coherent
    pitch/yaw/roll offset from the reference model (the same offset for all
    four channels) is polished with :func:`esis.optics.polish` on the mean
    of the four channels' :class:`esis.optics.LinearMerit`.  The optics are
    otherwise held at the reference fit.

    Parameters
    ----------
    instrument
        The reference model. If :obj:`None`,
        :func:`esis.flights.f1.optics.distortion_fit` is used, idealized.
    num_scene
        The number of samples along each axis of the resampled AIA scene.
    device
        The device on which to build and apply the regridding weights.
    frames
        The frames to fit.  If :obj:`None`, every frame is fit.
    path
        If given, the offsets are written to this path as an ECSV table, in
        the format of ``_data/distortion_pointing.ecsv``.
    directory
        A directory where the progress of every frame is logged.
    """
    if instrument is None:
        instrument = _idealized(
            esis.flights.f1.optics.distortion_fit(num_distribution=0)
        )
        instrument.wavelength = _wavelength_lines()
    num_channel = instrument.camera.channel.shape["channel"]
    num_time = esis.flights.f1.data.level_1().shape["time"]
    if frames is None:
        frames = tuple(range(num_time))
    log = _logger(directory, "pointing")

    reference = [
        esis.optics.DistortionParameters.from_instrument(instrument[dict(channel=c)])
        for c in range(num_channel)
    ]
    names = [f.name for f in dataclasses.fields(reference[0])]
    index = [names.index(n) for n in ("pitch", "yaw", "roll")]

    rows = []
    for t in frames:
        scene, observation = _frame(t, num_scene)
        merits = [
            esis.optics.LinearMerit(
                instrument=instrument[dict(channel=c)],
                parameters=reference[c],
                scene=scene,
                observation=observation[dict(channel=c)],
                device=device,
            )
            for c in range(num_channel)
        ]
        x_full = [na.pack(p).ndarray for p in reference]

        def objective(y):
            values = []
            for c in range(num_channel):
                x = x_full[c].copy()
                x[index] = x[index] + y
                values.append(merits[c](x))
            return float(np.mean(values))

        lower, upper = esis.flights.f1.optics.distortion_fit_bounds(reference[0])
        half = 0.5 * (na.pack(upper).ndarray - na.pack(lower).ndarray)[index]
        log(f"frame {t}: start {-objective(np.zeros(3)):.4f}")
        y, fun, num = esis.optics.polish(
            objective, np.zeros(3), -half, half, scale=0.01, log=log
        )
        log(f"frame {t}: {-fun:.4f} after {num} evaluations, offset {y}")
        rows.append((t, y[0], y[1], y[2]))

    table = astropy.table.QTable(
        dict(
            frame=np.array([r[0] for r in rows]),
            pitch=np.array([r[1] for r in rows]) * u.arcsec,
            yaw=np.array([r[2] for r in rows]) * u.arcsec,
            roll=np.array([r[3] for r in rows]) * u.deg,
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
                f"num_scene={num_scene}): a local polish of the mean linear merit "
                "of the four channels in pitch, yaw and roll, per frame"
            ),
        )
    )
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table

"""The reproducible distortion fit of the ESIS-I flight data."""

from __future__ import annotations
from typing import Callable
import copy
import functools
import dataclasses
import datetime
import importlib
import importlib.metadata
import multiprocessing
import pathlib
import numpy as np
import astropy.units as u
import astropy.constants
import astropy.table
import named_arrays as na
import optika
import esis
from esis.optics._distortions import _fit
from esis.optics._distortions import _alignment

__all__ = [
    "measure_window_edges",
    "fit_distortion_reference",
    "realign_distortion_reference",
    "fit_defocus_history",
    "fit_distortion_pointing",
    "acceptance",
]

_SHARED = ("displacement_primary", "z_primary", "roll_field_stop", "pitch", "yaw")
"""
The parameters that belong to the instrument rather than to a channel.

One primary mirror, one field stop, one payload pointing: the channels must
agree on these, and a fit which lets them differ per channel is using them
as stand-ins for the camera placement.
"""

# the fields that do not move the mapping, which the alignment cannot use
_PHOTOMETRIC = ("degradation",)

_MODE_KEYS = (
    "translation",
    "magnification",
    "rotation",
    "anisotropy",
    "shear",
    "scatter",
    "residual",
    "correlation",
)
"""The columns of the acceptance's decomposition of the tile shifts."""

_FREE_ABSOLUTE = (
    "yaw_grating",
    "pitch_grating",
    "spacing_rulings",
    "pitch",
    "yaw",
    "z_sensor",
    "roll_sensor",
    "pitch_sensor",
    "yaw_sensor",
)
"""
The terms the absolute stage searches.

One frame determines eight combinations of the parameters of a channel:
two translations, the dispersion, two scales, a rotation, a shear and one
more.  These nine terms span them with no null direction, which is what
lets a global search land on the same solution from any seed; every
larger set adds directions the frame cannot see and captures unreliably.
"""

_WINDOW = ("yaw_grating", "pitch_grating", "roll_sensor")
"""
The terms the outline stage fits to the window edges.

They move the image of the field stop on the sensor without moving the
sky within it, which the merit cannot see and the edges can.
"""

_SKY = ("spacing_rulings", "pitch", "yaw", "z_sensor", "yaw_sensor", "pitch_sensor")
"""The terms the outline stage re-polishes against the merit."""

_FREE_SHARED = (
    "yaw_grating",
    "pitch_grating",
    "roll_grating",
    "spacing_rulings",
    "z_sensor",
    "roll_sensor",
    "pitch_sensor",
    "yaw_sensor",
    "x_sensor",
    "y_sensor",
)
"""
The terms of each channel polished with the shared optics fixed.

The grating placement and the whole camera placement: what lets one
channel's mapping differ from another's by a scale, an anisotropy and a
rotation, which the internal alignment measures.  The instrument roll is
not among them, see :attr:`esis.optics.DistortionParameters.roll`.
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

These were meant to stay at their as-built values until the internal
alignment, since against the proxy scene they are partly degenerate with
the pointing and the grating.  On the as-built model of the flight they
cannot: with them held, the absolute fit of three channels of four stalls
at 0.70 to 0.76 in correlation where freeing them reaches 0.78 to 0.82, and
freeing the focus-preserving distance alone does not close the gap, nor
does the design model in place of the as-built one.  The absolute stage
therefore fits the terms one frame determines, and the internal alignment still has the
last word on the sensor.
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


def _frame_times() -> list[dict[str, str | float]]:
    """
    Read the exposure start, end and length of every frame of the flight.

    Returns
    -------
    One record per frame of :func:`esis.flights.f1.data.level_1`, with the
    UTC ``start`` and ``end`` as ISO strings and the ``exposure`` in seconds.
    The channels expose together, so channel 0 speaks for all.
    """
    inputs = esis.flights.f1.data.level_1().inputs
    start = inputs.time_start[dict(channel=0)].ndarray
    end = inputs.time_end[dict(channel=0)].ndarray
    return [
        dict(
            start=str(s.isot)[:23],
            end=str(e.isot)[:23],
            exposure=float((e - s).to_value("s")),
        )
        for s, e in zip(start, end)
    ]


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
    """Name the fields the absolute stage fits, which is every field."""
    return tuple(f.name for f in dataclasses.fields(parameters))


def _polish_own(
    job: tuple,
) -> tuple[esis.optics.DistortionParameters, float, int]:  # pragma: nocover
    """
    Polish one channel's own terms with the shared optics fixed.

    A module-level function so that the channels can be sent to worker
    processes.

    Parameters
    ----------
    job
        The channel, its parameters, the scene, its frame, the device,
        the indices of the packed parameters that belong to the channel, and
        the merit to maximize.

    Returns
    -------
    The polished parameters, their correlation, and the number of
    evaluations.
    """
    instrument, parameters, scene, observation, device, index_own, merit = job
    merit = esis.optics.LinearMerit(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        device=device,
        merit=merit,
    )
    if merit.merit == "least_squares" and np.all(na.value(parameters.degradation) == 1):
        # a start that knows nothing about the level of the frame
        parameters = copy.copy(parameters)
        parameters.degradation = merit.estimate_degradation(parameters)
    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)
    lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
    x_full = na.pack(parameters).ndarray
    index = np.array(index_own)
    objective = _fit._Subset(merit, x_full, index)
    y, fun, num = esis.optics.polish(
        objective,
        x_full[index],
        lb[index],
        ub[index],
        scale=0.01,
    )
    x_full[index] = y
    return na.unpack(x_full, parameters), -fun, num


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


def _outline_own(
    job: tuple,
) -> tuple[
    esis.optics.DistortionParameters, float, float, list[str]
]:  # pragma: nocover
    """
    Place one channel's windows on its measured edges.

    A module-level function so that the channels can be sent to worker
    processes.

    Parameters
    ----------
    job
        The channel, its parameters, the scene, its frame, the device, the
        merit, its measured edges, and the window and sky terms.

    Returns
    -------
    The parameters, their merit, the edge residual, and the log messages.
    """
    instrument, parameters, scene, observation, device, merit, edges, window, sky = job
    objective = esis.optics.LinearMerit(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        device=device,
        merit=merit,
    )
    messages = []
    result = esis.optics.fit_distortion_outline(
        objective,
        edges,
        window=window,
        sky=sky,
        bounds=esis.flights.f1.optics.distortion_fit_bounds(parameters),
        log=messages.append,
    )
    linear = objective.linearize(result.to_instrument(instrument))
    wavelength = instrument.wavelength
    footprints = [
        linear.footprint(wavelength[dict(wavelength=i)])
        for i in range(na.shape(wavelength)["wavelength"])
    ]
    residual = esis.optics.outline_residual(footprints, edges)
    return result, -objective(na.pack(result).ndarray), residual, messages


_PRIMARY_SCAN = tuple(d * u.mm for d in (-8, -6, -4, -2, 0))
"""
The primary-mirror displacements the shared stage tries.

Displacing the primary changes the scale of the sky on the sensor and not
the size of the windows, while the compound sensor move changes the size
of the windows and not the scale of the sky.  With the sky's scale pinned
by the merit, the width of the windows therefore measures the one shared
displacement, about half a percent of width per five millimetres.
"""


def _width_own(
    job: tuple,
) -> tuple[esis.optics.DistortionParameters, float, float]:  # pragma: nocover
    """
    Re-polish one channel's scale terms at a given primary displacement.

    A module-level function so that the channels can be sent to worker
    processes.

    Parameters
    ----------
    job
        The channel, its parameters, the scene, its frame, the device, the
        merit, its measured edges, and the names of the scale terms.

    Returns
    -------
    The polished parameters, their merit, and the error of the window
    widths.
    """
    instrument, parameters, scene, observation, device, merit, edges, terms = job
    objective = esis.optics.LinearMerit(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        device=device,
        merit=merit,
    )
    names = [f.name for f in dataclasses.fields(parameters)]
    index = np.array([names.index(n) for n in terms])
    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)
    lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
    x = na.pack(parameters).ndarray.copy()
    y, fun, _ = _fit.polish(
        _fit._Subset(objective, x.copy(), index),
        x[index],
        lb[index],
        ub[index],
        scale=0.01,
        num_round=1,
        maxfev=400,
    )
    x[index] = y
    result = na.unpack(x, parameters)
    linear = objective.linearize(result.to_instrument(instrument))
    wavelength = instrument.wavelength
    footprints = [
        linear.footprint(wavelength[dict(wavelength=i)])
        for i in range(na.shape(wavelength)["wavelength"])
    ]
    return result, -fun, esis.optics.width_residual(footprints, edges)


def scan_primary(
    instrument: esis.optics.Instrument,
    parameters: list[esis.optics.DistortionParameters],
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    edges: astropy.table.QTable,
    displacements: tuple[u.Quantity, ...] = _PRIMARY_SCAN,
    terms: tuple[str, ...] = ("z_sensor", "yaw_sensor", "pitch_sensor"),
    device: None | str = None,
    workers: int = 1,
    merit: str = "correlation",
    log: None | Callable[[str], None] = None,
) -> tuple[
    list[esis.optics.DistortionParameters], u.Quantity, list[float]
]:  # pragma: nocover
    """
    Find the one primary displacement whose windows have the measured width.

    At every trial displacement the scale terms of each channel are
    re-polished against the merit, which pins the scale of the sky, and the
    widths of the windows are compared with the measured edges.  The
    displacement with the smallest summed width error wins, refined by a
    parabola through its neighbours, and every channel is re-polished
    there.

    Parameters
    ----------
    instrument
        The instrument model.
    parameters
        The parameters of every channel.
    scene
        The scene of the reference frame.
    observation
        The reference frame of every channel.
    edges
        The measured edges of every channel's windows.
    displacements
        The displacements to try.
    terms
        The names of the terms re-polished at every displacement.
    device
        The device the merit runs on.
    workers
        The number of channels polished side by side.
    merit
        The comparison the polish maximizes.
    log
        A callable that records progress.

    Returns
    -------
    The parameters of every channel at the winning displacement, the
    displacement, and the width errors of every trial.
    """
    num_channel = len(parameters)

    def trial(displacement: u.Quantity) -> list[tuple]:
        jobs = []
        for c in range(num_channel):
            p = copy.copy(parameters[c])
            p.displacement_primary = displacement
            jobs.append(
                (
                    instrument[dict(channel=c)],
                    p,
                    scene,
                    observation[dict(channel=c)],
                    device,
                    merit,
                    edges[np.asarray(edges["channel"]) == c],
                    terms,
                )
            )
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(
                min(workers, num_channel)
            ) as pool:
                return pool.map(_width_own, jobs)
        return [_width_own(job) for job in jobs]

    errors = []
    for displacement in displacements:
        results = trial(displacement)
        error = float(np.sqrt(np.mean([r[2] ** 2 for r in results])))
        errors.append(error)
        if log is not None:
            log(
                f"primary {displacement:+.2f}: width error {error:.2f} px "
                "("
                + ", ".join(f"{r[2]:.2f}" for r in results)
                + "), merit "
                + ", ".join(f"{r[1]:.4f}" for r in results)
            )
    values = u.Quantity(displacements)
    k = int(np.argmin(errors))
    best = values[k]
    if 0 < k < len(errors) - 1:
        # a parabola through the minimum and its neighbours
        y0, y1, y2 = errors[k - 1], errors[k], errors[k + 1]
        denominator = y0 - 2 * y1 + y2
        if denominator > 0:
            step = values[k + 1] - values[k]
            best = values[k] + 0.5 * (y0 - y2) / denominator * step
    if log is not None:
        log(f"primary: {best:+.2f} from the width of the windows")
    results = trial(best)
    return [r[0] for r in results], best, errors


def _window_centers(
    instrument: esis.optics.Instrument,
    parameters: list[esis.optics.DistortionParameters],
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
) -> list[list[list[float]]]:
    """
    Locate the centre of every window on the sensor, in pixels.

    Parameters
    ----------
    instrument
        The instrument model.
    parameters
        The parameters of every channel.
    scene
        The scene of the reference frame.
    observation
        The reference frame of every channel.

    Returns
    -------
    A list indexed ``[channel][line]`` of ``[x, y]``.
    """
    result = []
    for c, p in enumerate(parameters):
        channel = instrument[dict(channel=c)]
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=p,
            scene=scene,
            observation=observation[dict(channel=c)],
        )
        linear = merit.linearize(p.to_instrument(channel))
        wavelength = channel.wavelength
        centers = []
        for i in range(na.shape(wavelength)["wavelength"]):
            footprint = linear.footprint(wavelength[dict(wavelength=i)])
            centers.append(
                [
                    float(np.mean(na.value(footprint.x).ndarray)),
                    float(np.mean(na.value(footprint.y).ndarray)),
                ]
            )
        result.append(centers)
    return result


def _versions() -> dict[str, str]:
    """
    Record the versions of the packages the fit ran on.

    An editable install reports the version its metadata was built with,
    which can lag the checkout, so the commit of a package that lives in a
    git checkout is recorded beside it.
    """
    import subprocess

    result = {}
    modules = dict(
        esis="euv-snapshot-imaging-spectrograph",
        optika="optika",
        named_arrays="named-arrays",
        regridding="regridding",
    )
    for module, name in modules.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        try:
            path = pathlib.Path(importlib.import_module(module).__file__).parent
            commit = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if commit.returncode == 0:
                version = f"{version} @ {commit.stdout.strip()}"
                status = subprocess.run(
                    ["git", "-C", str(path), "status", "--porcelain", "--", "."],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if status.returncode == 0 and status.stdout.strip():
                    # uncommitted changes under the package: the commit does
                    # not name the code that ran
                    version = version + " +dirty"
        except Exception:
            pass
        result[name] = version
    return result


def measure_window_edges(
    parameters: list[esis.optics.DistortionParameters],
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    frames: None | tuple[int, ...] = None,
    threshold: float = 0.5,
    lines: tuple[int, ...] = (0, 2),
    error_max: u.Quantity = 1 * u.pix,
    directory: None | str | pathlib.Path = None,
    path: None | str | pathlib.Path = None,
    path_drift: None | str | pathlib.Path = None,
) -> tuple[astropy.table.QTable, astropy.table.QTable]:  # pragma: nocover
    """
    Measure the edges of every window in every frame of the flight.

    The windows are the image of the field stop, and do not move with the
    pointing, so the same edge is measured once per frame against a
    different backdrop of solar structure each time and the median over the
    frames is what the outline stage fits.  The per-frame departures from
    that median are kept too, as the drift of the windows through the
    flight.

    Parameters
    ----------
    parameters
        The parameters of every channel, which place the outlines the
        edges are searched near.
    instrument
        The instrument model, see :func:`fit_distortion_reference`.
    num_scene
        The number of samples along each axis of the resampled scene, which
        sets the grid the channel is linearized on.
    frames
        The frames to measure.  If :obj:`None`, every frame whose mean
        signal exceeds `threshold` times the median over the flight.
    threshold
        The fraction of the flight's median signal below which a frame is
        left out.
    lines
        The indices of the spectral lines whose windows are measured; the
        middle window of ESIS-I sits on its neighbours and is left out.
    error_max
        Crossings with a larger formal error are left out of the median.
    directory
        The directory the log is written to.
    path
        Where to save the median edges, if anywhere.
    path_drift
        Where to save the per-frame drift, if anywhere.

    Returns
    -------
    The median edges, one row per channel, line, side and row or column,
    and the drift, one row per channel, frame and side.
    """
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    log = _logger(directory, "edges")
    level_1 = esis.flights.f1.data.level_1()
    axes = ("time", "channel", "detector_y", "detector_x")
    data = np.asarray(na.value(level_1.outputs).ndarray_aligned(axes), dtype=float)
    signal = data.mean(axis=(2, 3))
    if frames is None:
        kept = np.all(signal > threshold * np.median(signal, axis=0), axis=1)
        frames = tuple(int(t) for t in np.where(kept)[0])
    log(f"frames kept: {', '.join(str(t) for t in frames)}")
    scene, observation = _frame(15, num_scene)
    tables = []
    for c in range(num_channel):
        channel = instrument[dict(channel=c)]
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=parameters[c],
            scene=scene,
            observation=observation[dict(channel=c)],
        )
        linear = merit.linearize(parameters[c].to_instrument(channel))
        wavelength = channel.wavelength
        footprints = [
            linear.footprint(wavelength[dict(wavelength=i)])
            for i in range(na.shape(wavelength)["wavelength"])
        ]
        for t in frames:
            edges = esis.optics.measure_edges(data[t, c], footprints)
            edges = edges[np.isin(edges["line"], lines)]
            edges["frame"] = t
            edges["channel"] = c
            tables.append(edges)
            log(f"frame {t} channel {c}: {len(edges)} crossings")
    every = astropy.table.vstack(tables)
    every = every[every["error"] < error_max]
    keys = np.stack(
        [np.asarray(every[k]) for k in ("channel", "line", "side", "index")],
        axis=1,
    )
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    position = np.asarray(every["position"].to_value(u.pix))
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(unique) + 1))
    rows = []
    center = np.empty(len(every))
    for g in range(len(unique)):
        members = order[bounds[g] : bounds[g + 1]]
        values = position[members]
        median = float(np.median(values))
        spread = 1.4826 * float(np.median(np.abs(values - median)))
        rows.append((*unique[g], median, spread / np.sqrt(len(values)), len(values)))
        center[members] = median
    median_edges = astropy.table.QTable(
        rows=rows,
        names=("channel", "line", "side", "index", "position", "error", "num"),
        dtype=(int, int, int, int, float, float, int),
    )
    median_edges["position"].unit = u.pix
    median_edges["error"].unit = u.pix
    departure = position - center
    drift_rows = []
    for c in range(num_channel):
        for t in frames:
            for side in range(4):
                sel = (
                    (np.asarray(every["channel"]) == c)
                    & (np.asarray(every["frame"]) == t)
                    & (np.asarray(every["side"]) == side)
                )
                if sel.any():
                    drift_rows.append(
                        (c, t, side, float(np.median(departure[sel])), int(sel.sum()))
                    )
    drift = astropy.table.QTable(
        rows=drift_rows,
        names=("channel", "frame", "side", "shift", "num"),
        dtype=(int, int, int, float, int),
    )
    drift["shift"].unit = u.pix
    meta = dict(
        description=(
            "Where the edges of each window cross the rows and columns of the "
            "ESIS-I frames."
        ),
        frames=list(frames),
        lines=list(lines),
        error_max=float(error_max.to_value(u.pix)),
        sides=list(esis.optics.SIDES),
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        versions=_versions(),
    )
    median_edges.meta.update(meta)
    drift.meta.update(meta)
    drift.meta["description"] = (
        "The departure of each window's edges from their flight median, per frame."
    )
    if path is not None:
        median_edges.write(path, format="ascii.ecsv", overwrite=True)
    if path_drift is not None:
        drift.write(path_drift, format="ascii.ecsv", overwrite=True)
    log(f"{len(median_edges)} median edges, {len(drift)} drift rows")
    return median_edges, drift


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
    free_absolute: tuple[str, ...] = _FREE_ABSOLUTE,
    free_shared: tuple[str, ...] = _FREE_SHARED,
    edges: None | str | pathlib.Path | astropy.table.QTable = None,
    window: tuple[str, ...] = _WINDOW,
    sky: tuple[str, ...] = _SKY,
    primary: bool = True,
    merit: str = "correlation",
    popsize: int = 15,
    maxiter: int = 80,
    tol: float = 0.0,
    seed: int = 0,
) -> esis.optics.DistortionParameters:  # pragma: nocover
    """
    Fit the reference distortion parameters of ESIS-I from the as-built model.

    Reproduces the values stored in ``_data/distortion_reference.ecsv`` from
    the instrument model and the flight data alone, in these stages.

    1.  **Absolute.**  Each channel is fit on its own to the AIA proxy scene
        with :class:`esis.optics.LinearMerit` and
        :func:`esis.optics.fit_distortion`, over the terms one frame can
        determine: a seeded capture followed by a local polish.  About two
        hours per channel on one GPU.
    2.  **Outline.**  If the window edges measured through the flight are
        given, each channel's windows are placed on them with the grating
        angles and the sensor roll, and the sky is refit with the remaining
        terms, see :func:`esis.optics.fit_distortion_outline`.
    3.  **Primary.**  Optionally, one shared displacement of the primary
        from the width of the windows against the scale of the sky, see
        :func:`scan_primary`.
    4.  **Shared.**  The primary, the field-stop roll and the payload
        pointing belong to the instrument; they are set to their mean over
        the channels and each channel's own terms are polished again.
    5.  **Internal.**  The channels are aligned to one another on the sky
        with :func:`esis.optics.align_channels`, in the grating and camera
        terms only, to about a tenth of a pixel.  Minutes.

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
        The number of processes evaluating the population of the capture,
        and polishing the channels of the shared stage side by side.
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
    free_absolute
        The names of the fields the absolute stage searches, per channel.
        The default is the nine terms one frame determines.
    free_shared
        The names of the fields polished per channel with the shared optics
        fixed, and refit by the alignment.  The default is the grating and
        camera placement.  With the least-squares merit the degradation is
        polished too.
    edges
        The measured edges of the windows, see :func:`measure_window_edges`,
        or the path of their table.  If given, an outline stage places each
        channel's windows on them between the absolute and shared stages.
    window
        The names of the fields the outline stage fits to the edges.
    sky
        The names of the fields the outline stage re-polishes against the
        merit.
    primary
        Whether to find the shared primary displacement from the width of
        the windows, see :func:`scan_primary`; needs `edges`.
    merit
        The comparison every stage maximizes, see
        :class:`esis.optics.LinearMerit`: ``"correlation"`` or
        ``"least_squares"``.
    popsize
        The population of the capture of the absolute stage, per dimension.
    maxiter
        The number of generations of the capture.
    tol
        The tolerance at which the capture stops early; zero runs every
        generation.
    seed
        The seed of the capture.
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
        objective = esis.optics.LinearMerit(
            instrument=channel,
            parameters=p0,
            scene=scene,
            observation=observation[dict(channel=c)],
            device=device,
            merit=merit,
        )
        log(f"channel {c}: absolute fit")
        parameters[c] = esis.optics.fit_distortion(
            objective=objective,
            parameters=p0,
            bounds=esis.flights.f1.optics.distortion_fit_bounds(p0),
            workers=workers,
            free=free_absolute,
            popsize=popsize,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            log=log,
        )
    if any(p is None for p in parameters):
        raise ValueError("every channel needs parameters before the shared stage")
    # the merit of every channel as the absolute stage left it, whether it
    # ran here or its result was handed in
    scores = dict(
        absolute=[
            float(
                esis.optics.LinearMerit(
                    instrument=instrument[dict(channel=c)],
                    parameters=parameters[c],
                    scene=scene,
                    observation=observation[dict(channel=c)],
                    device=device,
                    merit=merit,
                ).correlation(parameters[c])
            )
            for c in range(num_channel)
        ]
    )

    # 1b. outline
    residuals = None
    if edges is not None:
        if not isinstance(edges, astropy.table.QTable):
            edges = astropy.table.QTable.read(edges, format="ascii.ecsv")
        log(
            "outline: "
            + ", ".join(window)
            + " on the edges; "
            + ", ".join(sky)
            + " on the merit"
        )
        jobs = [
            (
                instrument[dict(channel=c)],
                parameters[c],
                scene,
                observation[dict(channel=c)],
                device,
                merit,
                edges[np.asarray(edges["channel"]) == c],
                window,
                sky,
            )
            for c in range(num_channel)
        ]
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(
                min(workers, num_channel)
            ) as pool:
                results = pool.map(_outline_own, jobs)
        else:
            results = [_outline_own(job) for job in jobs]
        residuals = []
        for c, (parameters[c], rho, residual, messages) in enumerate(results):
            for message in messages:
                log(f"channel {c}: {message}")
            log(f"channel {c}: {rho:.4f}, edge residual {residual:.2f} px")
            residuals.append(residual)
        scores["outline"] = dict(merit=[float(r[1]) for r in results], edges=residuals)

    # 1c. the shared primary, from the width of the windows
    displacement = None
    if edges is not None and primary:
        parameters, displacement, widths = scan_primary(
            instrument=instrument,
            parameters=parameters,
            scene=scene,
            observation=observation,
            edges=edges,
            device=device,
            workers=workers,
            merit=merit,
            log=log,
        )
        scores["primary"] = dict(
            displacements=[float(d.to_value(u.mm)) for d in _PRIMARY_SCAN],
            widths=widths,
            displacement=float(displacement.to_value(u.mm)),
        )

    # 2. shared
    parameters = enforce_shared(parameters)
    names = [f.name for f in dataclasses.fields(parameters[0])]
    own = [n for n in free_shared if n not in _SHARED]
    if merit == "least_squares":
        own = own + [n for n in _PHOTOMETRIC if n not in own]
    index_own = [names.index(n) for n in own]
    log("polish of every channel with the shared optics fixed: " + ", ".join(own))
    jobs = [
        (
            instrument[dict(channel=c)],
            parameters[c],
            scene,
            observation[dict(channel=c)],
            device,
            index_own,
            merit,
        )
        for c in range(num_channel)
    ]
    if workers > 1:
        # the polish is serial within a channel but the channels are
        # independent, so they run side by side
        with multiprocessing.get_context("spawn").Pool(
            min(workers, num_channel)
        ) as pool:
            results = pool.map(_polish_own, jobs)
    else:
        results = [_polish_own(job) for job in jobs]
    for c, (parameters[c], rho, num) in enumerate(results):
        log(f"channel {c}: {rho:.4f} after {num} evaluations")
    scores["shared"] = [float(r[1]) for r in results]

    # 3. internal
    log("internal alignment")
    parameters, medians = esis.optics.align_channels(
        instruments=[instrument[dict(channel=c)] for c in range(num_channel)],
        parameters=parameters,
        frames=_frames_by_axis(observation, num_channel),
        scene_position=scene.inputs.position,
        wavelengths=_wavelengths_alignment(),
        free=tuple(n for n in own if n not in _PHOTOMETRIC),
        log=log,
    )
    scores["alignment"] = [float(m) for m in medians]

    # where the windows land, for a regression test of the committed table
    windows = _window_centers(instrument, parameters, scene, observation)

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
                    "polish of every channel against the AIA proxy scene"
                    + (
                        ", its windows placed on the measured edges"
                        if edges is not None
                        else ""
                    )
                    + ", the shared optics set to their mean, and the channels "
                    "aligned to one another on the sky at "
                    f"{', '.join(_LINES_ALIGNMENT)} "
                    "(median shift vs channel 1: "
                    f"{', '.join(f'{m:.2f}' for m in medians)} px)"
                ),
                stages=dict(
                    absolute=dict(
                        free=list(free_absolute),
                        popsize=popsize,
                        maxiter=maxiter,
                        tol=tol,
                        seed=seed,
                    ),
                    outline=(
                        dict(window=list(window), sky=list(sky), edges=residuals)
                        if edges is not None
                        else None
                    ),
                    primary=(
                        dict(
                            displacement=float(displacement.to_value(u.mm)),
                            scan=[float(d.to_value(u.mm)) for d in _PRIMARY_SCAN],
                        )
                        if displacement is not None
                        else None
                    ),
                    shared=dict(free=list(own), fixed=list(_SHARED)),
                    alignment=dict(
                        free=[n for n in own if n not in _PHOTOMETRIC],
                        lines=list(_LINES_ALIGNMENT),
                        anchor=1,
                    ),
                ),
                merit=merit,
                scores=scores,
                windows=windows,
                versions=_versions(),
                date=datetime.datetime.now().isoformat(timespec="seconds"),
            ),
        )
    return result


def realign_distortion_reference(
    reference: str | pathlib.Path,
    time: int = 15,
    num_scene: int = 401,
    instrument: None | esis.optics.Instrument = None,
    free_shared: tuple[str, ...] = _FREE_SHARED,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> esis.optics.DistortionParameters:  # pragma: nocover
    """
    Rerun only the internal alignment of a saved reference.

    The alignment reads the channels' frames, not the proxy scene, so it is
    minutes where the polish is hours; this repeats it on a saved reference,
    for instance after a change to how the frames are read, and carries the
    reference's record of its earlier stages into the result.

    Parameters
    ----------
    reference
        The path of the saved reference table.
    time
        The frame the reference was fit to.
    num_scene
        The number of samples along each axis of the resampled AIA scene,
        whose extent sets the sky grid.
    instrument
        The base model.  If :obj:`None`, the idealized as-built model.
    free_shared
        The free set of the shared stage; the alignment frees the same
        terms less the photometric ones.
    path
        If given, the realigned reference is written here.
    directory
        A directory where the progress is logged.
    """
    reference = pathlib.Path(reference)
    start = astropy.table.QTable.read(reference, format="ascii.ecsv")
    parameters_start = esis.optics.DistortionParameters.from_file(reference)
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    parameters = [_channel(parameters_start, c) for c in range(num_channel)]
    log = _logger(directory, "realign")
    scene, observation = _frame(time, num_scene)
    own = [n for n in free_shared if n not in _SHARED and n not in _PHOTOMETRIC]
    log("internal alignment of " + reference.name + ", free: " + ", ".join(own))
    parameters, medians = esis.optics.align_channels(
        instruments=[instrument[dict(channel=c)] for c in range(num_channel)],
        parameters=parameters,
        frames=_frames_by_axis(observation, num_channel),
        scene_position=scene.inputs.position,
        wavelengths=_wavelengths_alignment(),
        free=tuple(own),
        log=log,
    )
    windows = _window_centers(instrument, parameters, scene, observation)
    result = _stack(parameters, axis="channel")
    if path is not None:
        # the writer names the logical axis itself
        meta = {k: v for k, v in start.meta.items() if k != "axis"}
        meta["scores"] = dict(
            meta.get("scores", {}), alignment=[float(m) for m in medians]
        )
        meta["stages"] = dict(
            meta.get("stages", {}),
            alignment=dict(free=list(own), lines=list(_LINES_ALIGNMENT), anchor=1),
        )
        meta["windows"] = windows
        meta["realigned"] = dict(
            date=datetime.datetime.now().isoformat(timespec="seconds"),
            start=start.meta.get("date"),
            note=(
                "the internal alignment rerun on the saved reference with the "
                "frames read inside their windows (median shift vs channel 1: "
                + ", ".join(f"{m:.2f}" for m in medians)
                + " px)"
            ),
            versions=_versions(),
        )
        result.to_file(path, metadata=meta)
    return result


def _drift_shift(
    drift: astropy.table.QTable,
    channel: int,
    frame: int,
) -> np.ndarray:
    """
    Read the motion of one channel's windows in one frame, in pixels.

    The x motion is the mean over the left and right sides, the y motion
    over the top and bottom; a side not measured in that frame counts as
    zero.

    Parameters
    ----------
    drift
        The drift table, see :func:`measure_window_edges`.
    channel
        The channel.
    frame
        The frame.
    """
    rows = drift[
        (np.asarray(drift["channel"]) == channel)
        & (np.asarray(drift["frame"]) == frame)
    ]
    side = np.asarray(rows["side"])
    shift = rows["shift"].to_value(u.pix)
    result = np.zeros(2)
    for k, sides in enumerate(((0, 1), (2, 3))):
        values = shift[np.isin(side, sides)]
        if values.size:
            result[k] = values.mean()
    return result


def _drift_smooth(
    drift: astropy.table.QTable,
    num_channel: int,
    degree: int,
) -> Callable[[int, int], np.ndarray]:
    """
    Fit the motion of every channel's windows through the flight with a polynomial.

    The edges of a single frame place its windows to a few tenths of a
    pixel, and frames too dark to measure have no edges at all, so a
    per-frame shift applied as measured jitters from frame to frame and
    drops to zero where nothing was measured.  The motion is smooth, so a
    low-order polynomial in the frame index carries it instead; outside the
    measured frames the value at the nearest measured frame is held.

    Parameters
    ----------
    drift
        The drift table, see :func:`measure_window_edges`.
    num_channel
        The number of channels.
    degree
        The degree of the polynomial in the frame index.

    Returns
    -------
    A function of the channel and the frame returning the x and y motion
    of that channel's windows in pixels.

    Raises
    ------
    ValueError
        If a channel was measured in too few frames for the polynomial.
    """
    channel_of = np.asarray(drift["channel"])
    frame_of = np.asarray(drift["frame"])
    coefficients = np.zeros((num_channel, 2, degree + 1))
    span = np.zeros((num_channel, 2), dtype=int)
    for c in range(num_channel):
        # only the frames this channel was measured in: an unmeasured frame
        # is not a zero
        frames = sorted(set(int(t) for t in frame_of[channel_of == c]))
        if len(frames) <= degree:
            raise ValueError(
                f"channel {c} has {len(frames)} measured frames, too few for a "
                f"degree-{degree} polynomial"
            )
        measured = np.array([_drift_shift(drift, c, t) for t in frames])
        for k in range(2):
            coefficients[c, k] = np.polyfit(frames, measured[:, k], degree)
        span[c] = frames[0], frames[-1]

    def shift(channel: int, frame: int) -> np.ndarray:
        t = min(max(frame, span[channel, 0]), span[channel, 1])
        return np.array([np.polyval(coefficients[channel, k], t) for k in range(2)])

    return shift


def _drift_sensitivity(
    instrument: esis.optics.Instrument,
    reference: list[esis.optics.DistortionParameters],
    merits: list,
    step: u.Quantity = 1 * u.arcmin,
) -> np.ndarray:
    """
    Measure how far the windows move per arcminute of grating yaw and pitch.

    Parameters
    ----------
    instrument
        The instrument model.
    reference
        The parameters of every channel.
    merits
        The merit of every channel, which linearizes it.
    step
        The step of the finite difference.

    Returns
    -------
    An array indexed ``[channel, axis]``: pixels of x per arcminute of
    grating yaw, and of y per arcminute of grating pitch.
    """
    result = np.zeros((len(reference), 2))
    for c, (p, merit) in enumerate(zip(reference, merits)):
        channel = instrument[dict(channel=c)]
        wavelength = channel.wavelength[dict(wavelength=~0)]

        def center(q):
            linear = merit.linearize(q.to_instrument(channel))
            footprint = linear.footprint(wavelength)
            return np.array(
                [
                    float(np.mean(na.value(footprint.x).ndarray)),
                    float(np.mean(na.value(footprint.y).ndarray)),
                ]
            )

        c0 = center(p)
        yawed = copy.copy(p)
        yawed.yaw_grating = p.yaw_grating + step
        pitched = copy.copy(p)
        pitched.pitch_grating = p.pitch_grating + step
        result[c, 0] = (center(yawed) - c0)[0] / step.to_value(u.arcmin)
        result[c, 1] = (center(pitched) - c0)[1] / step.to_value(u.arcmin)
    return result


def _defocus_pattern(
    instrument: esis.optics.Instrument,
    reference: list[esis.optics.DistortionParameters],
    wavelengths: dict[str, u.Quantity],
    anchor: int = 1,
    step: u.Quantity = 0.1 * u.mm,
) -> np.ndarray:
    """
    Measure how a defocus of the primary moves each channel's sky against the anchor's.

    Each channel views the defocused solar image through its own sector of
    the primary, so the image shifts on the sky by the sector's offset times
    the defocus while the window edges stay put; against the anchor the
    shifts differ from channel to channel.  This is that pattern, from the
    model, per unit of :attr:`~esis.optics.DistortionParameters.z_primary`.

    Parameters
    ----------
    instrument
        The instrument model.
    reference
        The parameters of every channel.
    wavelengths
        The lines to measure at; the pattern is their mean.
    anchor
        The channel the others are measured against.
    step
        The defocus of the finite difference.

    Returns
    -------
    An array indexed ``[channel, axis]``: the shift on the sky in detector
    pixels of each channel relative to the anchor per millimetre of
    defocus; zero for the anchor.
    """
    axis = ("s", "s")
    origin = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 10.0, 0.0]) * u.arcsec, axes="s"),
        y=na.ScalarArray(np.array([0.0, 0.0, 10.0]) * u.arcsec, axes="s"),
    )

    def where(c, dz):
        p = copy.copy(reference[c])
        p.z_primary = p.z_primary + dz
        model = p.to_instrument(instrument[dict(channel=c)])
        linear = model.system.linearize(
            wavelength=model.wavelength, degree=2, field_stop=True
        )
        result = []
        for wavelength in wavelengths.values():
            x, y = _alignment.sensor_coordinates(
                linear.distortion, origin, wavelength, axis
            )
            x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
            jacobian = (
                np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]])
                / 10.0
            )
            result.append((np.array([x[0], y[0]]), jacobian))
        return result

    num_channel = len(reference)
    scale = None
    shifts = np.zeros((num_channel, 2))
    for c in range(num_channel):
        before, after = where(c, 0 * step), where(c, step)
        per_line = []
        for (s0, jacobian), (s1, _) in zip(before, after):
            # the sensor shift, taken back to the sky in arcseconds
            per_line.append(np.linalg.solve(jacobian, s1 - s0))
        shifts[c] = np.mean(per_line, axis=0)
        if c == anchor:
            # the anchor's pixels per arcsecond, the unit the shifts report in
            scale = float(np.mean([np.sqrt(abs(np.linalg.det(j))) for _, j in before]))
    shifts = (shifts - shifts[anchor]) * scale
    return shifts / step.to_value(u.mm)


def fit_defocus_history(
    reference: str | pathlib.Path | esis.optics.DistortionParameters,
    num_scene: int = 401,
    instrument: None | esis.optics.Instrument = None,
    frames: None | tuple[int, ...] = None,
    frame_reference: int = 15,
    drift: None | str | pathlib.Path | astropy.table.QTable = None,
    drift_degree: None | int = 3,
    degree: int = 1,
    anchor: int = 1,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> astropy.table.QTable:  # pragma: nocover
    """
    Measure the defocus of the primary through the flight from the channels.

    In every frame the channels' skies are measured against the anchor's
    (:func:`esis.optics.measure_channel_shifts`), with each channel's
    windows placed by that frame's drift as the pointing stage places them.
    The one defocus that best reproduces the shifts through the model's
    pattern (:func:`_defocus_pattern`) is solved per frame, and a polynomial
    in the frame index is fit through the flight.  The proxy scene takes no
    part: the channels are compared with one another, which resolves a
    tenth of a pixel where the scene resolves one, so the defocus comes out
    to a few microns.

    Parameters
    ----------
    reference
        The reference parameters, or the path of their table.
    num_scene
        The number of samples along each axis of the resampled AIA scene,
        whose extent sets the sky grid.
    instrument
        The base model.  If :obj:`None`, the idealized as-built model.
    frames
        The frames to measure.  If :obj:`None`, every frame.
    frame_reference
        The frame the reference was fit to, where the defocus is zero by
        definition; the fit is made relative to it.
    drift
        The per-frame drift of the windows, see :func:`measure_window_edges`.
    drift_degree
        The smoothing of the drift, see :func:`fit_distortion_pointing`.
    degree
        The degree of the polynomial in the frame index fit through the
        per-frame defocus.
    anchor
        The channel the others are measured against.
    path
        If given, the history is written here as an ECSV table with the
        per-frame and the fitted defocus.
    directory
        A directory where the progress is logged.
    """
    if not isinstance(reference, esis.optics.DistortionParameters):
        reference = esis.optics.DistortionParameters.from_file(reference)
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    parameters = [_channel(reference, c) for c in range(num_channel)]
    num_time = esis.flights.f1.data.level_1().shape["time"]
    if frames is None:
        frames = tuple(range(num_time))
    log = _logger(directory, "defocus")
    wavelengths = _wavelengths_alignment()

    pattern = _defocus_pattern(instrument, parameters, wavelengths, anchor)
    log(
        "sky shift against channel %d per mm of defocus [px]: " % anchor
        + "; ".join(f"{p[0]:+.1f}, {p[1]:+.1f}" for p in pattern)
    )
    others = [c for c in range(num_channel) if c != anchor]
    design = np.concatenate([pattern[c] for c in others])

    if drift is not None and not isinstance(drift, astropy.table.QTable):
        drift = astropy.table.QTable.read(drift, format="ascii.ecsv")
    shift_drift = None
    if drift is not None:
        if drift_degree is not None:
            shift_drift = _drift_smooth(drift, num_channel, drift_degree)
        else:
            shift_drift = functools.partial(_drift_shift, drift)
    sensitivity = None

    rows = []
    for t in frames:
        scene, observation = _frame(t, num_scene)
        sky = _alignment.sky_grid(scene.inputs.position, 1600)
        linears = []
        for c in range(num_channel):
            p = copy.copy(parameters[c])
            if shift_drift is not None:
                if sensitivity is None:
                    merits = [
                        esis.optics.LinearMerit(
                            instrument=instrument[dict(channel=k)],
                            parameters=parameters[k],
                            scene=scene,
                            observation=observation[dict(channel=k)],
                        )
                        for k in range(num_channel)
                    ]
                    sensitivity = _drift_sensitivity(instrument, parameters, merits)
                offset = shift_drift(c, t) / sensitivity[c]
                p.yaw_grating = p.yaw_grating + offset[0] * u.arcmin
                p.pitch_grating = p.pitch_grating + offset[1] * u.arcmin
            model = p.to_instrument(instrument[dict(channel=c)])
            linears.append(
                model.system.linearize(
                    wavelength=model.wavelength, degree=2, field_stop=True
                )
            )
        fields, scales = esis.optics.measure_channel_shifts(
            linears,
            _frames_by_axis(observation, num_channel),
            sky,
            wavelengths,
            anchor=anchor,
        )
        medians = esis.optics.median_shifts(fields, scales)
        measured = np.concatenate([medians[c] for c in others])
        ok = np.isfinite(measured)
        # a measured shift s means the anchor shows at r + s what the channel
        # shows at r (see shift_fft): the channel's image sits at -s, and the
        # defocus must move the model's image there
        if ok.any():
            z = -float(
                np.dot(design[ok], measured[ok]) / np.dot(design[ok], design[ok])
            )
            residual = float(np.sqrt(np.mean((measured[ok] + z * design[ok]) ** 2)))
        else:
            z, residual = np.nan, np.nan
        log(
            f"frame {t}: shifts vs channel {anchor} "
            + "; ".join(f"{medians[c][0]:+.2f}, {medians[c][1]:+.2f}" for c in others)
            + f" px -> defocus {1e3 * z:+.1f} um, residual {residual:.3f} px"
        )
        rows.append((t, z, residual))

    t_all = np.array([r[0] for r in rows], dtype=float)
    z_all = np.array([r[1] for r in rows])
    # the reference frame is where the defocus is zero by definition, so the
    # polynomial has no constant term; a frame that could not be measured
    # takes no part
    ok = np.isfinite(z_all)
    powers = np.stack(
        [(t_all - frame_reference) ** k for k in range(degree, 0, -1)], axis=-1
    )
    higher, *_ = np.linalg.lstsq(powers[ok], z_all[ok], rcond=None)
    coefficients = np.append(higher, 0.0)
    fitted = np.polyval(coefficients, t_all - frame_reference)
    scatter = float(
        np.std(
            z_all[ok]
            - np.polyval(
                np.polyfit(t_all[ok] - frame_reference, z_all[ok], degree),
                t_all - frame_reference,
            )
        )
    )
    log(
        f"defocus through the flight: degree-{degree} fit, "
        + ", ".join(f"{1e3 * c:+.2f}" for c in coefficients[:-1])
        + f" um per frame^k about frame {frame_reference}; "
        + f"scatter {1e3 * scatter:.1f} um"
    )
    table = astropy.table.QTable(
        dict(
            frame=np.array([r[0] for r in rows]),
            z_primary_measured=z_all * u.mm,
            residual=np.array([r[2] for r in rows]) * u.pix,
            z_primary=fitted * u.mm,
        )
    )
    table.meta.update(
        dict(
            description=(
                "The defocus of the primary at the field stop through the "
                "ESIS-I flight, relative to the reference frame, measured from "
                "the shifts of the channels' skies against one another and "
                "smoothed by a polynomial in the frame index; z_primary is the "
                "smoothed value the pointing stage applies."
            ),
            provenance=(
                f"esis.flights.f1.optics.fit_defocus_history(num_scene={num_scene}, "
                f"degree={degree}, anchor={anchor}, drift_degree={drift_degree})"
            ),
            frame_reference=int(frame_reference),
            pattern=[[float(v) for v in p] for p in pattern],
            coefficients=[float(c) for c in coefficients],
            scatter=scatter,
        )
    )
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def frame_parameters(
    reference: esis.optics.DistortionParameters,
    pointing: astropy.table.QTable,
    frame: int,
    channel: int,
    channel_offsets: bool = True,
) -> esis.optics.DistortionParameters:
    """
    Take one channel's reference parameters to one frame of the flight.

    Everything the pointing table carries for the frame is applied: the
    pointing of the payload, the defocus of the primary, the channel's
    grating offsets that follow its windows, and the channel's own pointing
    offset that registers it with the others (:func:`fit_coregistration`).
    The flight factory applies the same columns to the whole model.

    Parameters
    ----------
    reference
        The reference parameters, with a logical axis over the channels.
    pointing
        The pointing table, see :func:`fit_distortion_pointing`.
    frame
        The frame of :func:`esis.flights.f1.data.level_1`.
    channel
        The channel.
    channel_offsets
        Whether to apply the channel's own pointing offset, if the table
        has one.

    Raises
    ------
    ValueError
        If `frame` is not in the table exactly once.
    """
    rows = pointing[np.asarray(pointing["frame"]) == frame]
    if len(rows) != 1:
        raise ValueError(f"frame {frame} is not in the pointing table")
    row = rows[0]
    p = _channel(reference, channel)
    p.pitch = p.pitch + row["pitch"]
    p.yaw = p.yaw + row["yaw"]
    p.roll = p.roll + row["roll"]
    if "z_primary" in pointing.colnames:
        p.z_primary = p.z_primary + row["z_primary"]
    for name in ("yaw_grating", "pitch_grating"):
        if name in pointing.colnames:
            setattr(p, name, getattr(p, name) + row[name][channel])
    if channel_offsets:
        for name in ("pitch", "yaw"):
            if f"{name}_channel" in pointing.colnames:
                setattr(p, name, getattr(p, name) + row[f"{name}_channel"][channel])
    return p


def _pointing_pattern(
    instrument: esis.optics.Instrument,
    parameters: list[esis.optics.DistortionParameters],
    wavelengths: dict[str, u.Quantity],
    anchor: int = 1,
    step: u.Quantity = 1 * u.arcsec,
) -> np.ndarray:
    """
    Measure how a pointing offset of one channel moves that channel's sky.

    The same finite difference as :func:`_defocus_pattern`, in the channel's
    own pitch and yaw: where a point of the sky lands on the sensor before
    and after, taken back to the sky and expressed in the anchor's pixels.

    Parameters
    ----------
    instrument
        The instrument model.
    parameters
        The parameters of every channel.
    wavelengths
        The lines to measure at; the pattern is their mean.
    anchor
        The channel whose pixels the shifts are expressed in.
    step
        The offset of the finite difference.

    Returns
    -------
    An array indexed ``[channel, axis, term]``: the shift on the sky in
    detector pixels, along each axis of the sky, of each channel per
    arcsecond of its own pitch and of its own yaw.
    """
    axis = ("s", "s")
    origin = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 10.0, 0.0]) * u.arcsec, axes="s"),
        y=na.ScalarArray(np.array([0.0, 0.0, 10.0]) * u.arcsec, axes="s"),
    )

    def where(c, name, d):
        p = copy.copy(parameters[c])
        setattr(p, name, getattr(p, name) + d)
        model = p.to_instrument(instrument[dict(channel=c)])
        linear = model.system.linearize(
            wavelength=model.wavelength, degree=2, field_stop=True
        )
        result = []
        for wavelength in wavelengths.values():
            x, y = _alignment.sensor_coordinates(
                linear.distortion, origin, wavelength, axis
            )
            x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
            jacobian = (
                np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]])
                / 10.0
            )
            result.append((np.array([x[0], y[0]]), jacobian))
        return result

    num_channel = len(parameters)
    result = np.zeros((num_channel, 2, 2))
    scale = None
    for c in range(num_channel):
        before = where(c, "pitch", 0 * step)
        if c == anchor:
            scale = float(np.mean([np.sqrt(abs(np.linalg.det(j))) for _, j in before]))
        for k, name in enumerate(("pitch", "yaw")):
            after = where(c, name, step)
            result[c, :, k] = np.mean(
                [
                    np.linalg.solve(j, s1 - s0)
                    for (s0, j), (s1, _) in zip(before, after)
                ],
                axis=0,
            )
    return result * scale / step.to_value(u.arcsec)


def _fit_through_reference(
    frame: np.ndarray,
    value: np.ndarray,
    frame_reference: int,
    degree: int = 1,
    clip: float = 4.0,
    floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a polynomial in the frame index that vanishes at the reference frame.

    The reference frame is where every time-dependent term is zero by
    definition, so the polynomial has no constant.  Frames further than
    `clip` robust standard deviations from the fit, the dark ones at the
    ends of the flight in practice, are left out and the fit repeated.  A
    robust deviation estimated from thirty frames can come out several
    times too small, which would reject good frames, so it is never taken
    below `floor`, the precision of a single measurement.

    Parameters
    ----------
    frame
        The frame indices.
    value
        The measured values; a frame that could not be measured is NaN.
    frame_reference
        The frame at which the polynomial is zero.
    degree
        The degree of the polynomial.
    clip
        The threshold of the rejection, in robust standard deviations.
    floor
        The smallest standard deviation the rejection may assume.

    Returns
    -------
    The fitted values at every frame, whether each frame took part, and the
    robust scatter of the measured values about the fit.
    """
    t = np.asarray(frame, dtype=float) - frame_reference
    value = np.asarray(value, dtype=float)
    powers = np.stack([t**k for k in range(degree, 0, -1)], axis=-1)
    used = np.isfinite(value)
    fitted = np.zeros_like(value)
    scatter = np.nan
    for _ in range(5):
        if used.sum() <= degree:
            break
        coefficients, *_ = np.linalg.lstsq(powers[used], value[used], rcond=None)
        fitted = powers @ coefficients
        residual = value - fitted
        scatter = float(1.4826 * np.median(np.abs(residual[used])))
        keep = np.isfinite(value) & (np.abs(residual) <= clip * max(scatter, floor))
        if np.array_equal(keep, used):
            break
        used = keep
    return fitted, used, scatter


def fit_coregistration(
    reference: str | pathlib.Path | esis.optics.DistortionParameters,
    pointing: str | pathlib.Path | astropy.table.QTable,
    num_scene: int = 401,
    instrument: None | esis.optics.Instrument = None,
    frames: None | tuple[int, ...] = None,
    frame_reference: int = 15,
    degree: int = 2,
    anchor: int = 1,
    num_pass: int = 2,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> astropy.table.QTable:  # pragma: nocover
    """
    Register the channels with one another through the flight.

    With the pointing, the window drift and the defocus of every frame
    applied, each channel's sky still slides against the others' by a few
    tenths of a pixel between the ends of the flight: a translation of the
    whole image inside its window, the same at both lines and smooth in
    time.  The windows do not follow it, so it is not a
    motion of a grating or a camera, and no low-order figure of the primary
    reproduces its pattern over the channels; its origin is not understood.
    It is measured here and taken out as what it looks like, a pointing
    offset of each channel: in every frame the channels' skies are measured
    against the anchor's (:func:`esis.optics.measure_channel_shifts`), the
    shift of each channel is converted to the offset of its own pitch and
    yaw that removes it (:func:`_pointing_pattern`), and a polynomial in the
    frame index that vanishes at the reference frame is fit through the
    flight (:func:`_fit_through_reference`).  The offsets are then taken
    about their mean over the channels, so that the pointing of the payload
    fit against the scene is left where it was.  A second pass measures the
    channels again with the first fit applied and corrects it, which also
    checks the sign and the size of the first.

    Parameters
    ----------
    reference
        The reference parameters, or the path of their table.
    pointing
        The pointing table, or its path, see :func:`fit_distortion_pointing`.
        Any channel offsets it already carries are ignored.
    num_scene
        The number of samples along each axis of the resampled AIA scene,
        whose extent sets the sky grid.
    instrument
        The base model.  If :obj:`None`, the idealized as-built model.
    frames
        The frames to measure.  If :obj:`None`, every frame.
    frame_reference
        The frame the reference was fit to, where the offsets are zero.
    degree
        The degree of the polynomial in the frame index.  A line leaves
        0.06 px of the flight's shifts and a quadratic 0.04 px, about what
        one frame measures to.
    anchor
        The channel the others are measured against.
    num_pass
        The number of times the channels are measured.
    path
        If given, the history is written here as an ECSV table.
    directory
        A directory where the progress is logged.

    Returns
    -------
    One row per frame: the shift of every channel against the anchor as
    first measured and as measured in the last pass, the offsets that
    remove them, and ``pitch_channel`` and ``yaw_channel``, the fitted
    offsets that :func:`apply_coregistration` adds to the pointing table.
    """
    if not isinstance(reference, esis.optics.DistortionParameters):
        reference = esis.optics.DistortionParameters.from_file(reference)
    if not isinstance(pointing, astropy.table.QTable):
        pointing = astropy.table.QTable.read(pointing, format="ascii.ecsv")
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    if frames is None:
        frames = tuple(range(esis.flights.f1.data.level_1().shape["time"]))
    frames = tuple(sorted(frames))
    log = _logger(directory, "coregistration")
    wavelengths = _wavelengths_alignment()
    others = [c for c in range(num_channel) if c != anchor]

    pattern = _pointing_pattern(
        instrument,
        [
            frame_parameters(reference, pointing, frame_reference, c, False)
            for c in range(num_channel)
        ],
        wavelengths,
        anchor,
    )
    log(
        "sky shift per arcsecond of a channel's own pitch; yaw [px]: "
        + " | ".join(
            f"{p[0, 0]:+.2f}, {p[1, 0]:+.2f}; {p[0, 1]:+.2f}, {p[1, 1]:+.2f}"
            for p in pattern
        )
    )

    # every frame is read once: the frames themselves and the sky grid
    data = {}
    for t in frames:
        scene, observation = _frame(t, num_scene)
        # copies, so that the Level-1 cube each frame was cut from is freed
        data[t] = (
            [np.array(f) for f in _frames_by_axis(observation, num_channel)],
            _alignment.sky_grid(scene.inputs.position, 1600),
        )
        del scene, observation

    num = len(frames)
    # the motion on the sky, in pixels, that each channel's image needs
    # against the anchor's: as measured in each frame, and its fit
    motion = np.zeros((num, num_channel, 2))
    relative = np.zeros((num, num_channel, 2))
    offsets = np.zeros((num, num_channel, 2))  # arcseconds of pitch and yaw
    used = np.ones((num, num_channel, 2), dtype=bool)
    scatter = np.zeros((num_channel, 2))
    shifts_first = np.zeros((num, num_channel, 2))
    shifts_last = np.zeros((num, num_channel, 2))
    for k in range(num_pass):
        for i, t in enumerate(frames):
            frames_data, sky = data[t]
            linears = []
            for c in range(num_channel):
                p = frame_parameters(reference, pointing, t, c, False)
                p.pitch = p.pitch + offsets[i, c, 0] * u.arcsec
                p.yaw = p.yaw + offsets[i, c, 1] * u.arcsec
                model = p.to_instrument(instrument[dict(channel=c)])
                linears.append(
                    model.system.linearize(
                        wavelength=model.wavelength, degree=2, field_stop=True
                    )
                )
            fields, scales = esis.optics.measure_channel_shifts(
                linears, frames_data, sky, wavelengths, anchor=anchor
            )
            medians = esis.optics.median_shifts(fields, scales)
            for c in others:
                shifts_last[i, c] = medians[c]
                if k == 0:
                    shifts_first[i, c] = medians[c]
                # a measured shift s means the anchor shows at r + s what the
                # channel shows at r (see shift_fft): the channel's image sits
                # at -s, and the model's image must move there, on top of the
                # motion already applied
                motion[i, c] = relative[i, c] - medians[c]
            log(
                f"pass {k}, frame {t}: shifts vs channel {anchor} "
                + "; ".join(
                    f"{medians[c][0]:+.2f}, {medians[c][1]:+.2f}" for c in others
                )
                + " px"
            )
        rms = float(np.sqrt(np.nanmean(shifts_last[:, others] ** 2)))
        log(
            f"pass {k}: rms shift of the channels against channel {anchor} {rms:.3f} px"
        )
        for c in others:
            for j in range(2):
                relative[:, c, j], used[:, c, j], scatter[c, j] = (
                    _fit_through_reference(
                        np.array(frames),
                        motion[:, c, j],
                        frame_reference,
                        degree,
                        floor=0.05,
                    )
                )
        # about the mean of the channels, so that the payload's pointing, fit
        # against the scene with the channels as they were, stays where it is
        centered = relative - relative.mean(axis=1, keepdims=True)
        for c in range(num_channel):
            offsets[:, c] = np.linalg.solve(pattern[c], centered[:, c].T).T
        log(
            f"pass {k}: offsets at the first and the last frame, pitch; yaw [arcsec]: "
            + " | ".join(
                f"ch{c} {offsets[0, c, 0]:+.2f}, {offsets[-1, c, 0]:+.2f}; "
                f"{offsets[0, c, 1]:+.2f}, {offsets[-1, c, 1]:+.2f}"
                for c in range(num_channel)
            )
            + "; scatter of the measured motion about the fit [px]: "
            + ", ".join(f"{np.hypot(*scatter[c]):.3f}" for c in others)
        )

    table = astropy.table.QTable(
        dict(
            frame=np.array(frames),
            shift_x=shifts_first[..., 0] * u.pix,
            shift_y=shifts_first[..., 1] * u.pix,
            residual_x=shifts_last[..., 0] * u.pix,
            residual_y=shifts_last[..., 1] * u.pix,
            motion_x=motion[..., 0] * u.pix,
            motion_y=motion[..., 1] * u.pix,
            used=used.all(axis=-1),
            pitch_channel=offsets[..., 0] * u.arcsec,
            yaw_channel=offsets[..., 1] * u.arcsec,
        )
    )
    table.meta.update(
        dict(
            description=(
                "The registration of the ESIS-I channels with one another "
                "through the flight.  shift is the median tile shift of every "
                "channel's sky against the anchor's with the pointing, the "
                "window drift and the defocus applied, in detector pixels, and "
                "residual the same in the last pass, measured with the "
                "previous pass's offsets applied.  motion is the motion on the "
                "sky, in detector pixels, that each channel's image needs "
                "against the anchor's in each frame; a polynomial in the frame "
                "index that vanishes at the reference frame is fit through "
                "it, taken about the mean of the channels, and expressed as "
                "pitch_channel and yaw_channel, the offset of each channel's "
                "own pointing.  used marks the frames the fit kept.  The "
                "origin of the shifts is not understood: the windows do not "
                "follow them."
            ),
            provenance=(
                f"esis.flights.f1.optics.fit_coregistration(num_scene={num_scene}, "
                f"degree={degree}, anchor={anchor}, num_pass={num_pass})"
            ),
            frame_reference=int(frame_reference),
            anchor=int(anchor),
            pattern=[[[float(v) for v in row] for row in p] for p in pattern],
            scatter=[[float(v) for v in row] for row in scatter],
            rms_first=float(np.sqrt(np.nanmean(shifts_first[:, others] ** 2))),
            rms_last=float(np.sqrt(np.nanmean(shifts_last[:, others] ** 2))),
            date=datetime.datetime.now().isoformat(timespec="seconds"),
            versions=_versions(),
        )
    )
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def apply_coregistration(
    pointing: astropy.table.QTable,
    coregistration: astropy.table.QTable,
) -> astropy.table.QTable:
    """
    Add the channels' own pointing offsets to the pointing table.

    Parameters
    ----------
    pointing
        The pointing table, see :func:`fit_distortion_pointing`.
    coregistration
        The registration of the channels, see :func:`fit_coregistration`.

    Returns
    -------
    A copy of the pointing table with ``pitch_channel`` and ``yaw_channel``
    columns, one value per channel, which :func:`frame_parameters` and the
    flight factory add to each channel's pointing.

    Raises
    ------
    ValueError
        If a frame of the pointing table was not registered.
    """
    result = pointing.copy()
    frames = [int(t) for t in np.asarray(coregistration["frame"])]
    index = []
    for t in np.asarray(pointing["frame"]):
        if int(t) not in frames:
            raise ValueError(f"frame {int(t)} was not registered")
        index.append(frames.index(int(t)))
    for name in ("pitch_channel", "yaw_channel"):
        result[name] = coregistration[name][index]
    result.meta["coregistration"] = (
        "pitch_channel and yaw_channel are each channel's own pointing "
        "offset, added to the payload's, which registers the channels with "
        "one another through the flight; see fit_coregistration, whose "
        "provenance was " + str(coregistration.meta.get("provenance", "not recorded"))
    )
    return result


def fit_distortion_pointing(
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    device: None | str = None,
    frame_reference: int = 15,
    frames: None | tuple[int, ...] = None,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    merit: str = "correlation",
    drift: None | str | pathlib.Path | astropy.table.QTable = None,
    drift_degree: None | int = 3,
    defocus: None | str | pathlib.Path | astropy.table.QTable = None,
    relative: bool = True,
    free_roll: bool = False,
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
    merit
        The comparison the pointing maximizes, see
        :class:`esis.optics.LinearMerit`.
    drift
        The per-frame drift of the windows, see
        :func:`measure_window_edges`, or the path of its table.  If given,
        each frame's grating yaw and pitch are offset per channel so that
        its windows sit where the edges of that frame were measured, before
        the pointing is fit; the offsets are recorded in the result.
    drift_degree
        The degree of the polynomial in the frame index that smooths the
        drift through the flight, see :func:`_drift_smooth`; :obj:`None`
        applies each frame's drift as measured, and zero where a frame has
        no measured edges.
    defocus
        The defocus of the primary through the flight, see
        :func:`fit_defocus_history`, or the path of its table.  If given,
        each frame's ``z_primary`` is applied to every channel before the
        pointing is fit and is not itself fit: the proxy scene cannot tell
        a defocus from the internal alignment, but the channels can.
    relative
        Whether to take the offsets relative to `frame_reference` when it
        is among `frames`, see :func:`pointing_relative`.  A driver that
        fits one frame per job gathers the rows first and takes them
        relative once.
    free_roll
        Whether to fit the roll of the payload per frame as well.  The
        merit against the proxy scene cannot measure a roll to better than
        an arcminute, and a polish left free in it settles on its first
        step rather than on a value, so by default the roll is held at the
        reference's and the table's roll column is zero.
    frame_reference
        The frame the reference was fit to.  If it is among `frames`, its
        own offset is subtracted from every row, so that the table is
        relative to the reference frame and zero there; see
        :func:`pointing_relative`.
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
    common = ("pitch", "yaw", "roll") if free_roll else ("pitch", "yaw")
    index = [names.index(n) for n in common]
    index_roll = names.index("roll")
    index_defocus = names.index("z_primary")
    if defocus is not None and not isinstance(defocus, astropy.table.QTable):
        defocus = astropy.table.QTable.read(defocus, format="ascii.ecsv")
    index_grating = [names.index(n) for n in ("yaw_grating", "pitch_grating")]

    if drift is not None and not isinstance(drift, astropy.table.QTable):
        drift = astropy.table.QTable.read(drift, format="ascii.ecsv")
    if drift is not None and drift_degree is not None:
        shift_drift = _drift_smooth(drift, num_channel, drift_degree)
        measured = sorted(set(int(t) for t in np.asarray(drift["frame"])))
        log(
            f"window drift smoothed by a degree-{drift_degree} polynomial over "
            f"frames {measured[0]} to {measured[-1]}, held outside them"
        )
    elif drift is not None:
        shift_drift = functools.partial(_drift_shift, drift)
    sensitivity = None

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
                merit=merit,
            )
            for c in range(num_channel)
        ]
        x_full = [na.pack(p).ndarray for p in reference]
        offsets = np.zeros((num_channel, 2))
        shifts = np.zeros((num_channel, 2))
        z_primary = 0.0
        if defocus is not None:
            where = np.flatnonzero(np.asarray(defocus["frame"]) == t)
            if where.size:
                z_primary = float(defocus["z_primary"][where[0]].to_value(u.mm))
                for c in range(num_channel):
                    x_full[c][index_defocus] += z_primary
                log(f"frame {t}: defocus {1e3 * z_primary:+.1f} um applied")
        if drift is not None:
            if sensitivity is None:
                sensitivity = _drift_sensitivity(instrument, reference, merits)
                log(
                    "window motion per arcminute of grating yaw and pitch [px]: "
                    + "; ".join(f"{s[0]:.1f}, {s[1]:.1f}" for s in sensitivity)
                )
            for c in range(num_channel):
                shifts[c] = shift_drift(c, t)
                offsets[c] = shifts[c] / sensitivity[c]
                x_full[c][index_grating] += offsets[c]
            log(
                f"frame {t}: windows moved "
                + "; ".join(f"{s[0]:+.2f}, {s[1]:+.2f}" for s in shifts)
                + " px from the flight median"
            )

        def objective(y):
            values = []
            for c in range(num_channel):
                x = x_full[c].copy()
                x[index] = x[index] + y
                values.append(merits[c](x))
            return float(np.mean(values))

        lower, upper = esis.flights.f1.optics.distortion_fit_bounds(reference[0])
        half = 0.5 * (na.pack(upper).ndarray - na.pack(lower).ndarray)[index]
        log(f"frame {t}: start {-objective(np.zeros(len(index))):.4f}")
        y, fun, num = esis.optics.polish(
            objective, np.zeros(len(index)), -half, half, scale=0.01, log=log
        )
        log(
            f"frame {t}: {-fun:.4f} after {num} evaluations, offset "
            + ", ".join(f"{n} {v:+.3f}" for n, v in zip(common, y))
        )
        offset = np.zeros(3)
        offset[: len(y)] = y
        if not free_roll:
            offset[2] = x_full[0][index_roll] - x_full[0][index_roll]  # held: zero
        rows.append((t, offset, shifts, offsets, z_primary))

    table = astropy.table.QTable(
        dict(
            frame=np.array([r[0] for r in rows]),
            pitch=np.array([r[1][0] for r in rows]) * u.arcsec,
            yaw=np.array([r[1][1] for r in rows]) * u.arcsec,
            roll=np.array([r[1][2] for r in rows]) * u.deg,
            z_primary=np.array([r[4] for r in rows]) * u.mm,
            drift_x=np.array([r[2][:, 0] for r in rows]) * u.pix,
            drift_y=np.array([r[2][:, 1] for r in rows]) * u.pix,
            yaw_grating=np.array([r[3][:, 0] for r in rows]) * u.arcmin,
            pitch_grating=np.array([r[3][:, 1] for r in rows]) * u.arcmin,
        )
    )
    table.meta.update(
        dict(
            description=(
                "Fitted per-frame payload pointing offsets of the ESIS-I "
                "flight, relative to the reference fit, one row per frame "
                "of esis.flights.f1.data.level_1(). The offsets are common "
                "to all four channels (a rigid-payload model); z_primary is "
                "the defocus of the primary at the field stop, measured from "
                "the channels against one another and applied here, which "
                "moves every channel's sky along its dispersion without "
                "moving the windows.  drift_x and "
                "drift_y are the measured motion of each channel's windows "
                "from their flight median, and yaw_grating and pitch_grating "
                "the per-channel grating offsets that reproduce it."
            ),
            provenance=(
                "esis.flights.f1.optics.fit_distortion_pointing("
                f"num_scene={num_scene}, drift_degree={drift_degree}): a local "
                "polish of the mean linear merit of the four channels in "
                + (
                    "pitch, yaw and roll"
                    if free_roll
                    else "pitch and yaw, the roll held"
                )
                + ", per frame, with the measured defocus applied"
            ),
        )
    )
    if relative and frame_reference in frames:
        table = pointing_relative(table, frame_reference)
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def _channel(
    parameters: esis.optics.DistortionParameters,
    channel: int,
    axis: str = "channel",
) -> esis.optics.DistortionParameters:
    """
    Take one channel's parameters out of a stacked set.

    Parameters
    ----------
    parameters
        Parameters with a logical axis over the channels, as
        :meth:`esis.optics.DistortionParameters.from_file` loads them.
    channel
        The index along that axis.
    axis
        The name of that axis.
    """
    fields = {}
    for field in dataclasses.fields(parameters):
        value = getattr(parameters, field.name)
        if isinstance(value, na.AbstractArray) and axis in value.shape:
            value = value[{axis: channel}]
        fields[field.name] = value
    return type(parameters)(**fields)


def _dispersion(
    linear,
    wavelength: u.Quantity,
    axis: tuple[str, str] = ("sky_x", "sky_y"),
) -> tuple[np.ndarray, float]:
    """
    Find where a channel disperses on the sky at one line, and how fast.

    Parameters
    ----------
    linear
        The channel's linearized system.
    wavelength
        The line.
    axis
        The logical axes of the sky grid.

    Returns
    -------
    The unit vector on the sky along which a redshift displaces the image,
    in the sky grid's axes, and the velocity a displacement of one detector
    pixel along it stands for, in km/s.
    """
    origin = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 10.0, 0.0]) * u.arcsec, axes="s"),
        y=na.ScalarArray(np.array([0.0, 0.0, 10.0]) * u.arcsec, axes="s"),
    )
    step = 0.1 * u.AA
    x, y = _alignment.sensor_coordinates(
        linear.distortion, origin, wavelength, ("s", "s")
    )
    x1, y1 = _alignment.sensor_coordinates(
        linear.distortion, origin, wavelength + step, ("s", "s")
    )
    x, y, x1, y1 = (np.asarray(v).ravel() for v in (x, y, x1, y1))
    # detector pixels per arcsecond of sky along each sky axis
    jacobian = np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]]) / 10.0
    shift = np.array([x1[0] - x[0], y1[0] - y[0]])
    on_sky = np.linalg.solve(jacobian, shift)
    pixels_per_angstrom = float(np.hypot(*shift) / step.to_value(u.AA))
    velocity = float(
        (astropy.constants.c / pixels_per_angstrom * u.AA / wavelength).to_value(
            u.km / u.s
        )
    )
    return on_sky / np.linalg.norm(on_sky), velocity


def _mean_column(table: astropy.table.QTable, name: str) -> float:
    """Average a column over its rows, ignoring NaN, whatever unit it carries."""
    column = table[name]
    return float(np.nanmean(column.value if hasattr(column, "value") else column))


def acceptance(
    reference: esis.optics.DistortionParameters,
    pointing: astropy.table.QTable,
    edges: None | astropy.table.QTable = None,
    frames: tuple[int, ...] = (9, 12, 15, 18, 21, 24),
    frames_shifts: None | tuple[int, ...] = None,
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    device: None | str = None,
    merit: str = "correlation",
    num_sky: int = 1600,
    num_tile: int = 8,
    fraction_valid: float = 0.8,
    anchor: int = 1,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> astropy.table.QTable:  # pragma: nocover
    """
    Score a reference fit on frames it was not fit to, without an inversion.

    The coalignment metric is, for every frame, line and channel, the
    length of the median tile shift of the channel's sky against the
    anchor's (``shift_<line>``), which is how far the channel is
    misregistered as a whole, and the rms scatter of the tiles about that
    median (``scatter_<line>``), which is how far the distortion within
    the frame still differs between the two.  Both are in detector pixels
    on the anchor's scale; a pixel is about 17 km/s along the dispersion.


    Three measurements per frame and channel: the merit against the AIA
    proxy scene with that frame's pointing applied, the median tile shift
    of the channel against the anchor on the sky at each aligned line,
    which is the internal consistency of the mapping and, compared across
    lines, of its dispersion, and the residual of the reference's window
    outlines against the measured edges.

    Parameters
    ----------
    reference
        The reference parameters of every channel.
    pointing
        The per-frame pointing, see :func:`fit_distortion_pointing`.
    edges
        The measured window edges, see :func:`measure_window_edges`.
    frames_shifts
        The frames whose coalignment alone is measured, which needs no
        proxy scene and so is cheap; :obj:`None` measures every frame of
        the flight.  The rows of frames not in `frames` carry no merit.
    frames
        The frames to score.
    instrument
        The instrument model, see :func:`fit_distortion_reference`.
    num_scene
        The number of samples along each axis of the resampled scene.
    device
        The device the merit runs on.
    merit
        The comparison to report, see :class:`esis.optics.LinearMerit`.
    num_sky
        The number of samples along each axis of the common sky grid.
    num_tile
        The number of tiles along each axis when measuring shifts.
    fraction_valid
        The fraction of a tile which must lie inside both windows for its
        shift to count.  Looser than the alignment's, since the octagon's
        diagonal sides and the gap between windows cut into many tiles,
        and a tile's pixels outside the window take its own level.
    anchor
        The channel the others are compared with.
    path
        Where to save the table, if anywhere.
    directory
        The directory the log is written to.

    Returns
    -------
    A table with a row per frame and channel: the correlation, the
    least-squares score, and the median shift against the anchor at each
    aligned line in pixels; the outline residual per channel is in the
    metadata.
    """
    from esis.optics._distortions import _alignment

    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    log = _logger(directory, "acceptance")
    lines = _wavelengths_alignment()
    if frames_shifts is None:
        frames_shifts = tuple(range(esis.flights.f1.data.level_1().shape["time"]))
    rows = []
    mode_rows = []
    tile_rows = []
    velocity = {}
    for t in sorted(set(frames) | set(frames_shifts)):
        scored = t in frames
        scene, observation = _frame(t, num_scene)
        merits, linears = [], []
        for c in range(num_channel):
            p = frame_parameters(reference, pointing, t, c)
            channel = instrument[dict(channel=c)]
            model = p.to_instrument(channel)
            if scored:
                m = esis.optics.LinearMerit(
                    instrument=channel,
                    parameters=p,
                    scene=scene,
                    observation=observation[dict(channel=c)],
                    device=device,
                    merit=merit,
                )
                merits.append((m, p))
                linears.append(m.linearize(model))
            else:
                linears.append(
                    model.system.linearize(
                        wavelength=model.wavelength, degree=2, field_stop=True
                    )
                )
        scores = (
            [(m.correlation(p), m.score(p)) for m, p in merits]
            if scored
            else [(np.nan, np.nan)] * num_channel
        )
        frames_data = _frames_by_axis(observation, num_channel)
        sky = _alignment.sky_grid(scene.inputs.position, num_sky)
        shifts = {c: {} for c in range(num_channel)}
        scatter = {c: {} for c in range(num_channel)}
        along = {c: {} for c in range(num_channel)}
        for name, wavelength in lines.items():
            dispersion = {
                c: _dispersion(linears[c], wavelength) for c in range(num_channel)
            }
            velocity[name] = float(
                np.mean([dispersion[c][1] for c in range(num_channel)])
            )
            fields, scales = _alignment.measure_channel_shifts(
                linears,
                frames_data,
                sky,
                {name: wavelength},
                anchor,
                num_tile,
                fraction_valid=fraction_valid,
            )
            scale = float(np.mean(scales))
            for c, per_line in fields.items():
                modes = _alignment.shift_modes(per_line, scale, num_sky)
                mode_rows.append((t, c, name, *[modes[k] for k in _MODE_KEYS]))
                for field in per_line:
                    for k in range(field.dx.size):
                        tile_rows.append(
                            (
                                t,
                                c,
                                name,
                                field.i[k],
                                field.j[k],
                                field.dx[k] * scale,
                                field.dy[k] * scale,
                            )
                        )
            for c in range(num_channel):
                if c == anchor:
                    shifts[c][name] = 0.0
                    scatter[c][name] = 0.0
                    along[c][name] = 0.0
                    continue
                if not fields.get(c):
                    # nothing measurable is not perfect alignment
                    shifts[c][name] = np.nan
                    scatter[c][name] = np.nan
                    along[c][name] = np.nan
                    continue
                dx = np.concatenate([f.dx for f in fields[c]])
                dy = np.concatenate([f.dy for f in fields[c]])
                # the length of the median shift, not the median length: the
                # latter never falls below the noise of a single tile
                shifts[c][name] = float(np.hypot(np.median(dx), np.median(dy)) * scale)
                # the part of it along the channel's own dispersion is the
                # part a velocity would be charged for
                along[c][name] = float(
                    abs(np.array([np.median(dx), np.median(dy)]) @ dispersion[c][0])
                    * scale
                )
                scatter[c][name] = float(
                    np.sqrt(
                        np.mean((dx - np.median(dx)) ** 2 + (dy - np.median(dy)) ** 2)
                    )
                    * scale
                )
        for c in range(num_channel):
            rows.append(
                (
                    t,
                    c,
                    scores[c][0],
                    scores[c][1],
                    *[shifts[c][name] for name in lines],
                    *[scatter[c][name] for name in lines],
                    *[along[c][name] for name in lines],
                )
            )
            log(
                f"frame {t} channel {c}: correlation {scores[c][0]:.4f}, "
                f"score {scores[c][1]:.4f}, shift "
                + ", ".join(f"{name} {shifts[c][name]:.2f} px" for name in lines)
                + ", scatter "
                + ", ".join(f"{name} {scatter[c][name]:.2f} px" for name in lines)
            )
    table = astropy.table.QTable(
        rows=rows,
        names=("frame", "channel", "correlation", "score")
        + tuple(f"shift_{name.replace(' ', '_')}" for name in lines)
        + tuple(f"scatter_{name.replace(' ', '_')}" for name in lines)
        + tuple(f"along_{name.replace(' ', '_')}" for name in lines),
    )
    for name in lines:
        table[f"shift_{name.replace(' ', '_')}"].unit = u.pix
        table[f"scatter_{name.replace(' ', '_')}"].unit = u.pix
        table[f"along_{name.replace(' ', '_')}"].unit = u.pix
    modes = astropy.table.QTable(
        rows=mode_rows or None,
        names=("frame", "channel", "line", *_MODE_KEYS),
    )
    for key in _MODE_KEYS:
        if key != "correlation":
            modes[key].unit = u.pix
    tiles = astropy.table.QTable(
        rows=tile_rows or None,
        names=("frame", "channel", "line", "i", "j", "dx", "dy"),
    )
    for key in ("dx", "dy"):
        tiles[key].unit = u.pix
    if path is not None:
        base = pathlib.Path(path)
        modes.meta.update(
            dict(
                description=(
                    "The tile shifts of every channel against the anchor "
                    "decomposed into optical modes: the translation, and the "
                    "shift at the field's edge that a magnification, a rotation, "
                    "an anisotropy and a shear would make, with the scatter of "
                    "the tiles about the median, the residual after the linear "
                    "modes, and the correlation of neighbouring tiles' residuals, "
                    "which is near zero when what is left is noise."
                ),
                anchor=anchor,
            )
        )
        modes.write(
            base.with_name(base.stem + "_modes.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
        tiles.meta.update(
            dict(
                description=(
                    "Every tile shift behind the coalignment metric: the "
                    "sky-grid indices of the tile's centre and the shift of the "
                    "channel's sky against the anchor's, in detector pixels."
                ),
                anchor=anchor,
                num_sky=num_sky,
                num_tile=num_tile,
                fraction_valid=fraction_valid,
            )
        )
        tiles.write(
            base.with_name(base.stem + "_tiles.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
    for name in lines:
        own = modes[
            (np.asarray(modes["line"]) == name)
            & (np.asarray(modes["channel"]) != anchor)
        ]
        if len(own):
            log(
                f"modes at {name}, mean over frames and channels: "
                + ", ".join(f"{k} {_mean_column(own, k):.3f}" for k in _MODE_KEYS)
            )
    others = np.asarray(table["channel"]) != anchor
    coalignment = {}
    for name in lines:
        key = name.replace(" ", "_")
        shift = np.asarray(table[f"shift_{key}"].value)[others]
        spread = np.asarray(table[f"scatter_{key}"].value)[others]
        component = np.asarray(table[f"along_{key}"].value)[others]
        own = modes[
            (np.asarray(modes["line"]) == name)
            & (np.asarray(modes["channel"]) != anchor)
        ]
        coalignment[name] = dict(
            rms=float(np.sqrt(np.nanmean(shift**2))),
            max=float(np.nanmax(shift)),
            scatter=float(np.nanmean(spread)),
            rms_along_dispersion=float(np.sqrt(np.nanmean(component**2))),
            max_along_dispersion=float(np.nanmax(component)),
            km_per_s_per_pixel=velocity.get(name, float("nan")),
            rms_velocity=float(
                np.sqrt(np.nanmean(component**2)) * velocity.get(name, float("nan"))
            ),
            max_velocity=float(np.nanmax(component) * velocity.get(name, float("nan"))),
            **(
                {
                    k: _mean_column(own, k)
                    for k in _MODE_KEYS
                    if k not in ("translation", "scatter")
                }
                if len(own)
                else {}
            ),
        )
        log(
            f"coalignment at {name} over {len(set(table['frame']))} frames: "
            f"rms {coalignment[name]['rms']:.3f} px, "
            f"max {coalignment[name]['max']:.3f} px, "
            f"tile scatter {coalignment[name]['scatter']:.3f} px"
        )
    outline = None
    if edges is not None:
        scene, observation = _frame(15, num_scene)
        outline = []
        for c in range(num_channel):
            p = _channel(reference, c)
            channel = instrument[dict(channel=c)]
            m = esis.optics.LinearMerit(
                instrument=channel,
                parameters=p,
                scene=scene,
                observation=observation[dict(channel=c)],
                device=device,
            )
            linear = m.linearize(p.to_instrument(channel))
            wavelength = channel.wavelength
            footprints = [
                linear.footprint(wavelength[dict(wavelength=i)])
                for i in range(na.shape(wavelength)["wavelength"])
            ]
            own = edges[np.asarray(edges["channel"]) == c]
            outline.append(
                dict(
                    residual=esis.optics.outline_residual(footprints, own),
                    width=esis.optics.width_residual(footprints, own),
                )
            )
            log(
                f"channel {c}: outline residual {outline[-1]['residual']:.2f} px, "
                f"width residual {outline[-1]['width']:.2f} px"
            )
    table.meta.update(
        dict(
            description=(
                "The reference fit scored on frames of the flight: the merit "
                "with each frame's pointing applied, the median tile shift of "
                f"every channel against channel {anchor} on the sky at each "
                "aligned line, and the residual of the window outlines against "
                "the measured edges."
            ),
            frames=list(frames),
            frames_shifts=list(frames_shifts),
            anchor=anchor,
            coalignment=coalignment,
            metric=(
                "shift_<line>: the length of the median tile shift of a "
                "channel's sky against the anchor's, in detector pixels on the "
                "anchor's scale, a whole-channel misregistration; scatter_<line>: "
                "the rms of the tiles about that median, the distortion left "
                "between the two; along_<line>: the component of the median "
                "shift along the channel's own dispersion, the part a velocity "
                "is charged for; coalignment: the rms and maximum of the shift, "
                "the mean scatter, and the rms and maximum along the dispersion "
                "in pixels and in km/s, over every frame and channel but the "
                "anchor"
            ),
            outline=outline,
            date=datetime.datetime.now().isoformat(timespec="seconds"),
            versions=_versions(),
        )
    )
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def pointing_relative(table: astropy.table.QTable, frame: int) -> astropy.table.QTable:
    """
    Make a table of pointing offsets relative to one of its frames.

    The reference fit carries the absolute pointing of the frame it was fit
    to, and a per-frame fit of the pointing alone lands a fraction of an
    arcsecond from it, since it optimizes the mean merit of the four
    channels in three terms rather than every channel in every term.  The
    offsets are therefore taken relative to the reference frame's own fit,
    which puts that frame at zero and the others where the per-frame fits
    place them with respect to it.

    Parameters
    ----------
    table
        A table with ``frame``, ``pitch``, ``yaw`` and ``roll`` columns.
    frame
        The frame to take as the origin.

    Raises
    ------
    ValueError
        If `frame` is not in the table.
    """
    where = np.flatnonzero(np.asarray(table["frame"]) == frame)
    if where.size != 1:
        raise ValueError(f"frame {frame} is not in the table")
    result = table.copy()
    origin = dict()
    for name in (
        "pitch",
        "yaw",
        "roll",
        "z_primary",
        "drift_x",
        "drift_y",
        "yaw_grating",
        "pitch_grating",
    ):
        if name not in table.colnames:
            continue
        origin[name] = table[name][where[0]]
        result[name] = table[name] - origin[name]
    result.meta["frame_reference"] = int(frame)
    result.meta["offset_reference"] = (
        f"the fit of frame {frame} alone landed at pitch {origin['pitch']:.3f}, "
        f"yaw {origin['yaw']:.3f}, roll {origin['roll']:.5f} from the reference, "
        "which is subtracted from every row"
    )
    return result

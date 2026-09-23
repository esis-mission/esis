"""The reproducible distortion fit of the ESIS-I flight data."""

from __future__ import annotations
from typing import Callable
import copy
import dataclasses
import datetime
import importlib.metadata
import multiprocessing
import pathlib
import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na
import optika
import esis
from esis.optics._distortions import _fit

__all__ = [
    "measure_window_edges",
    "fit_distortion_reference",
    "fit_distortion_pointing",
]

_SHARED = ("displacement_primary", "roll_field_stop", "pitch", "yaw")

# the fields that do not move the mapping, which the alignment cannot use
_PHOTOMETRIC = ("degradation",)

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

These were meant to stay at their as-built values until the internal
alignment, since against the proxy scene they are partly degenerate with
the pointing and the grating.  On the as-built model of the flight they
cannot: with them held, the absolute fit of three channels of four stalls
at 0.70 to 0.76 in correlation where freeing them reaches 0.78 to 0.82, and
freeing the focus-preserving distance alone does not close the gap, nor
does the design model in place of the as-built one.  The absolute stage
therefore fits every parameter, and the internal alignment still has the
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


def _versions() -> dict[str, str]:
    """Record the versions of the packages the fit ran on."""
    result = {}
    for name in (
        "euv-snapshot-imaging-spectrograph",
        "optika",
        "named-arrays",
        "regridding",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unknown"
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
    scores = dict(absolute=[None] * num_channel)

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
                versions=_versions(),
                date=datetime.datetime.now().isoformat(timespec="seconds"),
            ),
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
                merit=merit,
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
    if frame_reference in frames:
        table = pointing_relative(table, frame_reference)
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
    for name in ("pitch", "yaw", "roll"):
        origin[name] = table[name][where[0]]
        result[name] = table[name] - origin[name]
    result.meta["frame_reference"] = int(frame)
    result.meta["offset_reference"] = (
        f"the fit of frame {frame} alone landed at pitch {origin['pitch']:.3f}, "
        f"yaw {origin['yaw']:.3f}, roll {origin['roll']:.5f} from the reference, "
        "which is subtracted from every row"
    )
    return result

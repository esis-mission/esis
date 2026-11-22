"""The reproducible distortion fit of the ESIS-I flight data."""

from __future__ import annotations
from typing import Callable
import copy
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
    "measure_field_stop",
    "fit_distortion_reference",
    "fit_distortion_channel",
    "fit_defocus_history",
    "frame_parameters",
    "fit_distortion_pointing",
    "acceptance",
]

_SHARED = ("displacement_primary", "z_primary", "roll_field_stop", "pitch", "yaw")
"""
The parameters that belong to the instrument rather than to a channel.

One primary mirror, one field stop, one payload pointing: a fit which lets
them differ per channel is using them as stand-ins for the camera placement.
"""

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

The nine terms one frame determines with no null direction; a larger set
adds directions the frame cannot see, and the capture lands elsewhere from
every seed.
"""

_WINDOW = ("yaw_grating", "pitch_grating", "roll_field_stop")
"""
The terms the outline stage fits to the window edges.

The grating's yaw and pitch move the image of the field stop on the sensor
with the sky; the merit sees that only as a trade with the pointing, the
edges see it directly.  The roll of the field stop turns the outline without
moving the sky at all, which only the edges can see.  The roll of the sensor,
which turns both, is left to the merit.
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

The grating and camera placement, which let one channel's mapping differ
from another's by a scale, an anisotropy and a rotation.  The instrument
roll is not among them, see :attr:`esis.optics.DistortionParameters.roll`.
"""

_LINES_ALIGNMENT = ("He I", "O V")
"""The bright, isolated lines the channels are aligned on."""


def _idealized(instrument: esis.optics.Instrument) -> esis.optics.Instrument:
    """
    Replace the material models of an instrument with ideal ones.

    The distortion fit measures geometry, not throughput, so the coatings,
    filter and sensor are made ideal to save raytracing time, and the
    uncertainty wrappers are stripped so the distortion parameters are plain
    scalars.

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


path_field_stop: None | str | pathlib.Path = None
"""
The table of vertices of the field stop the fit is made with.

:obj:`None`, the default, for the regular octagon of the design, which the
first pass of the fit starts from; a table such as the output of
:func:`measure_field_stop` for each later pass, which iterates the
measurement (``docs/reports/distortion_fit/reproduce.py`` sets it).
"""


def _base(instrument: None | esis.optics.Instrument) -> esis.optics.Instrument:
    if instrument is None:
        instrument = _idealized(esis.flights.f1.optics.as_built(num_distribution=0))
        instrument.wavelength = _wavelength_lines()
        # the measured polygon, or the regular octagon of the design
        if path_field_stop is not None:
            instrument.field_stop.vertices = esis.flights.f1.optics.field_stop_vertices(
                path_field_stop
            )
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


def _read_table(
    table: None | str | pathlib.Path | astropy.table.QTable,
) -> None | astropy.table.QTable:
    """Read an ECSV table given by path, passing a table or :obj:`None` through."""
    if table is None or isinstance(table, astropy.table.QTable):
        return table
    return astropy.table.QTable.read(table, format="ascii.ecsv")


def _map_channels(function: Callable, jobs: list, workers: int) -> list:
    """Apply a function to each channel's job, side by side if workers allow."""
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(function, jobs)
    return [function(job) for job in jobs]


def _footprints(linear, wavelength: na.AbstractScalar) -> list:
    """Outline the window of every line of a linearized channel on the sensor."""
    return [
        linear.footprint(wavelength[dict(wavelength=i)])
        for i in range(na.shape(wavelength)["wavelength"])
    ]


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
        the seed of the merit's samples.

    Returns
    -------
    The polished parameters, their correlation, and the number of
    evaluations.
    """
    instrument, parameters, scene, observation, device, index_own, seed = job
    merit = esis.optics.LinearMerit(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        device=device,
        seed=seed,
    )
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
        The channel, its parameters, the scene, its frame, the device, its
        measured edges, and the seed of the merit's samples.

    Returns
    -------
    The parameters, their merit, the edge residual, and the log messages.
    """
    instrument, parameters, scene, observation, device, edges, seed = job
    objective = esis.optics.LinearMerit(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        device=device,
        seed=seed,
    )
    messages = []
    result = esis.optics.fit_distortion_outline(
        objective,
        edges,
        window=_WINDOW,
        sky=_SKY,
        bounds=esis.flights.f1.optics.distortion_fit_bounds(parameters),
        log=messages.append,
    )
    linear = objective.linearize(result.to_instrument(instrument))
    footprints = _footprints(linear, instrument.wavelength)
    residual = esis.optics.outline_residual(footprints, edges)
    return result, -objective(na.pack(result).ndarray), residual, messages


def _center(footprint: na.AbstractCartesian2dVectorArray) -> list[float]:
    """
    Locate the centre of a window's outline, in pixels.

    The centre of the bounding box rather than the mean of the vertices,
    which sit wherever the library chose to sample the outline.
    """
    x = np.asarray(na.value(footprint.x).ndarray, dtype=float)
    y = np.asarray(na.value(footprint.y).ndarray, dtype=float)
    return [float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2)]


def _window_centers(
    instrument: esis.optics.Instrument,
    parameters: list[esis.optics.DistortionParameters],
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    seed: int,
) -> list[list[list[float]]]:  # pragma: nocover
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
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

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
            seed=seed,
        )
        linear = merit.linearize(p.to_instrument(channel))
        footprints = _footprints(linear, channel.wavelength)
        result.append([_center(f) for f in footprints])
    return result


def _environment() -> dict:
    """
    Record the environment a table was made in.

    The interpreter, the platform, the version of every installed
    distribution, and the checkouts of the packages the fit depends on
    (:func:`_versions`).
    """
    import platform
    import sys

    packages = {}
    for d in importlib.metadata.distributions():
        name = d.metadata["Name"]
        if name:
            packages[name] = d.version
    return dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=dict(sorted(packages.items(), key=lambda kv: kv[0].lower())),
        checkouts=_versions(),
    )


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
        except importlib.metadata.PackageNotFoundError:  # pragma: nocover
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
                    version = version + " +dirty"
        except Exception:  # pragma: nocover
            pass
        result[name] = version
    return result


def _stamp(seed: int) -> dict:
    """Date, environment and seed, which every table's metadata carries."""
    return dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        environment=_environment(),
        seed=seed,
    )


def _groups(
    table: astropy.table.QTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group crossings by channel, line, side and row or column.

    Returns
    -------
    The unique keys, and the order and bounds that list each group's
    members: group ``g`` is ``order[bounds[g]:bounds[g + 1]]``.
    """
    keys = np.stack(
        [np.asarray(table[k]) for k in ("channel", "line", "side", "index")],
        axis=1,
    )
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(unique) + 1))
    return unique, order, bounds


def window_drift(
    crossings: astropy.table.QTable,
    num_channel: int,
    error_max: u.Quantity = 1 * u.pix,
) -> astropy.table.QTable:
    """
    Measure the motion of every window's sides through the flight.

    Each row or column's crossing of a side is taken against its own median
    over the flight, and the median of those departures over the side's
    rows or columns is the side's motion in that frame.

    Parameters
    ----------
    crossings
        The crossings of every frame, with ``channel``, ``frame``, ``line``,
        ``side``, ``index``, ``position`` and ``error`` columns, such as
        :func:`esis.optics.measure_edge_steps` or
        :func:`esis.optics.measure_edges` give them with the frame and the
        channel added.
    num_channel
        The number of channels.
    error_max
        Crossings with a larger formal error are left out.

    Returns
    -------
    One row per channel, frame and side: the ``shift`` from the flight
    median, and the ``num`` of crossings it is the median of.
    """
    every = crossings[crossings["error"] < error_max]
    _, order, bounds = _groups(every)
    position = np.asarray(every["position"].to_value(u.pix))
    center = np.empty(len(every))
    for g in range(len(bounds) - 1):
        members = order[bounds[g] : bounds[g + 1]]
        center[members] = np.median(position[members])
    departure = position - center
    channel = np.asarray(every["channel"])
    frame = np.asarray(every["frame"])
    side = np.asarray(every["side"])
    rows = []
    for c in range(num_channel):
        for t in sorted(set(int(v) for v in frame[channel == c])):
            for k in range(len(esis.optics.SIDES)):
                sel = (channel == c) & (frame == t) & (side == k)
                if sel.any():
                    median = float(np.median(departure[sel]))
                    rows.append((c, t, k, median, int(sel.sum())))
    result = astropy.table.QTable(
        rows=rows,
        names=("channel", "frame", "side", "shift", "num"),
        dtype=(int, int, int, float, int),
    )
    result["shift"].unit = u.pix
    return result


def _refer_to_frame(
    edges: astropy.table.QTable,
    drift: astropy.table.QTable,
    frame: int,
    degree: int,
) -> tuple[astropy.table.QTable, astropy.table.QTable]:
    """
    Refer the median edges of the windows and their drift to one frame.

    The edges are moved by the drift at the frame, as :func:`_drift_smooth`
    smooths it, and the drift is taken from there, so that the smoothed
    drift is zero at the frame.  The left and right sides move with x, the
    top and bottom with y, as :func:`_drift_shift` reads them.

    Parameters
    ----------
    edges
        The median edges, one row per channel, line, side and row or column.
    drift
        The drift from the median, see :func:`window_drift`.
    frame
        The frame to refer them to.
    degree
        The degree of the polynomial that smooths the drift.

    Returns
    -------
    Copies of the edges and the drift.
    """
    edges = edges.copy()
    drift = drift.copy()
    channels = sorted(set(int(c) for c in np.asarray(drift["channel"])))
    smooth = _drift_smooth(drift, max(channels) + 1, degree)
    for c in channels:
        origin = smooth(c, frame)
        for k in range(len(esis.optics.SIDES)):
            offset = (origin[0] if k < 2 else origin[1]) * u.pix
            mine = (np.asarray(edges["channel"]) == c) & (
                np.asarray(edges["side"]) == k
            )
            edges["position"][mine] += offset
            mine = (np.asarray(drift["channel"]) == c) & (
                np.asarray(drift["side"]) == k
            )
            drift["shift"][mine] -= offset
    return edges, drift


def measure_window_edges(
    parameters: list[esis.optics.DistortionParameters],
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    frames: None | tuple[int, ...] = None,
    frame_reference: int = 15,
    threshold: float = 0.5,
    lines: tuple[int, ...] = (0, 2),
    error_max: u.Quantity = 1 * u.pix,
    drift_degree: int = 3,
    directory: None | str | pathlib.Path = None,
    path: None | str | pathlib.Path = None,
    path_drift: None | str | pathlib.Path = None,
    seed: int = 0,
) -> tuple[astropy.table.QTable, astropy.table.QTable]:  # pragma: nocover
    """
    Measure the edges of every window at the reference frame, and their drift.

    The windows are the image of the field stop, and do not move with the
    pointing, so the same edge is measured once per frame against a
    different backdrop of solar structure each time.  Each side of each
    window is fit in each frame as one blurred straight edge, see
    :func:`esis.optics.measure_edges`, and the median over the frames, with
    the median width of its blur, places it.

    The windows do drift, by a few tenths of a pixel through the flight.
    The drift is read from a different fit: a step fit to every row and
    column on its own, see :func:`esis.optics.measure_edge_steps` and
    :func:`window_drift`.  The straight edges place a side better, but
    structure crossing a side as the pointing drifts moves its straight edge
    by a few tenths of a pixel from frame to frame, more than the median of
    the rows' changes; on the flight the two sides of one window move apart
    by 0.14--0.39 px rms with the straight edges and 0.06--0.20 px with the
    steps.

    Both are referred to the reference frame, which the sky of the reference
    fit is fit to: the median edges are moved by the drift at that frame,
    as the later stages smooth it (:func:`_drift_smooth`), and the drift is
    taken from there.  The reference then places its windows where they
    were at that frame, and the drift is zero there.

    Parameters
    ----------
    parameters
        The parameters of every channel, which place the outlines the
        edges are searched near.
    instrument
        The instrument model, see :func:`fit_distortion_channel`.
    num_scene
        The number of samples along each axis of the resampled scene, which
        sets the grid the channel is linearized on.
    frames
        The frames to measure.  If :obj:`None`, every frame whose mean
        signal exceeds `threshold` times the median over the flight.
    frame_reference
        The frame the reference is fit to.
    threshold
        The fraction of the flight's median signal below which a frame is
        left out.
    lines
        The indices of the spectral lines whose windows are measured; the
        middle window of ESIS-I sits on its neighbours and is left out.
    error_max
        Crossings with a larger formal error are left out.
    drift_degree
        The degree of the polynomial in the frame index that smooths the
        drift, see :func:`_drift_smooth`.
    directory
        The directory the log is written to.
    path
        Where to save the edges, if anywhere.
    path_drift
        Where to save the per-frame drift, if anywhere.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    The edges at the reference frame, one row per channel, line, side and
    row or column, with the median width of the side; and the drift from
    the reference frame, one row per channel, frame and side.
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
    scene, observation = _frame(frame_reference, num_scene)
    tables = []
    tables_steps = []
    for c in range(num_channel):
        channel = instrument[dict(channel=c)]
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=parameters[c],
            scene=scene,
            observation=observation[dict(channel=c)],
            seed=seed,
        )
        linear = merit.linearize(parameters[c].to_instrument(channel))
        footprints = _footprints(linear, channel.wavelength)
        for t in frames:
            for found, measure in (
                (tables, esis.optics.measure_edges),
                (tables_steps, esis.optics.measure_edge_steps),
            ):
                table = measure(data[t, c], footprints)
                table = table[np.isin(table["line"], lines)]
                table["frame"] = t
                table["channel"] = c
                found.append(table)
            log(
                f"frame {t} channel {c}: {len(tables[-1])} crossings, "
                f"{len(tables_steps[-1])} steps"
            )

    every = astropy.table.vstack(tables)
    every = every[every["error"] < error_max]
    unique, order, bounds = _groups(every)
    position = np.asarray(every["position"].to_value(u.pix))
    width = np.asarray(every["width"].to_value(u.pix))
    rows = []
    for g in range(len(unique)):
        members = order[bounds[g] : bounds[g + 1]]
        values = position[members]
        median = float(np.median(values))
        spread = 1.4826 * float(np.median(np.abs(values - median)))
        error = spread / np.sqrt(len(values))
        rows.append((*unique[g], median, error, np.median(width[members]), len(values)))
    edges = astropy.table.QTable(
        rows=rows,
        names=("channel", "line", "side", "index", "position", "error", "width", "num"),
        dtype=(int, int, int, int, float, float, float, int),
    )

    for name in ("position", "error", "width"):
        edges[name].unit = u.pix
    drift = window_drift(astropy.table.vstack(tables_steps), num_channel, error_max)
    edges, drift = _refer_to_frame(edges, drift, frame_reference, drift_degree)

    meta = dict(
        frames=list(frames),
        frame_reference=int(frame_reference),
        lines=list(lines),
        error_max=float(error_max.to_value(u.pix)),
        drift_degree=int(drift_degree),
        sides=list(esis.optics.SIDES),
        **_stamp(seed),
    )
    edges.meta.update(
        description=(
            "Where the edges of each window cross the rows and columns of the "
            "ESIS-I frames at the reference frame: each side fit as one "
            "blurred straight edge in every frame, the median over the frames "
            "moved by the smoothed drift at the reference frame, and the "
            "width of its blur."
        ),
        **meta,
    )
    drift.meta.update(
        description=(
            "The motion of each window's sides from the reference frame, per "
            "frame, from the steps fit to every row and column."
        ),
        **meta,
    )
    if path is not None:
        edges.write(path, format="ascii.ecsv", overwrite=True)
    if path_drift is not None:
        drift.write(path_drift, format="ascii.ecsv", overwrite=True)
    log(f"{len(edges)} edges, {len(drift)} drift rows")
    return edges, drift


class _Polynomial2d:
    """A cubic polynomial map of the plane, fit by least squares."""

    def __init__(self, x: np.ndarray, y: np.ndarray, u_: np.ndarray, v: np.ndarray):
        self.center = np.array([x.mean(), y.mean()])
        self.scale = max(np.ptp(x), np.ptp(y)) / 2
        a = self._design(x, y)
        self.cu = np.linalg.lstsq(a, u_, rcond=None)[0]
        self.cv = np.linalg.lstsq(a, v, rcond=None)[0]
        self.error = float(np.max(np.hypot(a @ self.cu - u_, a @ self.cv - v)))

    def _design(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = (np.asarray(x, dtype=float) - self.center[0]) / self.scale
        y = (np.asarray(y, dtype=float) - self.center[1]) / self.scale
        return np.stack([x**i * y**j for i in range(4) for j in range(4 - i)], axis=-1)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self._design(x, y)
        return a @ self.cu, a @ self.cv


def _field_stop_maps(
    model: esis.optics.Instrument,
    linear: optika.systems.AbstractLinearSystem,
    wavelength: u.Quantity,
    extent: u.Quantity = 520 * u.arcsec,
    num: int = 41,
) -> tuple[_Polynomial2d, _Polynomial2d]:
    """
    Map the sensor to the field stop and back at one wavelength.

    A grid on the sky is traced through the channel by its chief rays to the
    field stop, in the stop's own frame, and mapped to the sensor by the
    linearized distortion, the mapping every other stage uses; cubic
    polynomials fit between the two carry the sensor to the field stop and
    the field stop to the sensor without the sky in between.

    Parameters
    ----------
    model
        One channel's instrument at the parameters of the frame.
    linear
        Its linearization, see :meth:`esis.optics.LinearMerit.linearize`.
    wavelength
        The wavelength of the line.
    extent
        The half-width of the grid on the sky.
    num
        The number of points along each axis of the grid.

    Returns
    -------
    The map from the sensor (pixels) to the field stop (mm), and back.
    """
    grid = np.linspace(-1, 1, num) * extent.to_value(u.arcsec)
    gx, gy = np.meshgrid(grid, grid)
    sky = na.Cartesian2dVectorArray(
        x=na.ScalarArray(gx.ravel() * u.arcsec, axes="sky"),
        y=na.ScalarArray(gy.ravel() * u.arcsec, axes="sky"),
    )
    system = model.system
    sx, sy = _alignment.sensor_coordinates(
        linear.distortion, sky, wavelength, ("sky", "sky")
    )
    rays = system.raytrace(
        wavelength=wavelength,
        field=sky,
        pupil=na.Cartesian2dVectorArray(0, 0),
        normalized_field=False,
        efficiency=False,
    )
    stop = system.field_stop
    position = rays.outputs.position[{system.axis_surface: system.index_field_stop}]
    position = stop.transformation.inverse(position)
    fx = np.asarray(na.value(na.as_named_array(position.x).to(u.mm)).ndarray).ravel()
    fy = np.asarray(na.value(na.as_named_array(position.y).to(u.mm)).ndarray).ravel()
    sx, sy = np.ravel(sx), np.ravel(sy)
    radius = 1.3 * float(np.max(np.hypot(*_aperture_vertices(model).T)))
    ok = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(fx) & np.isfinite(fy)
    ok &= np.hypot(fx, fy) < radius
    return (
        _Polynomial2d(sx[ok], sy[ok], fx[ok], fy[ok]),
        _Polynomial2d(fx[ok], fy[ok], sx[ok], sy[ok]),
    )


def _aperture_vertices(model: esis.optics.Instrument) -> np.ndarray:
    """Read the vertices of a channel's field stop in its own frame, in mm."""
    vertices = model.field_stop.aperture.vertices
    return np.stack(
        [
            np.asarray(na.value(na.as_named_array(c).to(u.mm)).ndarray, float).ravel()
            for c in (vertices.x, vertices.y)
        ],
        axis=1,
    )


def measure_field_stop(
    reference: esis.optics.DistortionParameters,
    pointing: astropy.table.QTable,
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    frames: None | tuple[int, ...] = None,
    threshold: float = 0.5,
    line: int = 2,
    trim: float = 0.12,
    directory: None | str | pathlib.Path = None,
    path: None | str | pathlib.Path = None,
    path_sides: None | str | pathlib.Path = None,
    seed: int = 0,
) -> tuple[astropy.table.QTable, astropy.table.QTable]:  # pragma: nocover
    """
    Measure the shape of the field stop from its image in every channel.

    Each channel images the same field stop, so each measures its shape
    through its own grating and sensor.  In each channel, at the frame the
    pointing table is referenced to:

    1.  The current outline of the stop is mapped to the sensor at one
        line, and every side of the window, the diagonals too, is measured
        in every frame by :func:`esis.optics.measure_polygon_edges`, moved
        back by the frame's drift of the windows, and the median taken over
        the flight at each point along the side.
    2.  The points are carried to the plane of the field stop, in its own
        frame, by :func:`_field_stop_maps`, and a straight line fit to each
        side gives its offset along the outward normal of the current side
        and its tilt against it.
    3.  What each channel's own placement can do to its image of the stop
        is taken out: a rotation, fit to the tilts of its sides, and a
        translation and a scale, fit to their offsets.  These are fit
        together with one shape for every channel, the tilt and offset of
        each side that no rotation, translation or scale can make, so that
        a side a channel did not measure does not bias what is taken out.

    The vertices of the new stop are where the sides of that shape meet,
    grown by the mean of the channels' scales.  The
    rotation is left to the roll of the field stop, which the outline stage
    fits.  The spread of the channels about the shape, kept in the metadata,
    is what says whether they agree.

    Parameters
    ----------
    reference
        The reference parameters of every channel.
    pointing
        The pointing table, whose ``drift_x`` and ``drift_y`` move the
        windows of each frame back to the reference frame.
    instrument
        The instrument model, see :func:`fit_distortion_reference`.  Its
        field stop is the one measured against.
    num_scene
        The number of samples along each axis of the resampled scene, which
        sets the grid the channel is linearized on.
    frames
        The frames to measure.  If :obj:`None`, every frame whose mean
        signal exceeds `threshold` times the median over the flight.
    threshold
        The fraction of the flight's median signal below which a frame is
        left out.
    line
        The index of the line whose windows are measured: O V, the
        brightest, whose window lies whole on the sensor in every channel.
    trim
        The fraction of each side left out at either end, see
        :func:`esis.optics.measure_polygon_edges`.
    directory
        The directory the log is written to.
    path
        Where to save the vertices, if anywhere.
    path_sides
        Where to save every channel's offsets and tilts, if anywhere.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    The vertices of the measured stop, in mm in its own frame, and the
    offset and tilt of every side as each channel measured it, before and
    after its rotation, translation and scale are taken out.

    Raises
    ------
    ValueError
        If no channel measured some side of the stop.
    """
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    log = _logger(directory, "field_stop")
    level_1 = esis.flights.f1.data.level_1()
    axes = ("time", "channel", "detector_y", "detector_x")
    data = np.asarray(na.value(level_1.outputs).ndarray_aligned(axes), dtype=float)
    signal = data.mean(axis=(2, 3))
    if frames is None:
        kept = np.all(signal > threshold * np.median(signal, axis=0), axis=1)
        frames = tuple(int(t) for t in np.where(kept)[0])
    time = int(pointing.meta.get("frame_reference", 15))
    log(f"frames kept: {', '.join(str(t) for t in frames)}; reference frame {time}")
    scene, observation = _frame(time, num_scene)
    frame_column = np.asarray(pointing["frame"])
    rows = []
    vertices = None
    for c in range(num_channel):
        channel = instrument[dict(channel=c)]
        parameters = frame_parameters(reference, pointing, time, c)
        model = parameters.to_instrument(channel)
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=parameters,
            scene=scene,
            observation=observation[dict(channel=c)],
            seed=seed,
        )
        linear = merit.linearize(model)
        wavelength = channel.wavelength[dict(wavelength=line)]
        to_stop, to_sensor = _field_stop_maps(model, linear, wavelength)
        vertices = _aperture_vertices(model)
        num_vertex = len(vertices)
        polygon = np.stack(to_sensor(vertices[:, 0], vertices[:, 1]), axis=1)
        log(
            f"channel {c}: field stop to sensor to {to_sensor.error:.3f} px, "
            f"sensor to field stop to {to_stop.error * 1e3:.3f} um"
        )
        tables = []
        for t in frames:
            edges = esis.optics.measure_polygon_edges(data[t, c], polygon, trim=trim)
            row = pointing[frame_column == t]
            if len(row):
                edges["x"] -= row["drift_x"][0][c].to(u.pix)
                edges["y"] -= row["drift_y"][0][c].to(u.pix)
            edges["frame"] = t
            tables.append(edges)
        every = astropy.table.vstack(tables)
        for k in range(num_vertex):
            e = every[np.asarray(every["side"]) == k]
            if len(e) < 30:
                continue
            along = np.asarray(e["along"].to_value(u.pix))
            values = np.unique(along)
            points = np.array(
                [
                    [
                        np.median(e["x"].to_value(u.pix)[along == a]),
                        np.median(e["y"].to_value(u.pix)[along == a]),
                    ]
                    for a in values
                ]
            )
            fx, fy = to_stop(points[:, 0], points[:, 1])
            start, end = vertices[k], vertices[(k + 1) % num_vertex]
            tangent = (end - start) / np.hypot(*(end - start))
            normal = np.array([tangent[1], -tangent[0]])
            if np.dot((start + end) / 2, normal) < 0:
                normal = -normal
            tangent = np.array([-normal[1], normal[0]])
            relative = np.stack([fx, fy], axis=1) - (start + end) / 2
            slope, offset = np.polyfit(relative @ tangent, relative @ normal, 1)
            azimuth = np.degrees(np.arctan2(normal[1], normal[0])) % 360
            num_frame = len(np.unique(np.asarray(e["frame"])))
            rows.append(
                (c, k, azimuth, offset, np.arctan(slope), num_frame, len(values))
            )
        log(f"channel {c}: {sum(r[0] == c for r in rows)} sides measured")
    sides = astropy.table.QTable(
        rows=rows,
        names=("channel", "side", "azimuth", "offset", "tilt", "num_frame", "num"),
        dtype=(int, int, float, float, float, int, int),
    )
    sides["azimuth"].unit = u.deg
    sides["offset"].unit = u.mm
    sides["tilt"].unit = u.rad

    # take out what each channel's placement can do to its image of the stop
    edge = np.roll(vertices, -1, axis=0) - vertices
    apothem = float(
        np.mean(
            np.abs(edge[:, 0] * vertices[:, 1] - edge[:, 1] * vertices[:, 0])
            / np.hypot(edge[:, 0], edge[:, 1])
        )
    )
    # one shape for every channel, fit with every channel's rotation (to the
    # tilts) and translation and scale (to the offsets) at once, so that a side
    # a channel did not measure does not bias what is taken out of it; the
    # shape carries no rotation, translation or scale of its own
    channel = np.asarray(sides["channel"])
    k = np.asarray(sides["side"])
    missing = sorted(set(range(num_vertex)) - set(int(i) for i in k))
    if missing:
        raise ValueError(
            f"no channel measured side(s) {missing} of the field stop, so its "
            "shape cannot be fit"
        )
    # the direction of each side's outward normal, from the stop's vertices,
    # which are the same in every channel
    middle = (vertices + np.roll(vertices, -1, axis=0)) / 2
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normal *= np.sign(np.sum(middle * normal, axis=1))[:, None]
    azimuth = np.arctan2(normal[:, 1], normal[:, 0])
    weight = 1e3
    a = np.zeros((len(sides) + 1, num_channel + num_vertex))
    a[np.arange(len(sides)), channel] = 1
    a[np.arange(len(sides)), num_channel + k] = 1
    a[-1, num_channel:] = weight
    b = np.append(sides["tilt"].to_value(u.rad), 0)
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    rotation = [float(np.degrees(v)) for v in solution[:num_channel]]
    tilt = solution[num_channel:]
    spread_tilt = np.sqrt(np.mean((b - a @ solution)[:-1] ** 2))
    gauge = np.stack([np.cos(azimuth), np.sin(azimuth), np.ones(num_vertex)])
    a = np.zeros((len(sides) + 3, 3 * num_channel + num_vertex))
    rows = np.arange(len(sides))
    a[rows, 3 * channel] = np.cos(azimuth[k])
    a[rows, 3 * channel + 1] = np.sin(azimuth[k])
    a[rows, 3 * channel + 2] = apothem
    a[rows, 3 * num_channel + k] = 1
    a[len(sides) :, 3 * num_channel :] = weight * gauge
    b = np.append(sides["offset"].to_value(u.mm), np.zeros(3))
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    shift = [
        [float(v * 1e3) for v in solution[3 * c : 3 * c + 2]]
        for c in range(num_channel)
    ]
    scale = [float(solution[3 * c + 2]) for c in range(num_channel)]
    offset = solution[3 * num_channel :]
    spread_offset = np.sqrt(np.mean((b - a @ solution)[: len(sides)] ** 2))
    common_scale = float(np.mean(scale))
    log(
        "rotation of each channel's image of the stop [deg]: "
        + ", ".join(f"{r:+.3f}" for r in rotation)
    )
    log("scale [ppm]: " + ", ".join(f"{s * 1e6:+.0f}" for s in scale))
    log(
        f"shape: tilt {np.degrees(np.sqrt(np.mean(tilt**2))):.3f} deg rms, "
        f"offset {np.sqrt(np.mean(offset**2)) * 1e3:.2f} um rms; "
        f"channels about it: tilt {np.degrees(spread_tilt):.3f} deg rms, "
        f"offset {spread_offset * 1e3:.2f} um rms"
    )

    # the sides of the new stop, and where they meet
    lines = []
    for k in range(num_vertex):
        start, end = vertices[k], vertices[(k + 1) % num_vertex]
        tangent = (end - start) / np.hypot(*(end - start))
        normal = np.array([tangent[1], -tangent[0]])
        if np.dot((start + end) / 2, normal) < 0:
            normal = -normal
        tangent = np.array([-normal[1], normal[0]])
        middle = (start + end) / 2 + (offset[k] + common_scale * apothem) * normal
        direction = tangent + np.tan(tilt[k]) * normal
        lines.append((middle, direction / np.hypot(*direction)))
    new = []
    for k in range(num_vertex):
        (m0, d0), (m1, d1) = lines[(k - 1) % num_vertex], lines[k]
        s = np.linalg.solve(np.stack([d0, -d1], axis=1), m1 - m0)
        new.append(m0 + s[0] * d0)
    new = np.array(new)
    result = astropy.table.QTable(
        dict(x=new[:, 0] * u.mm, y=new[:, 1] * u.mm),
    )
    meta = dict(
        description=(
            "The vertices of the field stop of ESIS-I in its own frame, measured "
            "from its image in every channel, without its rotation."
        ),
        frames=list(frames),
        line=int(line),
        reference_frame=time,
        rotation=rotation,
        scale=scale,
        shift_um=shift,
        tilt_deg=[float(v) for v in np.degrees(tilt)],
        offset_um=[float(v) for v in offset * 1e3],
        spread_tilt_deg=float(np.degrees(spread_tilt)),
        spread_offset_um=float(spread_offset * 1e3),
        **_stamp(seed),
    )
    result.meta.update(meta)
    sides.meta.update(meta)
    if path is not None:
        result.write(path, format="ascii.ecsv", overwrite=True)
    if path_sides is not None:
        sides.write(path_sides, format="ascii.ecsv", overwrite=True)
    return result, sides


def fit_distortion_channel(
    channel: int,
    instrument: None | esis.optics.Instrument = None,
    time: int = 15,
    num_scene: int = 401,
    device: None | str = None,
    workers: int = 1,
    popsize: int = 15,
    maxiter: int = 80,
    seed: int = 0,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> esis.optics.DistortionParameters:  # pragma: nocover
    """
    Fit one channel of ESIS-I on its own, the first stage of the reference fit.

    The channel is fit to the AIA proxy scene of one frame with
    :class:`esis.optics.LinearMerit` and :func:`esis.optics.fit_distortion`,
    a seeded capture followed by a local polish, over the nine terms one
    frame determines (:data:`_FREE_ABSOLUTE`).  Each channel is hours of
    work, so the channels are fit separately and
    :func:`fit_distortion_reference` combines them.

    Requires the ESIS Level-1 data and the AIA synthetic scene (downloaded
    and cached on first use).

    Parameters
    ----------
    channel
        The index of the channel.
    instrument
        The starting instrument model. If :obj:`None`, the as-built model
        (:func:`esis.flights.f1.optics.as_built`) is used, idealized for
        raytracing speed, with the field stop of :data:`path_field_stop`.
    time
        The frame index of the reference frame.
    num_scene
        The number of samples along each axis of the resampled AIA scene.
    device
        The device on which to build and apply the regridding weights, for
        example ``"cuda"``; :obj:`None` for the host.
    workers
        The number of processes evaluating the population of the capture.
    popsize
        The population of the capture, per dimension.
    maxiter
        The number of generations of the capture, all of which are run.
    seed
        The seed of the capture, and of the samples the channel is
        linearized on, see :class:`esis.optics.LinearMerit`.
    path
        If given, the fitted parameters are written here as a one-row ECSV
        table.
    directory
        A directory where the progress is logged.
    """
    instrument = _base(instrument)
    scene, observation = _frame(time, num_scene)
    log = _logger(directory, f"channel_{channel}")
    model = instrument[dict(channel=channel)]
    p0 = esis.optics.DistortionParameters.from_instrument(model)
    merit = esis.optics.LinearMerit(
        instrument=model,
        parameters=p0,
        scene=scene,
        observation=observation[dict(channel=channel)],
        device=device,
        seed=seed,
    )
    log(f"channel {channel}: absolute fit of " + ", ".join(_FREE_ABSOLUTE))
    result = esis.optics.fit_distortion(
        objective=merit,
        parameters=p0,
        bounds=esis.flights.f1.optics.distortion_fit_bounds(p0),
        workers=workers,
        free=_FREE_ABSOLUTE,
        popsize=popsize,
        maxiter=maxiter,
        tol=0.0,
        seed=seed,
        log=log,
    )
    log(f"channel {channel}: correlation {merit.correlation(result):.4f}")
    if path is not None:
        result.to_file(
            path,
            metadata=dict(
                channel=channel,
                stage="absolute",
                time=time,
                free=list(_FREE_ABSOLUTE),
                popsize=popsize,
                maxiter=maxiter,
                **_stamp(seed),
            ),
        )
    return result


def fit_distortion_reference(
    parameters: list[esis.optics.DistortionParameters],
    edges: str | pathlib.Path | astropy.table.QTable,
    instrument: None | esis.optics.Instrument = None,
    time: int = 15,
    num_scene: int = 401,
    device: None | str = None,
    workers: int = 1,
    seed: int = 0,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
) -> esis.optics.DistortionParameters:  # pragma: nocover
    """
    Fit the reference distortion parameters of ESIS-I from the channels' own fits.

    Reproduces the values stored in ``_data/distortion_reference.ecsv`` from
    the absolute fit of every channel (:func:`fit_distortion_channel`), in
    three stages.

    1.  **Outline.**  Each channel's windows are placed on the window edges
        measured through the flight with the grating angles and the roll of
        the field stop (:data:`_WINDOW`), and the sky is refit with the
        remaining terms (:data:`_SKY`), see
        :func:`esis.optics.fit_distortion_outline`.
    2.  **Shared.**  The primary, the field-stop roll and the payload
        pointing belong to the instrument (:data:`_SHARED`); they are set to
        their mean over the channels and each channel's own terms
        (:data:`_FREE_SHARED`) are polished again.
    3.  **Internal.**  The channels are aligned to one another on the sky
        with :func:`esis.optics.align_channels`, in those terms only.

    Parameters
    ----------
    parameters
        The absolute fit of every channel, see :func:`fit_distortion_channel`.
    edges
        The measured edges of the windows at the reference frame, see
        :func:`measure_window_edges`, or the path of their table.
    instrument
        The instrument model, see :func:`fit_distortion_channel`.
    time
        The frame index of the reference frame.
    num_scene
        The number of samples along each axis of the resampled AIA scene.
    device
        The device on which to build and apply the regridding weights.
    workers
        The number of processes the channels are fit in side by side.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.
    path
        If given, the fitted parameters are written to this path as an ECSV
        table, in the format of ``_data/distortion_reference.ecsv``.
    directory
        A directory where the progress of every stage is logged.

    Raises
    ------
    ValueError
        If `parameters` does not have one entry per channel.
    """
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    scene, observation = _frame(time, num_scene)
    log = _logger(directory, "reference")
    parameters = list(parameters)
    if len(parameters) != num_channel:
        raise ValueError(f"expected {num_channel} channels, got {len(parameters)}")

    def merit(c: int) -> esis.optics.LinearMerit:
        return esis.optics.LinearMerit(
            instrument=instrument[dict(channel=c)],
            parameters=parameters[c],
            scene=scene,
            observation=observation[dict(channel=c)],
            device=device,
            seed=seed,
        )

    scores = dict(
        absolute=[
            float(merit(c).correlation(parameters[c])) for c in range(num_channel)
        ]
    )

    # outline
    edges = _read_table(edges)
    log(
        "outline: "
        + ", ".join(_WINDOW)
        + " on the edges; "
        + ", ".join(_SKY)
        + " on the merit"
    )
    jobs = [
        (
            instrument[dict(channel=c)],
            parameters[c],
            scene,
            observation[dict(channel=c)],
            device,
            edges[np.asarray(edges["channel"]) == c],
            seed,
        )
        for c in range(num_channel)
    ]
    results = _map_channels(_outline_own, jobs, workers)
    residuals = []
    for c, (parameters[c], rho, residual, messages) in enumerate(results):
        for message in messages:
            log(f"channel {c}: {message}")
        log(f"channel {c}: {rho:.4f}, edge residual {residual:.2f} px")
        residuals.append(residual)
    scores["outline"] = dict(merit=[float(r[1]) for r in results], edges=residuals)

    # shared
    parameters = enforce_shared(parameters)
    names = [f.name for f in dataclasses.fields(parameters[0])]
    own = [n for n in _FREE_SHARED if n not in _SHARED]
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
            seed,
        )
        for c in range(num_channel)
    ]
    results = _map_channels(_polish_own, jobs, workers)
    for c, (parameters[c], rho, num) in enumerate(results):
        log(f"channel {c}: {rho:.4f} after {num} evaluations")
    scores["shared"] = [float(r[1]) for r in results]

    # internal
    log("internal alignment")
    parameters, medians = esis.optics.align_channels(
        instruments=[instrument[dict(channel=c)] for c in range(num_channel)],
        parameters=parameters,
        frames=_frames_by_axis(observation, num_channel),
        scene_position=scene.inputs.position,
        wavelengths=_wavelengths_alignment(),
        free=tuple(own),
        # each channel's box about its own values
        bounds=[esis.flights.f1.optics.distortion_fit_bounds(p) for p in parameters],
        seed=seed,
        log=log,
    )
    scores["alignment"] = [float(m) for m in medians]

    # where the windows land, for a regression test of the committed table
    windows = _window_centers(instrument, parameters, scene, observation, seed)

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
                    "polish of every channel against the AIA proxy scene, its "
                    "windows placed on the measured edges, the shared optics "
                    "set to their mean, and the channels aligned to one "
                    f"another on the sky at {', '.join(_LINES_ALIGNMENT)} "
                    "(median shift vs channel 1: "
                    f"{', '.join(f'{m:.2f}' for m in medians)} px)"
                ),
                stages=dict(
                    absolute=dict(free=list(_FREE_ABSOLUTE)),
                    outline=dict(window=list(_WINDOW), sky=list(_SKY), edges=residuals),
                    shared=dict(free=list(own), fixed=list(_SHARED)),
                    alignment=dict(
                        free=list(own),
                        lines=list(_LINES_ALIGNMENT),
                        anchor=1,
                    ),
                ),
                scores=scores,
                windows=windows,
                **_stamp(seed),
            ),
        )
    return result


def _drift_shift(
    drift: astropy.table.QTable,
    channel: int,
    frame: int,
) -> np.ndarray:
    """
    Read the motion of one channel's windows in one frame, in pixels.

    The x motion is the mean over the left and right sides, the y motion
    over the top and bottom; an axis none of whose sides was measured in
    that frame is NaN, not zero.

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
    result = np.full(2, np.nan)
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

    A per-frame shift applied as measured jitters by a few tenths of a pixel
    and is missing in frames too dark to measure; the motion is smooth, so a
    low-order polynomial in the frame index carries it instead, held at the
    nearest measured frame outside them.  Each axis is fit over the frames
    in which it was measured, since a channel can see only one side along
    an axis and lose it in a frame while keeping the other axis.

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
        If an axis of a channel was measured in too few frames for the
        polynomial.
    """
    channel_of = np.asarray(drift["channel"])
    frame_of = np.asarray(drift["frame"])
    coefficients = np.zeros((num_channel, 2, degree + 1))
    span = np.zeros((num_channel, 2, 2), dtype=int)
    for c in range(num_channel):
        frames = np.array(sorted(set(int(t) for t in frame_of[channel_of == c])))
        measured = np.array([_drift_shift(drift, c, t) for t in frames])
        measured = measured.reshape(len(frames), 2)
        for k in range(2):
            # only the frames this axis was measured in: an unmeasured frame
            # is not a zero
            where = np.isfinite(measured[:, k])
            if where.sum() <= degree:
                raise ValueError(
                    f"channel {c} has {where.sum()} frames measured in "
                    f"{'xy'[k]}, too few for a degree-{degree} polynomial"
                )
            coefficients[c, k] = np.polyfit(frames[where], measured[where, k], degree)
            span[c, k] = frames[where][0], frames[where][-1]

    def shift(channel: int, frame: int) -> np.ndarray:
        return np.array(
            [
                np.polyval(
                    coefficients[channel, k],
                    min(max(frame, span[channel, k, 0]), span[channel, k, 1]),
                )
                for k in range(2)
            ]
        )

    return shift


def _drift_consistent(
    drift: astropy.table.QTable,
    degree: int,
    frame_reference: int,
) -> None:
    """
    Check that a stage smooths the drift the way its table was referred.

    :func:`measure_window_edges` refers the drift to the reference frame
    through a polynomial of one degree, so that the smoothed drift is zero
    there; a stage that smooths it with another degree, or takes another
    frame as the reference, would see a drift that is not zero at its
    reference.

    Parameters
    ----------
    drift
        The drift table, whose metadata may name the degree and the frame it
        was referred with.
    degree
        The degree the stage smooths the drift with.
    frame_reference
        The frame the stage takes as the reference.

    Raises
    ------
    ValueError
        If the table names a different degree or reference frame.
    """
    for key, value in (("drift_degree", degree), ("frame_reference", frame_reference)):
        if key in drift.meta and int(drift.meta[key]) != int(value):
            raise ValueError(
                f"the drift was referred with {key}={drift.meta[key]}, "
                f"but this stage uses {key}={value}"
            )


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


def _image_jacobian(
    linear,
    wavelength: u.Quantity,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate the origin of the sky on the sensor at one line, with its Jacobian.

    Parameters
    ----------
    linear
        The channel's linearized system.
    wavelength
        The line.

    Returns
    -------
    The sensor position of the origin of the sky in pixels, and the ``2x2``
    Jacobian in detector pixels per arcsecond along each sky axis.
    """
    origin = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 10.0, 0.0]) * u.arcsec, axes="s"),
        y=na.ScalarArray(np.array([0.0, 0.0, 10.0]) * u.arcsec, axes="s"),
    )
    x, y = _alignment.sensor_coordinates(
        linear.distortion, origin, wavelength, ("s", "s")
    )
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    jacobian = np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]]) / 10.0
    return np.array([x[0], y[0]]), jacobian


def _image_jacobians(
    parameters: esis.optics.DistortionParameters,
    channel: esis.optics.Instrument,
    wavelengths: dict[str, u.Quantity],
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Linearize one channel at given parameters and locate the sky at every line.

    Parameters
    ----------
    parameters
        The channel's parameters.
    channel
        The channel's instrument model.
    wavelengths
        The lines, by name.
    seed
        The seed of the samples the channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    :func:`_image_jacobian` at every line.
    """
    model = parameters.to_instrument(channel)
    linear = model.system.linearize(
        wavelength=model.wavelength,
        degree=2,
        field_stop=True,
        seed_field=seed,
        seed_pupil=seed,
    )
    return [_image_jacobian(linear, w) for w in wavelengths.values()]


def _pixels_per_arcsec(jacobians: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Find a channel's detector pixels per arcsecond along each axis of the sky.

    The same units as :func:`esis.optics.median_shifts`, so that a pattern from
    the model and a measured shift can be compared.

    Parameters
    ----------
    jacobians
        :func:`_image_jacobian` at every line.

    Returns
    -------
    An array indexed ``[axis]``, the mean of the lines.
    """
    return np.mean([np.linalg.norm(j, axis=0) for _, j in jacobians], axis=0)


def _defocus_pattern(
    instrument: esis.optics.Instrument,
    reference: list[esis.optics.DistortionParameters],
    wavelengths: dict[str, u.Quantity],
    anchor: int = 1,
    step: u.Quantity = 0.1 * u.mm,
    relative: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    Measure how a defocus of the primary moves each channel's sky against the anchor's.

    Each channel views the defocused image through its own sector of the
    primary, so the image shifts on the sky by the sector's offset times the
    defocus while the window edges stay put.  This is that pattern, from the
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
    relative
        Whether to take each channel's shift relative to the anchor's.  If
        :obj:`False`, each channel's own shift per millimetre of the focus
        of its own sector of the primary, in the anchor's pixels.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    An array indexed ``[channel, axis]``: the shift on the sky in detector
    pixels of each channel relative to the anchor per millimetre of
    defocus; zero for the anchor.
    """

    def where(c, dz):
        p = copy.copy(reference[c])
        p.z_primary = p.z_primary + dz
        return _image_jacobians(p, instrument[dict(channel=c)], wavelengths, seed)

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
            scale = _pixels_per_arcsec(before)
    if relative:
        shifts = shifts - shifts[anchor]
    return shifts * scale / step.to_value(u.mm)


def _solve_sector_focus(
    design: np.ndarray,
    measured: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Solve for the focus of each sector of the primary from the channels' shifts.

    Parameters
    ----------
    design
        The shift of each channel against the anchor per unit of each sector's
        focus, indexed ``[component, sector]``.
    measured
        The measured shifts, NaN where a channel has no tiles to measure.

    Returns
    -------
    The focus of each sector, NaN where no measured shift constrains it, and
    the rms of the residual, NaN if there are too few shifts to solve.
    """
    num_sector = design.shape[1]
    ok = np.isfinite(measured)
    if ok.sum() < num_sector:
        return np.full(num_sector, np.nan), np.nan
    # a measured shift s means the anchor shows at r + s what the channel
    # shows at r (see shift_fft): the channel's image sits at -s
    z, *_ = np.linalg.lstsq(design[ok], -measured[ok], rcond=None)
    residual = float(np.sqrt(np.mean((measured[ok] + design[ok] @ z) ** 2)))
    # lstsq gives a sector with no shift its minimum norm, zero, which is not
    # a measurement
    z[~np.any(design[ok], axis=0)] = np.nan
    return z, residual


def fit_defocus_history(
    reference: str | pathlib.Path | esis.optics.DistortionParameters,
    num_scene: int = 401,
    instrument: None | esis.optics.Instrument = None,
    frames: None | tuple[int, ...] = None,
    frame_reference: int = 15,
    drift: None | str | pathlib.Path | astropy.table.QTable = None,
    drift_degree: int = 3,
    degree: int = 2,
    anchor: int = 1,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    seed: int = 0,
) -> astropy.table.QTable:  # pragma: nocover
    """
    Measure the focus of each sector of the primary through the flight.

    Each channel views the Sun through its own sector of the primary, so a
    defocus moves its image along its own dispersion while the windows stay
    put (:func:`_defocus_pattern`).  In every frame the channels' skies are
    measured against the anchor's (:func:`esis.optics.measure_channel_shifts`),
    with the windows placed by that frame's drift as the pointing stage
    places them, and the focus of every sector is solved from the shifts
    through the model's pattern: four unknowns, the anchor's included, from
    six measured components.  A polynomial in the frame index, zero at the
    reference frame, is fit through each sector's history.  The mean over
    the sectors is the defocus of the primary as a whole.

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
        The degree of the polynomial in the frame index fit through each
        sector's per-frame focus.
    anchor
        The channel the others are measured against.
    path
        If given, the history is written here as an ECSV table with the
        per-frame and the fitted defocus.
    directory
        A directory where the progress is logged.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.
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

    pattern = _defocus_pattern(
        instrument, parameters, wavelengths, anchor, relative=False, seed=seed
    )
    log(
        "sky shift of each channel per mm of its own sector's focus [px]: "
        + "; ".join(f"{p[0]:+.1f}, {p[1]:+.1f}" for p in pattern)
    )
    others = [c for c in range(num_channel) if c != anchor]
    # the shift of channel c against the anchor is the anchor's motion less
    # the channel's: one column per sector's focus
    design = np.zeros((2 * len(others), num_channel))
    for i, c in enumerate(others):
        design[2 * i : 2 * i + 2, c] = pattern[c]
        design[2 * i : 2 * i + 2, anchor] = -pattern[anchor]

    drift = _read_table(drift)
    shift_drift = None
    if drift is not None:
        _drift_consistent(drift, drift_degree, frame_reference)
        shift_drift = _drift_smooth(drift, num_channel, drift_degree)
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
                            seed=seed,
                        )
                        for k in range(num_channel)
                    ]
                    sensitivity = _drift_sensitivity(instrument, parameters, merits)
                # the reference places the windows where they were at its
                # own frame, from which the drift is measured
                offset = shift_drift(c, t) / sensitivity[c]
                p.yaw_grating = p.yaw_grating + offset[0] * u.arcmin
                p.pitch_grating = p.pitch_grating + offset[1] * u.arcmin
            model = p.to_instrument(instrument[dict(channel=c)])
            linears.append(
                model.system.linearize(
                    wavelength=model.wavelength,
                    degree=2,
                    field_stop=True,
                    seed_field=seed,
                    seed_pupil=seed,
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
        z, residual = _solve_sector_focus(design, measured)
        shifts = np.full((num_channel, 2), np.nan)
        for c in others:
            shifts[c] = medians[c]
        log(
            f"frame {t}: shifts vs channel {anchor} "
            + "; ".join(f"{medians[c][0]:+.2f}, {medians[c][1]:+.2f}" for c in others)
            + " px -> focus of each sector "
            + ", ".join(f"{1e3 * v:+.1f}" for v in z)
            + f" um, residual {residual:.3f} px"
        )
        rows.append((t, z, residual, shifts))

    t_all = np.array([r[0] for r in rows], dtype=float)
    z_all = np.array([r[1] for r in rows])  # frame, channel
    fitted = np.zeros_like(z_all)
    used = np.ones(z_all.shape, dtype=bool)
    scatter = np.zeros(num_channel)
    for c in range(num_channel):
        fitted[:, c], used[:, c], scatter[c] = _fit_through_reference(
            t_all, z_all[:, c], frame_reference, degree, floor=0.003
        )
    log(
        f"focus of each sector through the flight: degree-{degree} fits about "
        f"frame {frame_reference}; at the first and the last frame "
        + "; ".join(
            f"ch{c} {1e3 * fitted[0, c]:+.1f} -> {1e3 * fitted[-1, c]:+.1f}"
            for c in range(num_channel)
        )
        + " um; scatter "
        + ", ".join(f"{1e3 * s:.1f}" for s in scatter)
        + " um"
    )
    table = astropy.table.QTable(
        dict(
            frame=np.array([r[0] for r in rows]),
            z_primary_measured=z_all * u.mm,
            residual=np.array([r[2] for r in rows]) * u.pix,
            shift_x=np.array([r[3][:, 0] for r in rows]) * u.pix,
            shift_y=np.array([r[3][:, 1] for r in rows]) * u.pix,
            used=used,
            z_primary=fitted * u.mm,
        )
    )
    table.meta.update(
        dict(
            description=(
                "The focus of each sector of the primary at the field stop "
                "through the ESIS-I flight, one value per channel, relative to "
                "the reference frame, measured from the shifts of the "
                "channels' skies against one another (shift_x and shift_y, "
                "with the pointing stage's window drift applied and no focus) "
                "and smoothed by a polynomial in the frame index; z_primary is "
                "the smoothed value the pointing stage applies to each "
                "channel.  The mean over the channels is the defocus of the "
                "primary as a whole."
            ),
            provenance=(
                f"esis.flights.f1.optics.fit_defocus_history(num_scene={num_scene}, "
                f"degree={degree}, anchor={anchor}, drift_degree={drift_degree})"
            ),
            frame_reference=int(frame_reference),
            anchor=int(anchor),
            pattern=[[float(v) for v in p] for p in pattern],
            scatter=[float(s) for s in scatter],
            **_stamp(seed),
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
) -> esis.optics.DistortionParameters:
    """
    Take one channel's reference parameters to one frame of the flight.

    Everything the pointing table carries for the frame is applied: the
    pointing of the payload, the focus of the channel's sector of the
    primary, and the grating offsets that follow its windows.  The flight
    factory applies the same columns to the whole model.

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
    p.z_primary = p.z_primary + row["z_primary"][channel]
    p.yaw_grating = p.yaw_grating + row["yaw_grating"][channel]
    p.pitch_grating = p.pitch_grating + row["pitch_grating"][channel]
    return p


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
    `clip` robust standard deviations from the fit are left out and the fit
    repeated; the deviation is never taken below `floor`, since a robust
    estimate from a few dozen frames can come out far too small.

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


def fit_distortion_pointing(
    instrument: None | esis.optics.Instrument = None,
    num_scene: int = 401,
    device: None | str = None,
    frame_reference: int = 15,
    frames: None | tuple[int, ...] = None,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    drift: None | str | pathlib.Path | astropy.table.QTable = None,
    drift_degree: int = 3,
    defocus: None | str | pathlib.Path | astropy.table.QTable = None,
    seed: int = 0,
    relative: bool = True,
) -> astropy.table.QTable:  # pragma: nocover
    r"""
    Fit the per-frame payload pointing of the whole flight.

    Reproduces the time series stored in ``_data/distortion_pointing.ecsv``:
    for every frame of :func:`esis.flights.f1.data.level_1`, one pitch and
    yaw offset from the reference model, common to the four channels, is
    polished with :func:`esis.optics.polish` on the mean of their
    :class:`esis.optics.LinearMerit`.  The optics are otherwise held at the
    reference fit, with the measured window drift and sector focus applied.

    Parameters
    ----------
    instrument
        The reference model. If :obj:`None`,
        :func:`esis.flights.f1.optics.distortion_fit` is used, idealized.
    num_scene
        The number of samples along each axis of the resampled AIA scene.
    device
        The device on which to build and apply the regridding weights.
    frame_reference
        The frame the reference was fit to, from which the drift is
        measured.  With `relative`, the table is taken relative to it.
    frames
        The frames to fit.  If :obj:`None`, every frame is fit.
    path
        If given, the offsets are written to this path as an ECSV table, in
        the format of ``_data/distortion_pointing.ecsv``.
    directory
        A directory where the progress of every frame is logged, in
        ``pointing.log``, or in ``pointing_<frame>.log`` when one frame is
        fit.
    drift
        The per-frame drift of the windows, see
        :func:`measure_window_edges`, or the path of its table.  If given,
        each frame's grating yaw and pitch are offset per channel so that
        its windows sit where the edges of that frame were measured, before
        the pointing is fit; the offsets are recorded in the result.
    drift_degree
        The degree of the polynomial in the frame index that smooths the
        drift through the flight, see :func:`_drift_smooth`.
    defocus
        The focus of each sector of the primary through the flight, see
        :func:`fit_defocus_history`, or the path of its table.  If given,
        each frame's ``z_primary`` is applied before the pointing is fit and
        is not itself fit: the proxy scene cannot tell a defocus from the
        internal alignment, but the channels can.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.
    relative
        If the reference frame is among `frames`, subtract its own offset
        from every row, so that the table is relative to the reference frame
        and zero there; see :func:`pointing_relative`.  A driver that fits
        one frame per job passes :obj:`False`, so that every row, the
        reference frame's included, is the absolute fit, then gathers the
        rows and takes them relative once.
    """
    if instrument is None:
        instrument = _idealized(esis.flights.f1.optics.distortion_fit())
        instrument.wavelength = _wavelength_lines()
    num_channel = instrument.camera.channel.shape["channel"]
    num_time = esis.flights.f1.data.level_1().shape["time"]
    if frames is None:
        frames = tuple(range(num_time))
    # a driver that fits one frame per job, side by side, gets one log per
    # frame, since each log starts its file afresh
    name = "pointing" if len(frames) > 1 else f"pointing_{frames[0]:02d}"
    log = _logger(directory, name)

    reference = [
        esis.optics.DistortionParameters.from_instrument(instrument[dict(channel=c)])
        for c in range(num_channel)
    ]
    names = [f.name for f in dataclasses.fields(reference[0])]
    # the merit against the proxy scene cannot measure a roll to better
    # than an arcminute, so the roll is held at the reference's
    common = ("pitch", "yaw")
    index = [names.index(n) for n in common]
    index_defocus = names.index("z_primary")
    index_grating = [names.index(n) for n in ("yaw_grating", "pitch_grating")]
    defocus = _read_table(defocus)

    drift = _read_table(drift)
    shift_drift = None
    if drift is not None:
        _drift_consistent(drift, drift_degree, frame_reference)
        shift_drift = _drift_smooth(drift, num_channel, drift_degree)
        measured = sorted(set(int(t) for t in np.asarray(drift["frame"])))
        log(
            f"window drift smoothed by a degree-{drift_degree} polynomial over "
            f"frames {measured[0]} to {measured[-1]}, held outside them"
        )
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
                seed=seed,
            )
            for c in range(num_channel)
        ]
        x_full = [na.pack(p).ndarray for p in reference]
        offsets = np.zeros((num_channel, 2))
        shifts = np.zeros((num_channel, 2))
        z_primary = np.zeros(num_channel)
        if defocus is not None:
            where = np.flatnonzero(np.asarray(defocus["frame"]) == t)
            if where.size:
                value = np.asarray(defocus["z_primary"][where[0]].to_value(u.mm))
                z_primary = np.broadcast_to(value, (num_channel,)).copy()
                for c in range(num_channel):
                    x_full[c][index_defocus] += z_primary[c]
                log(
                    f"frame {t}: focus of each sector "
                    + ", ".join(f"{1e3 * z:+.1f}" for z in z_primary)
                    + " um applied"
                )
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
                + f" px from frame {frame_reference}"
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
        offset = np.zeros(3)  # pitch, yaw, roll; the roll stays zero if held
        offset[: len(y)] = y
        rows.append((t, offset, shifts, offsets, z_primary))

    table = astropy.table.QTable(
        dict(
            frame=np.array([r[0] for r in rows]),
            pitch=np.array([r[1][0] for r in rows]) * u.arcsec,
            yaw=np.array([r[1][1] for r in rows]) * u.arcsec,
            roll=np.array([r[1][2] for r in rows]) * u.deg,
            z_primary=np.array([r[4] for r in rows]) * u.mm,  # frame, channel
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
                "the focus of each channel's sector of the primary at the "
                "field stop, measured from the channels against one another "
                "and applied here, which moves each channel's sky along its "
                "dispersion without moving the windows.  drift_x and "
                "drift_y are the measured motion of each channel's windows "
                "from the reference frame, and yaw_grating and pitch_grating "
                "the per-channel grating offsets that reproduce it."
            ),
            provenance=(
                "esis.flights.f1.optics.fit_distortion_pointing("
                f"num_scene={num_scene}, drift_degree={drift_degree}): a local "
                "polish of the mean linear merit of the four channels in "
                "pitch and yaw, the roll held, per frame, with the measured "
                "defocus applied"
            ),
            **_stamp(seed),
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


def _dispersion(linear, wavelength: u.Quantity) -> tuple[np.ndarray, float]:
    """
    Find where a channel disperses on the sky at one line, and how fast.

    Parameters
    ----------
    linear
        The channel's linearized system.
    wavelength
        The line.

    Returns
    -------
    The unit vector on the sky along which a redshift displaces the image,
    in the sky grid's axes, and the velocity a displacement of one detector
    pixel along it stands for, in km/s.
    """
    step = 0.1 * u.AA
    position, jacobian = _image_jacobian(linear, wavelength)
    position_step, _ = _image_jacobian(linear, wavelength + step)
    shift = position_step - position
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
    num_sky: int = 1600,
    num_tile: int = 8,
    fraction_valid: float = 0.8,
    anchor: int = 1,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    seed: int = 0,
) -> astropy.table.QTable:  # pragma: nocover
    """
    Score a reference fit on frames it was not fit to, without an inversion.

    Three measurements per frame and channel: the merit against the AIA
    proxy scene with that frame's pointing applied; the coalignment, which
    is the length of the median tile shift of the channel's sky against the
    anchor's at each aligned line (``shift_<line>``, how far the channel is
    misregistered as a whole) and the rms scatter of the tiles about that
    median (``scatter_<line>``, the distortion still differing between the
    two), both in detector pixels on the anchor's scale; and the residual of
    the reference's window outlines against the measured edges.

    Parameters
    ----------
    reference
        The reference parameters of every channel.
    pointing
        The per-frame pointing, see :func:`fit_distortion_pointing`.
    edges
        The measured window edges, see :func:`measure_window_edges`.
    frames
        The frames to score.
    frames_shifts
        The frames whose coalignment alone is measured, which needs no
        proxy scene and so is cheap; :obj:`None` measures every frame of
        the flight.  The rows of frames not in `frames` carry no merit.
    instrument
        The instrument model, see :func:`fit_distortion_reference`.
    num_scene
        The number of samples along each axis of the resampled scene.
    device
        The device the merit runs on.
    num_sky
        The number of samples along each axis of the common sky grid.
    num_tile
        The number of tiles along each axis when measuring shifts.
    fraction_valid
        The fraction of a tile which must lie inside both windows for its
        shift to count.  Looser than the alignment's, since the octagon's
        diagonal sides and the gap between windows cut into many tiles.
    anchor
        The channel the others are compared with.
    path
        Where to save the table, if anywhere.
    directory
        The directory the log is written to.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    A table with a row per frame and channel: the correlation, and the
    median shift against the anchor at each aligned line in pixels; the
    outline residual per channel is in the metadata.
    """
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
                    seed=seed,
                )
                merits.append((m, p))
                linears.append(m.linearize(model))
            else:
                linears.append(
                    model.system.linearize(
                        wavelength=model.wavelength,
                        degree=2,
                        field_stop=True,
                        seed_field=seed,
                        seed_pupil=seed,
                    )
                )
        correlations = (
            [m.correlation(p) for m, p in merits] if scored else [np.nan] * num_channel
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
            scale = np.mean(scales, axis=0)
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
                                field.dx[k] * scale[0],
                                field.dy[k] * scale[1],
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
                dx = np.concatenate([f.dx for f in fields[c]]) * scale[0]
                dy = np.concatenate([f.dy for f in fields[c]]) * scale[1]
                # the length of the median shift, not the median length: the
                # latter never falls below the noise of a single tile
                shifts[c][name] = float(np.hypot(np.median(dx), np.median(dy)))
                # the part along the channel's own dispersion reads as velocity
                along[c][name] = float(
                    abs(np.array([np.median(dx), np.median(dy)]) @ dispersion[c][0])
                )
                scatter[c][name] = float(
                    np.sqrt(
                        np.mean((dx - np.median(dx)) ** 2 + (dy - np.median(dy)) ** 2)
                    )
                )
        for c in range(num_channel):
            rows.append(
                (
                    t,
                    c,
                    correlations[c],
                    *[shifts[c][name] for name in lines],
                    *[scatter[c][name] for name in lines],
                    *[along[c][name] for name in lines],
                )
            )
            log(
                f"frame {t} channel {c}: correlation {correlations[c]:.4f}, shift "
                + ", ".join(f"{name} {shifts[c][name]:.2f} px" for name in lines)
                + ", scatter "
                + ", ".join(f"{name} {scatter[c][name]:.2f} px" for name in lines)
            )
    table = astropy.table.QTable(
        rows=rows,
        names=("frame", "channel", "correlation")
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
                **_stamp(seed),
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
                **_stamp(seed),
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
                seed=seed,
            )
            linear = m.linearize(p.to_instrument(channel))
            footprints = _footprints(linear, channel.wavelength)
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
                "The reference fit scored on frames of the flight: the "
                "correlation with each frame's pointing applied, the median "
                "tile shift of "
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
            **_stamp(seed),
        )
    )
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def pointing_relative(table: astropy.table.QTable, frame: int) -> astropy.table.QTable:
    """
    Make a table of pointing offsets relative to one of its frames.

    A per-frame fit of the pointing alone lands a fraction of an arcsecond
    from the reference fit of the same frame, since it optimizes the mean
    merit of the channels in fewer terms; subtracting the reference frame's
    own fit puts that frame at zero and the others relative to it.  The
    drift of the windows and the focus of the sectors are zero there
    already, since both are measured from the reference frame.

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

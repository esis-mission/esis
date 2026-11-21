"""
Iterate the measured field stop to convergence and record the fit it gives.

The field stop is measured through a distortion fit, and the fit is made with
the field stop, so the two are iterated: the first fit is made with the
regular octagon of the design, and each later one with the stop the previous
fit measured, until a measurement no longer moves the stop.  These functions
decide when that is, measure the clocking of the stop on the sky through the
last fit, and write the tables the package commits from the passes.
"""

from __future__ import annotations
import pathlib
import shutil
import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na
import esis
from . import _fits

__all__ = [
    "PARKER2022",
    "TABLES",
    "field_stop_change",
    "field_stop_converged",
    "registration_rms",
    "measure_clocking",
    "record_distortion_fit",
]

TABLES = (
    "distortion_reference.ecsv",
    "distortion_pointing.ecsv",
    "defocus.ecsv",
    "window_edges.ecsv",
    "window_drift.ecsv",
    "acceptance.ecsv",
    "acceptance_modes.ecsv",
    "acceptance_tiles.ecsv",
)
"""The tables of a pass that the package commits, besides the field stop."""

TOLERANCE = 0.05 * u.deg
"""
How little a measurement must turn the sides of the stop it was made
through to end the iteration, as the rms of their tilts.

The tilts are what the regular octagon of the design gets wrong: the first
pass, fit with it, measures sides tilted by 0.49 deg rms.  Once the stop has
settled, refitting through the stop a pass measured tilts its sides again by
up to 0.04 deg rms, the fit's own scatter (passes 1 to 3 of the validation
run, which reproduce the pointing of the committed fit to within the scatter
between seeds), and the channels' measurements of one stop scatter by
0.025 deg about their common shape.  The tolerance is a little above that
scatter, so the iteration ends at the first pass after the stop has settled.

The offsets of the sides and the scales of the channels are recorded but do
not end the iteration: they scatter from pass to pass by as much as the
regular octagon is off (1.1 um against 1.2 um), so they cannot tell a
settled stop from an unsettled one.
"""

PARKER2022 = dict(
    description=(
        "The orientation of the field stop as the pointing figure of Parker et "
        "al. (2022) gives it, for comparison.  pixels are the eight corners "
        "picked by hand on its Level-3 O V image, (x, y) in Level-3 pixels; "
        "that image is on the pixel grid of AIA 304 offset by 300 pixels in "
        "each axis, and its coordinates leave out AIA's own roll, crota2_deg "
        "(the CROTA2 of the AIA 304 images of the flight).  corners_deg is "
        "each corner's departure from its nominal multiple of 45 degrees "
        "about their centroid, by nominal azimuth 0, 45, ..., 315, on the sky "
        "with that roll, and corners_grid_deg the same on the AIA pixel grid "
        "without it, computed with the full WCS of the AIA 304 image at "
        "2019-09-30T18:08:43Z.  level3_rotation_deg is the rotation of each "
        "camera's image against the AIA pixel grid in that paper's Level-3 "
        "co-alignment, the median over its sequences of the affine rot_angle "
        "in ov_Level3_transform.pickle less its nominal -22.5 - 45 k degrees; "
        "it includes the camera's roll on its detector."
    ),
    pixels=[
        [629, 11],
        [1077, 185],
        [1267, 630],
        [1081, 1070],
        [644, 1255],
        [197, 1075],
        [17, 633],
        [190, 200],
    ],
    crota2_deg=-0.1315,
    corners_deg=[-0.706, -0.253, -0.190, -1.133, -0.938, -0.657, -0.348, -0.498],
    corners_grid_deg=[-0.575, -0.122, -0.058, -1.001, -0.807, -0.525, -0.217, -0.366],
    level3_rotation_deg=[0.869, 0.354, 0.371, 1.046],
)
"""What the earlier paper gives for the orientation of the stop and the cameras."""


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def field_stop_change(table: astropy.table.Table) -> dict:
    """
    Summarize how far a measurement of the field stop moved it.

    Parameters
    ----------
    table
        A measurement, as :func:`measure_field_stop` writes it, whose tilts
        and offsets are those of its sides against the stop it was measured
        through.

    Returns
    -------
    The rms tilt and offset of the sides, the rms of the channels about the
    common shape, and each channel's rotation and scale.
    """
    meta = table.meta
    return dict(
        shape_tilt_deg=_rms(meta["tilt_deg"]),
        shape_offset_um=_rms(meta["offset_um"]),
        spread_tilt_deg=float(meta["spread_tilt_deg"]),
        spread_offset_um=float(meta["spread_offset_um"]),
        rotation_deg=[float(v) for v in meta["rotation"]],
        scale=[float(v) for v in meta["scale"]],
    )


def field_stop_converged(
    table: astropy.table.Table,
    tolerance: u.Quantity = TOLERANCE,
) -> bool:
    """
    Decide whether a measurement of the field stop left it where it was.

    Parameters
    ----------
    table
        A measurement, as :func:`measure_field_stop` writes it.
    tolerance
        The largest rms tilt of the sides that counts as not moving it.
    """
    change = field_stop_change(table)
    return bool(change["shape_tilt_deg"] < tolerance.to_value(u.deg))


def registration_rms(
    acceptance: astropy.table.Table,
    drift: astropy.table.Table,
) -> dict[str, float]:
    """
    Measure the registration of the channels a fit leaves, per line.

    The rms over the channels other than the anchor and over the frames whose
    window edges were measured of the median tile shift against the anchor,
    in pixels, as the paper quotes it.

    Parameters
    ----------
    acceptance
        The acceptance table, see :func:`acceptance`.
    drift
        The drift table, see :func:`measure_window_edges`, whose frames are
        the measured ones.
    """
    anchor = int(acceptance.meta["anchor"])
    measured = np.unique(np.asarray(drift["frame"]))
    rows = acceptance[
        np.isin(np.asarray(acceptance["frame"]), measured)
        & (np.asarray(acceptance["channel"]) != anchor)
    ]
    result = {}
    for line, column in (("He I", "shift_He_I"), ("O V", "shift_O_V")):
        values = u.Quantity(rows[column]).to_value(u.pix)
        result[line] = float(np.sqrt(np.nanmean(np.square(values))))
    return result


def measure_clocking(
    reference: esis.optics.DistortionParameters,
    pointing: astropy.table.QTable,
    instrument: None | esis.optics.Instrument = None,
    frame: int = 15,
    line: int = 2,
    extent: u.Quantity = 520 * u.arcsec,
    num: int = 41,
) -> dict:  # pragma: nocover
    """
    Measure the orientation of the field stop on the sky.

    In each channel, a grid on the sky is traced by its chief rays to the
    field stop, in the stop's own frame, a cubic fit between the two carries
    the stop's vertices back to the sky, and their position angles are read
    from solar north, positive counterclockwise (north through east).  The
    sky is that of the AIA images, whose own roll the scene applies.

    Parameters
    ----------
    reference
        The reference parameters of every channel.
    pointing
        The pointing table.
    instrument
        The instrument model, see :func:`fit_distortion_reference`.  Its
        field stop is the one whose orientation is measured.
    frame
        The frame to measure at.
    line
        The index of the line to trace at.
    extent
        The half-width of the grid on the sky.
    num
        The number of points along each axis of the grid.

    Returns
    -------
    ``north_deg``, the position angle of the corner nearest north about the
    centre of the stop, per channel; ``corners_deg``, each corner's departure
    from its nominal multiple of 45 degrees about the centroid of the
    corners, by nominal azimuth 0, 45, ..., 315, per channel; and
    ``rotation_deg``, the mean of those over the corners and channels, the
    roll of the stop.
    """
    instrument = _fits._base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    grid = np.linspace(-1, 1, num) * extent.to_value(u.arcsec)
    gx, gy = np.meshgrid(grid, grid)
    sky = na.Cartesian2dVectorArray(
        x=na.ScalarArray(gx.ravel() * u.arcsec, axes="sky"),
        y=na.ScalarArray(gy.ravel() * u.arcsec, axes="sky"),
    )

    def angle(dx, dy):
        return np.degrees(np.arctan2(-dx, dy))

    north = []
    corners = []
    for c in range(num_channel):
        parameters = _fits.frame_parameters(reference, pointing, frame, c)
        model = parameters.to_instrument(instrument[dict(channel=c)])
        system = model.system
        wavelength = model.wavelength[dict(wavelength=line)]
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
        fx, fy = (
            np.asarray(na.value(na.as_named_array(p).to(u.mm)).ndarray, float).ravel()
            for p in (position.x, position.y)
        )
        vertices = _fits._aperture_vertices(model)
        radius = 1.3 * np.hypot(vertices[:, 0], vertices[:, 1]).max()
        kept = np.isfinite(fx) & np.isfinite(fy) & (np.hypot(fx, fy) < radius)
        to_sky = _fits._Polynomial2d(
            fx[kept], fy[kept], gx.ravel()[kept], gy.ravel()[kept]
        )
        sx, sy = to_sky(np.append(vertices[:, 0], 0), np.append(vertices[:, 1], 0))
        cx, cy = sx[-1], sy[-1]
        sx, sy = sx[:-1], sy[:-1]
        k = int(np.argmax(sy - cy))
        north.append(float(angle(sx[k] - cx, sy[k] - cy)))
        a = angle(sx - sx.mean(), sy - sy.mean())
        departure = a - np.round(a / 45) * 45
        order = np.argsort(np.round(a / 45) % 8)
        corners.append([float(departure[i]) for i in order])
    return dict(
        description=(
            "The orientation of the stop on the sky through the fit, from a "
            "chief-ray trace of a grid on the sky to the stop, inverted by a "
            "cubic.  Angles are position angles from solar north, positive "
            "counterclockwise (north through east); the sky frame is that of "
            "the AIA images with their own roll applied.  north_deg is the "
            "position angle of the corner nearest north about the centre of "
            "the stop, per channel; corners_deg is each corner's departure "
            "from its nominal multiple of 45 degrees about the centroid of "
            "the corners, by nominal azimuth 0, 45, ..., 315, per channel; "
            "rotation_deg is the mean of those, the roll of the stop."
        ),
        frame=int(frame),
        line=int(line),
        north_deg=north,
        rotation_deg=float(np.mean(corners)),
        corners_deg=corners,
    )


def record_distortion_fit(
    passes: list[str | pathlib.Path],
    directory: str | pathlib.Path,
    clocking: None | dict = None,
    tolerance: u.Quantity = TOLERANCE,
) -> astropy.table.QTable:
    """
    Write the tables the package commits from the passes of the iteration.

    The fit is the last pass's: its tables are copied, and its field stop is
    the one that pass was fit with, ``field_stop_input.ecsv`` in its
    directory, whose metadata gains the history of the iteration, the
    clocking on the sky through the fit, and what the earlier paper gives
    for comparison.

    Parameters
    ----------
    passes
        The directory of every pass, in order: the first fit with the
        regular octagon, each later one with the stop the one before it
        measured, and the last one's measurement the check that ended the
        iteration.
    directory
        The directory to write the tables into.
    clocking
        The clocking of the stop, see :func:`measure_clocking`; if
        :obj:`None`, measured through the last pass.
    tolerance
        The tolerance the iteration ended on, see :func:`field_stop_converged`.

    Returns
    -------
    The field stop, as written.

    Raises
    ------
    ValueError
        If there is only one pass, the fit with the regular octagon, or if
        the last pass's measurement moved the stop.
    """
    passes = [pathlib.Path(p) for p in passes]
    directory = pathlib.Path(directory)
    final = passes[-1]
    if len(passes) < 2:
        raise ValueError("the fit with the regular octagon cannot be the last pass")

    history = []
    for k, p in enumerate(passes):
        measured = astropy.table.QTable.read(p / "field_stop.ecsv", format="ascii.ecsv")
        reference = astropy.table.QTable.read(
            p / "distortion_reference.ecsv", format="ascii.ecsv"
        )
        entry = dict(
            directory=p.name,
            field_stop="regular" if k == 0 else passes[k - 1].name,
            outline_edges_px=[
                float(v) for v in reference.meta["stages"]["outline"]["edges"]
            ],
            **field_stop_change(measured),
        )
        entry["converged"] = field_stop_converged(measured, tolerance)
        history.append(entry)
    if not history[-1]["converged"]:
        raise ValueError(f"the measurement of {final} moved the field stop")

    directory.mkdir(parents=True, exist_ok=True)

    for name in TABLES:
        shutil.copyfile(final / name, directory / name)

    stop = astropy.table.QTable.read(
        final / "field_stop_input.ecsv", format="ascii.ecsv"
    )
    if clocking is None:
        instrument = _fits._base(None)
        instrument.field_stop.vertices = esis.flights.f1.optics.field_stop_vertices(
            final / "field_stop_input.ecsv"
        )
        clocking = measure_clocking(
            reference=esis.optics.DistortionParameters.from_file(
                final / "distortion_reference.ecsv"
            ),
            pointing=astropy.table.QTable.read(
                final / "distortion_pointing.ecsv", format="ascii.ecsv"
            ),
            instrument=instrument,
        )
    stop.meta["history"] = dict(
        description=(
            "The passes of the iteration that measured the stop, in order.  "
            "field_stop is what each pass was fit with: the regular octagon "
            "of the design, or the stop the named pass measured.  "
            "outline_edges_px is the rms distance of the measured window "
            "edges from the model's outline after that pass's outline stage, "
            "per channel.  shape_tilt_deg and shape_offset_um are the rms "
            "change of the sides of the stop the pass measured against the "
            "stop it was fit with, spread_tilt_deg and spread_offset_um the "
            "rms of the channels about the common shape, rotation_deg and "
            "scale each channel's image of the stop against the stop it was "
            "fit with.  The iteration ends at the first pass whose "
            "measurement is within the tolerance; that pass is the fit, and "
            "this table is the stop it was fit with."
        ),
        tolerance_tilt_deg=float(tolerance.to_value(u.deg)),
        passes=history,
    )
    stop.meta["clocking"] = clocking
    stop.meta["parker2022"] = dict(PARKER2022)
    stop.write(directory / "field_stop.ecsv", format="ascii.ecsv", overwrite=True)

    return stop

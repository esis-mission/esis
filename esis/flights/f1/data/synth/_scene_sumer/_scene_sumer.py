import csv
import pathlib
import functools
import numpy as np
import astropy.units as u
import astropy.time
import astropy.table
import soho
import esis
from ..._sumer import path_sumer
from ....spectrum import O_V

__all__ = [
    "passes_sumer",
    "scene_sumer",
]

_path_passes = pathlib.Path(__file__).parent / "passes_sumer.csv"
"""The table of the passes, which :func:`passes_sumer` reads."""

_tolerance_start = 60 * u.s
"""
How far from the start of a pass a time may be to name it,
so that it can be given to the minute.
"""


@functools.cache
def _rows() -> tuple[dict[str, str], ...]:
    """Return the rows of the table of the passes, as text."""
    with open(_path_passes, newline="") as f:
        return tuple(csv.DictReader(f))


def passes_sumer() -> astropy.table.QTable:
    """
    Return the passes of SUMER which the training set of ESIS was made from.

    The passes are rasters of O V 629.7 Angstroms of the Solar Ultraviolet
    Measurements of Emitted Radiation (SUMER) spectrograph,
    from which the training set of the neural-network inversion of ESIS was
    made, one row for each pass, in order of their start.

    A pass is one sweep of the slit of SUMER across its field.
    A sequence of SUMER which sweeps its field more than once is split into
    passes where the slit turns back or jumps,
    and only some of the passes of a sequence which repeats are used,
    since they are much alike.

    The columns are:

    ``time_start``, ``time_stop``
        The start of the first exposure of the pass,
        and the time before which its last exposure started.
    ``file``
        The level-1 file of SUMER which holds the pass,
        in :func:`~esis.flights.f1.data.path_sumer`.
    ``exposure``
        The exposure time of each exposure.
    ``slit``
        The slit, as the files of SUMER name it.
    ``study``, ``objective``
        The study which observed the pass, and its science objective,
        from ESA's catalogue of SOHO.
    ``kind``
        ``"qs"`` for quiet Sun, coronal holes, and the network,
        ``"ar"`` for active regions, sunspots, and loops.
    ``polar``
        Whether the pass is a scan of a polar coronal hole,
        whose slit has stripes, which :func:`scene_sumer` removes.
    ``split``
        ``"train"``, ``"validation"``, or ``"test"``:
        the part of the training set the frames of the pass went into.
        Whole sequences were held out for validation and testing,
        so a pass held out has no near twin among those trained on.
    ``num_exposure``
        The number of exposures of the pass.
    """
    rows = _rows()
    return astropy.table.QTable(
        {
            "time_start": astropy.time.Time([row["time_start"] for row in rows]),
            "time_stop": astropy.time.Time([row["time_stop"] for row in rows]),
            "file": [row["file"] for row in rows],
            "exposure": [float(row["exposure"]) for row in rows] * u.s,
            "slit": [row["slit"] for row in rows],
            "study": [row["study"] for row in rows],
            "objective": [row["objective"] for row in rows],
            "kind": [row["kind"] for row in rows],
            "polar": [row["polar"] == "true" for row in rows],
            "split": [row["split"] for row in rows],
            "num_exposure": [int(row["num_exposure"]) for row in rows],
        }
    )


def _pass(time: astropy.time.Time) -> astropy.table.Row:
    """
    Return the pass which starts at a time, or else which holds it.

    The start may be off by up to :data:`_tolerance_start`.

    Parameters
    ----------
    time
        The time.

    Raises
    ------
    ValueError
        If no pass starts at or holds the time.
    """
    table = passes_sumer()
    offset = np.abs((table["time_start"] - time).to(u.s))
    nearest = int(np.argmin(offset))
    if offset[nearest] <= _tolerance_start:
        return table[nearest]
    holds = np.flatnonzero((table["time_start"] <= time) & (time < table["time_stop"]))
    if holds.size:
        return table[int(holds[0])]
    raise ValueError(
        f"No pass of the table starts at or holds {time.isot}; "
        f"the nearest starts at {table['time_start'][nearest].isot}."
    )


def scene_sumer(
    time_start: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    velocity_max: None | u.Quantity = None,
    velocity_continuum: None | u.Quantity = 100 * u.km / u.s,
    velocities_neighbour: None | u.Quantity = [-175, +215] * u.km / u.s,
    height_neighbour_max: float = 0.15,
    fill: None | u.Quantity = None,
    remove_bad_steps: bool = True,
    trim_slit: bool = True,
    remove_stripes: None | bool = None,
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    axis_velocity: str = "velocity",
) -> soho.sumer.SpectrographObservation:
    r"""
    Create a synthetic scene of the :math:`\text{O\,V}\;630\,\AA` line from SUMER.

    The scene is a raster of the Solar Ultraviolet Measurements of Emitted
    Radiation (SUMER) spectrograph, which observed the line itself,
    in the second order of its grating.

    The level-1 files are those of :func:`~esis.flights.f1.data.path_sumer`,
    loaded with :meth:`~soho.sumer.SpectrographObservation.from_level1`,
    which puts the zero of the velocities at the mean position of the line,
    since SUMER has no absolute wavelength scale.
    Bad raster steps are replaced, the rows beyond the ends of the slit are
    left out, and the stripes of the scans of the polar coronal holes are
    removed, before :func:`~esis.data.synth.scene_sumer_observation` makes
    the scene, which subtracts the neighbouring lines of O V where they are
    weak, so that its wings are kept as far as the window reaches.

    Parameters
    ----------
    time_start
        The start of a pass of :func:`passes_sumer`, to within a minute,
        or a time during it, if `time_stop` is :obj:`None`;
        otherwise the earliest start of the exposures.
    time_stop
        If :obj:`None`, the pass named by `time_start` is used,
        with its exposure time and slit.
        Otherwise, every exposure on O V which started before it is used,
        whatever its exposure time and slit,
        which must then be the same for all of them.
    velocity_max
        The largest speed of the scene,
        as in :func:`~esis.data.synth.scene_sumer_observation`.
    velocity_continuum
        The least speed of the columns the continuum is measured from,
        as in :func:`~esis.data.synth.scene_sumer_observation`.
    velocities_neighbour
        Where the neighbouring lines of O V are expected,
        as in :func:`~esis.data.synth.scene_sumer_observation`.
    height_neighbour_max
        The height above which a neighbouring line is not subtracted,
        as in :func:`~esis.data.synth.scene_sumer_observation`.
    fill
        The radiance of the voxels which SUMER did not observe,
        as in :func:`~esis.data.synth.scene_sumer_observation`.
    remove_bad_steps
        Whether to replace bad raster steps,
        with :meth:`~soho.sumer.SpectrographObservation.remove_bad_steps`.
    trim_slit
        Whether to leave out the rows beyond the ends of the slit,
        with :meth:`~soho.sumer.SpectrographObservation.trim_slit`.
    remove_stripes
        Whether to remove the stripes along the slit,
        with :meth:`~soho.sumer.SpectrographObservation.remove_stripes`.
        If :obj:`None`, they are removed from the scans of the polar coronal
        holes, as the table of the passes marks them.
    axis_detector_x
        The logical axis corresponding to the steps of the raster.
    axis_detector_y
        The logical axis corresponding to the position along the slit.
    axis_velocity
        The logical axis corresponding to changes in line-of-sight velocity.

    Examples
    --------
    Load the pass of the quiet Sun near disk center which is held out of the
    training set for testing, and display it as a false-color image.

    .. jupyter-execute::

        import esis

        scene = esis.flights.f1.data.synth.scene_sumer("1996-09-16T16:07")

        scene.show();
    """
    start = astropy.time.Time(time_start)
    timedelta = None
    slit = None
    polar = False
    if time_stop is None:
        row = _pass(start)
        start = row["time_start"]
        stop = row["time_stop"]
        timedelta = row["exposure"]
        slit = row["slit"]
        polar = bool(row["polar"])
    else:
        stop = astropy.time.Time(time_stop)

    observation = soho.sumer.SpectrographObservation.from_level1(
        path=path_sumer(start, stop),
        wavelength=O_V.wavelength,
        order=2,
        time_start=start,
        time_stop=stop,
        timedelta=timedelta,
        slit=slit,
        axis_wavelength=axis_velocity,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )
    if remove_bad_steps:
        observation = observation.remove_bad_steps()
    if trim_slit:
        observation = observation.trim_slit()
    if remove_stripes is None:
        remove_stripes = polar
    if remove_stripes:
        observation = observation.remove_stripes()

    return esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_max=velocity_max,
        velocity_continuum=velocity_continuum,
        velocities_neighbour=velocities_neighbour,
        height_neighbour_max=height_neighbour_max,
        fill=fill,
    )

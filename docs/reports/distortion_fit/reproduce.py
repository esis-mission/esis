#!/usr/bin/env python3
"""
Reproduce the committed distortion fit of ESIS-I from the model and the data.

On one machine, the whole fit is one command::

    python reproduce.py run <directory>

It writes every pass of the iteration into ``<directory>`` and the tables to
commit into ``<directory>/committed``.  A stage whose output is already there
is not run again, so an interrupted run picks up where it stopped.  Compare
the result with the committed tables, or with another run, with::

    python reproduce.py compare <directory>/committed [<other>]

The field stop is iterated with the fit.  Pass 0 fits every channel from
scratch with the regular octagon of the design and measures the field stop
through that fit; each later pass refits, from the window edges on, with the
stop the pass before it measured, reusing pass 0's absolute fits of the
channels, and measures it again.  The iteration ends at the first pass whose
measurement does not turn the sides of the stop
(:data:`esis.flights.f1.optics._fits._record.TOLERANCE`), after at most four
passes; that pass is the fit, which is scored on frames across the flight and
recorded.

The stages of a pass, which a cluster can run as separate jobs (see
``slurm/chain.sh``)::

    python reproduce.py channel <c> <directory>   # absolute fit of channel c
    python reproduce.py edges <directory>         # window edges and their drift
    python reproduce.py combine <directory>       # outline, shared and internal stages
    python reproduce.py defocus <directory>       # focus of each sector, per frame
    python reproduce.py pointing <t> <directory>  # pointing of frame t
    python reproduce.py gather <directory>        # pointing rows -> one table
    python reproduce.py fieldstop <directory>     # measure the field stop
    python reproduce.py converged <directory>     # exit 0 if it stayed, 3 if not
    python reproduce.py accept <directory>        # score the fit across the flight
    python reproduce.py record <out> <pass>...    # the tables to commit

Each pass is fit with the field stop in ``field_stop_input.ecsv`` in its
directory, or with the regular octagon if there is none.

Options, given before the command, are also read from the environment:
``--device`` (``ESIS_DEVICE``: ``cpu``, the default, or ``cuda`` for a GPU,
which needs ``torch``), ``--workers`` (``ESIS_WORKERS``, the processes the
capture and the channels are spread over, default 1), ``--seed``
(``ESIS_SEED``, the seed of the capture and of the samples every channel is
linearized on, default 0) and ``--num-scene`` (``ESIS_NUM_SCENE``, the
sampling of the AIA scene, default 401: 2.1 arcsec, about 2.8 ESIS pixels).
"""

from __future__ import annotations
import argparse
import logging
import os
import pathlib
import shutil
import sys

import numpy as np
import astropy.table
import astropy.units as u
import esis
from esis.flights.f1.optics._fits import _fits, _record
from esis.flights.f1.optics._instruments import _instruments

# a dependency configures the root logger at INFO, which lets numba report
# every device allocation
logging.getLogger("numba").setLevel(logging.WARNING)

NUM_CHANNEL = 4
NUM_FRAME = 30
"""The frames of :func:`esis.flights.f1.data.level_1`, the flight."""
FRAME_REFERENCE = 15
"""The frame the reference is fit to."""
DRIFT_DEGREE = 3
"""The degree of the polynomial in the frame index that smooths the window drift."""
DEFOCUS_DEGREE = 2
"""The degree of the polynomial in the frame index through each sector's focus."""
MAX_PASSES = 4

options = argparse.Namespace(device=None, workers=1, seed=0, num_scene=401)


def _channels(directory: pathlib.Path) -> list[esis.optics.DistortionParameters]:
    return [
        esis.optics.DistortionParameters.from_file(directory / f"channel_{c}.ecsv")
        for c in range(NUM_CHANNEL)
    ]


def field_stop_fit(directory: pathlib.Path) -> None | pathlib.Path:
    """
    Find the field stop a directory's tables were fit with.

    A pass was fit with the stop in its ``field_stop_input.ecsv``, or with
    the regular octagon if there is none; its own ``field_stop.ecsv`` is what
    it measured.  A directory of committed tables, which has no channel
    fits, was fit with the stop in its ``field_stop.ecsv``.

    Parameters
    ----------
    directory
        A pass, or a directory of committed tables.

    Returns
    -------
    The path of the stop's vertices, or :obj:`None` for the regular octagon.
    """
    path = directory / "field_stop_input.ecsv"
    if path.exists():
        return path
    path = directory / "field_stop.ecsv"
    if path.exists() and not (directory / "channel_0.ecsv").exists():
        return path
    return None


def _field_stop(directory: pathlib.Path) -> None:
    """Model the field stop the directory's tables were fit with."""
    _fits.path_field_stop = field_stop_fit(directory)


def _read(directory: pathlib.Path, name: str) -> astropy.table.QTable:
    return astropy.table.QTable.read(directory / name, format="ascii.ecsv")


def channel(c: int, directory: pathlib.Path) -> None:
    """Fit one channel on its own and save it as a one-row table."""
    _field_stop(directory)
    _fits.fit_distortion_channel(
        channel=c,
        time=FRAME_REFERENCE,
        num_scene=options.num_scene,
        device=options.device,
        workers=options.workers,
        seed=options.seed,
        path=directory / f"channel_{c}.ecsv",
        directory=directory,
    )


def edges(directory: pathlib.Path) -> None:
    """Measure the window edges near the absolute fits' outlines, and their drift."""
    _field_stop(directory)
    _fits.measure_window_edges(
        parameters=_channels(directory),
        num_scene=options.num_scene,
        frame_reference=FRAME_REFERENCE,
        drift_degree=DRIFT_DEGREE,
        directory=directory,
        path=directory / "window_edges.ecsv",
        path_drift=directory / "window_drift.ecsv",
        seed=options.seed,
    )


def combine(directory: pathlib.Path) -> None:
    """Combine the channels' own fits into the reference."""
    _field_stop(directory)
    _fits.fit_distortion_reference(
        parameters=_channels(directory),
        edges=directory / "window_edges.ecsv",
        time=FRAME_REFERENCE,
        num_scene=options.num_scene,
        device=options.device,
        workers=options.workers,
        seed=options.seed,
        path=directory / "distortion_reference.ecsv",
        directory=directory,
    )


def defocus(directory: pathlib.Path) -> None:
    """Measure the focus of each sector of the primary through the flight."""
    _field_stop(directory)
    _fits.fit_defocus_history(
        reference=directory / "distortion_reference.ecsv",
        num_scene=options.num_scene,
        frame_reference=FRAME_REFERENCE,
        drift=directory / "window_drift.ecsv",
        drift_degree=DRIFT_DEGREE,
        degree=DEFOCUS_DEGREE,
        path=directory / "defocus.ecsv",
        directory=directory,
        seed=options.seed,
    )


def pointing(t: int, directory: pathlib.Path) -> None:
    """Fit the pointing of one frame and save it as a one-row table."""
    _field_stop(directory)
    parameters = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    )
    _fits.fit_distortion_pointing(
        instrument=parameters.to_instrument(_fits._base(None)),
        num_scene=options.num_scene,
        device=options.device,
        frame_reference=FRAME_REFERENCE,
        frames=(t,),
        drift=directory / "window_drift.ecsv",
        drift_degree=DRIFT_DEGREE,
        defocus=directory / "defocus.ecsv",
        path=directory / f"pointing_{t:02d}.ecsv",
        directory=directory,
        seed=options.seed,
        # every row absolute, the reference frame's too: gather takes them
        # relative once
        relative=False,
    )


def gather(directory: pathlib.Path) -> None:
    """
    Gather the per-frame pointing rows into one table, relative to the reference.

    Raises
    ------
    FileNotFoundError
        If a frame's row is missing.
    """
    paths = sorted(directory.glob("pointing_*.ecsv"))
    if len(paths) != NUM_FRAME:
        raise FileNotFoundError(f"{len(paths)} of {NUM_FRAME} frames in {directory}")
    tables = [astropy.table.QTable.read(p, format="ascii.ecsv") for p in paths]
    table = astropy.table.vstack(tables, metadata_conflicts="silent")
    table.sort("frame")
    table.meta.update(tables[0].meta)
    table = _fits.pointing_relative(table, frame=FRAME_REFERENCE)
    table.write(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv", overwrite=True
    )


def fieldstop(directory: pathlib.Path) -> None:
    """Measure the shape of the field stop from every channel's O V window."""
    _field_stop(directory)
    _fits.measure_field_stop(
        reference=esis.optics.DistortionParameters.from_file(
            directory / "distortion_reference.ecsv"
        ),
        pointing=_read(directory, "distortion_pointing.ecsv"),
        num_scene=options.num_scene,
        directory=directory,
        path=directory / "field_stop.ecsv",
        path_sides=directory / "field_stop_sides.ecsv",
        seed=options.seed,
    )


def converged(directory: pathlib.Path) -> bool:
    """Decide whether the pass's measurement of the field stop moved it."""
    table = _read(directory, "field_stop.ecsv")
    change = _record.field_stop_change(table)
    result = _record.field_stop_converged(table)
    print(
        f"{directory}: the sides of the field stop turned by "
        f"{change['shape_tilt_deg']:.4f} deg rms (tolerance {_record.TOLERANCE}) "
        f"and moved by {change['shape_offset_um']:.3f} um rms; "
        + ("converged" if result else "not converged")
    )
    return result


def accept(directory: pathlib.Path) -> None:
    """Score the reference on frames across the flight, without an inversion."""
    _field_stop(directory)
    _fits.acceptance(
        reference=esis.optics.DistortionParameters.from_file(
            directory / "distortion_reference.ecsv"
        ),
        pointing=_read(directory, "distortion_pointing.ecsv"),
        edges=_read(directory, "window_edges.ecsv"),
        num_scene=options.num_scene,
        device=options.device,
        path=directory / "acceptance.ecsv",
        directory=directory,
        seed=options.seed,
    )


def record(directory: pathlib.Path, passes: list[pathlib.Path]) -> None:
    """Write the tables to commit from the passes of the iteration."""
    final = passes[-1]
    instrument = _fits._base(None)
    instrument.field_stop.vertices = esis.flights.f1.optics.field_stop_vertices(
        final / "field_stop_input.ecsv"
    )
    clocking = _record.measure_clocking(
        reference=esis.optics.DistortionParameters.from_file(
            final / "distortion_reference.ecsv"
        ),
        pointing=_read(final, "distortion_pointing.ecsv"),
        instrument=instrument,
    )
    _record.record_distortion_fit(passes=passes, directory=directory, clocking=clocking)
    print(f"{len(passes)} passes -> {directory}")


def _stage(name: str, output: pathlib.Path, function, *args) -> None:
    """Run a stage of :func:`run` unless its output is already there."""
    if output.exists():
        print(f"{name}: {output} exists, skipped")
        return
    print(f"{name} -> {output}", flush=True)
    function(*args)


def run_pass(directory: pathlib.Path, k: int) -> pathlib.Path:
    """Run pass `k` of the iteration in this process."""
    p = directory / f"pass_{k}"
    p.mkdir(parents=True, exist_ok=True)
    if k > 0:
        if not (p / "field_stop_input.ecsv").exists():
            shutil.copyfile(
                directory / f"pass_{k - 1}" / "field_stop.ecsv",
                p / "field_stop_input.ecsv",
            )
        for c in range(NUM_CHANNEL):
            name = f"channel_{c}.ecsv"
            if not (p / name).exists():
                shutil.copyfile(directory / "pass_0" / name, p / name)
    for c in range(NUM_CHANNEL):
        _stage(f"pass {k} channel {c}", p / f"channel_{c}.ecsv", channel, c, p)
    # the drift is written last, so it marks the stage done
    _stage(f"pass {k} edges", p / "window_drift.ecsv", edges, p)
    _stage(f"pass {k} combine", p / "distortion_reference.ecsv", combine, p)
    _stage(f"pass {k} defocus", p / "defocus.ecsv", defocus, p)
    for t in range(NUM_FRAME):
        _stage(f"pass {k} pointing {t}", p / f"pointing_{t:02d}.ecsv", pointing, t, p)
    _stage(f"pass {k} gather", p / "distortion_pointing.ecsv", gather, p)
    _stage(f"pass {k} field stop", p / "field_stop.ecsv", fieldstop, p)
    return p


def run(directory: pathlib.Path) -> None:
    """
    Run the whole fit, iterating the field stop, and record it.

    Raises
    ------
    SystemExit
        If the field stop has not converged after :data:`MAX_PASSES` passes.
    """
    directory.mkdir(parents=True, exist_ok=True)
    passes = []
    for k in range(MAX_PASSES):
        passes.append(run_pass(directory, k))
        # the fit with the regular octagon never ends the iteration
        if converged(passes[-1]) and k > 0:
            break
    else:
        raise SystemExit(f"the field stop did not converge in {MAX_PASSES} passes")
    final = passes[-1]
    _stage("accept", final / "acceptance.ecsv", accept, final)
    record(directory / "committed", passes)


def compare(directory: pathlib.Path, other: None | pathlib.Path = None) -> None:
    """
    Compare a fit with the committed one, or with another fit.

    Prints the differences that matter downstream: the pointing through the
    flight, the focus of each sector, the drift of the windows, the vertices
    of the field stop, the scores of the reference, and the registration of
    the channels the acceptance measures.  The seed of the capture alone
    moves the pointing by a few hundredths of an arcsecond and the
    registration by about 0.02 px.
    """
    if other is None:
        other = _instruments._directory_data
    names = _record.TABLES + ("field_stop.ecsv",)
    a = {n: _read(directory, n) for n in names if (directory / n).exists()}
    b = {n: _read(other, n) for n in names if (other / n).exists()}
    print(f"{directory} against {other} (whose values are in parentheses)")

    pa, pb = a["distortion_pointing.ecsv"], b["distortion_pointing.ecsv"]
    for name, unit in (
        ("pitch", u.arcsec),
        ("yaw", u.arcsec),
        ("z_primary", u.um),
        ("drift_x", u.pix),
        ("drift_y", u.pix),
    ):
        d = np.asarray((pa[name] - pb[name]).to_value(unit))
        print(f"  {name}: rms {_rms(d):.3f} {unit}, max {np.abs(d).max():.3f}")

    fa, fb = a["field_stop.ecsv"], b["field_stop.ecsv"]
    d = np.hypot(*(np.asarray((fa[k] - fb[k]).to_value(u.um)) for k in ("x", "y")))
    print(f"  field stop vertices: rms {_rms(d):.2f} um, max {d.max():.2f}")

    sa = a["distortion_reference.ecsv"].meta["scores"]
    sb = b["distortion_reference.ecsv"].meta["scores"]
    for stage in ("absolute", "shared"):
        pairs = zip(sa[stage], sb[stage])
        print(
            f"  {stage} correlation: "
            + ", ".join(f"{x:.4f} ({y:.4f})" for x, y in pairs)
        )
    ra = _record.registration_rms(a["acceptance.ecsv"], a["window_drift.ecsv"])
    rb = _record.registration_rms(b["acceptance.ecsv"], b["window_drift.ecsv"])
    for line in ra:
        print(f"  registration {line}: {ra[line]:.3f} px ({rb[line]:.3f})")


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def configure(argv: None | list[str] = None) -> argparse.Namespace:
    """
    Read the options, from the arguments or the environment, into :data:`options`.

    Parameters
    ----------
    argv
        The arguments; if :obj:`None`, only the environment is read.

    Returns
    -------
    The command and its arguments, and the options.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the distortion fit of ESIS-I.",
        epilog="The commands are listed in the docstring of this script.",
    )
    environ = os.environ.get
    parser.add_argument("--device", default=environ("ESIS_DEVICE", "cpu"))
    parser.add_argument("--workers", type=int, default=environ("ESIS_WORKERS", 1))
    parser.add_argument("--seed", type=int, default=environ("ESIS_SEED", 0))
    parser.add_argument("--num-scene", type=int, default=environ("ESIS_NUM_SCENE", 401))
    if argv is not None:
        parser.add_argument("command")
        parser.add_argument("arguments", nargs="*")
    parsed = parser.parse_args([] if argv is None else argv)
    device = parsed.device.strip().lower()
    options.device = None if device in ("", "cpu", "none") else parsed.device
    options.workers = parsed.workers
    options.seed = parsed.seed
    options.num_scene = parsed.num_scene
    return parsed


def main(argv: list[str]) -> int:
    """
    Parse the options and run one command.

    Raises
    ------
    SystemExit
        If the command is not one of this script's.
    """
    parsed = configure(argv)
    command = parsed.command
    paths = [pathlib.Path(a) for a in parsed.arguments]
    if command == "run":
        run(*paths)
    elif command == "channel":
        channel(int(parsed.arguments[0]), paths[1])
    elif command == "pointing":
        pointing(int(parsed.arguments[0]), paths[1])
    elif command in ("edges", "combine", "defocus", "gather", "fieldstop", "accept"):
        globals()[command](*paths)
    elif command == "converged":
        return 0 if converged(*paths) else 3
    elif command == "record":
        record(paths[0], paths[1:])
    elif command == "compare":
        compare(*paths)
    else:
        raise SystemExit(f"unknown command {command!r}; see {__file__}")
    return 0


if __name__ == "__main__":
    # a guard the worker processes of the capture need
    raise SystemExit(main(sys.argv[1:]))

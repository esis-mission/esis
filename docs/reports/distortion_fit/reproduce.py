#!/usr/bin/env python3
"""
Reproduce the committed distortion fit of ESIS-I from the model and the data.

The fit is three stages (see
:func:`esis.flights.f1.optics.fit_distortion_reference`); the first, the
absolute fit of each channel, is an hour on one GPU per channel, so it is
split across jobs.  Run one of::

    python reproduce.py channel <c> <directory>   # absolute fit of channel c
    python reproduce.py combine <directory>       # shared optics + alignment -> ECSV
    python reproduce.py pointing <t> <directory>  # per-frame pointing of frame t
    python reproduce.py gather <directory>        # pointing rows -> ECSV

Environment: ESIS_DEVICE (default ``cuda``), ESIS_WORKERS (default 6),
ESIS_FREE (colon-separated fields to fit per channel; default all).
"""

import logging
import os
import pathlib
import re
import sys

import astropy.table
import esis
from esis.flights.f1.optics._fits import _fits

# a dependency configures the root logger at INFO, which lets numba report
# every device allocation
logging.getLogger("numba").setLevel(logging.WARNING)

DEVICE = os.environ.get("ESIS_DEVICE", "cuda")
WORKERS = int(os.environ.get("ESIS_WORKERS", "6"))
# ESIS_FREE names the fields fit per channel, separated by colons (a comma
# would be split by Slurm's --export); unset fits every field
FREE = (
    tuple(n for n in re.split(r"[:,\s]+", os.environ["ESIS_FREE"]) if n)
    if os.environ.get("ESIS_FREE")
    else None
)

# ESIS_MERIT selects the comparison the fit maximizes: "correlation" (the
# default) or "least_squares", which also fits the degradation of each channel
MERIT = os.environ.get("ESIS_MERIT", "correlation")


def channel(c: int, directory: pathlib.Path) -> None:
    """Fit one channel absolutely and save it as a one-row ECSV."""
    instrument = _fits._base(None)
    scene, observation = _fits._frame(15, 401)
    log = _fits._logger(directory, f"channel_{c}")
    channel = instrument[dict(channel=c)]
    p0 = esis.optics.DistortionParameters.from_instrument(channel)
    merit = esis.optics.LinearMerit(
        instrument=channel,
        parameters=p0,
        scene=scene,
        observation=observation[dict(channel=c)],
        device=DEVICE,
        merit=MERIT,
    )
    fitted = esis.optics.fit_distortion(
        objective=merit,
        parameters=p0,
        bounds=esis.flights.f1.optics.distortion_fit_bounds(p0),
        workers=WORKERS,
        free=FREE if FREE is not None else _fits._names_absolute(p0),
        log=log,
    )
    fitted.to_file(directory / f"channel_{c}.ecsv", metadata=dict(channel=c))
    log(
        f"channel {c}: correlation {merit.correlation(fitted):.4f}, "
        f"least-squares score {merit.score(fitted):.4f}"
    )


def combine(directory: pathlib.Path) -> None:
    """Run the shared and internal stages from the saved per-channel fits."""
    parameters = [
        esis.optics.DistortionParameters.from_file(directory / f"channel_{c}.ecsv")
        for c in range(4)
    ]
    _fits.fit_distortion_reference(
        device=DEVICE,
        workers=WORKERS,
        merit=MERIT,
        channels=(),
        parameters=parameters,
        free=FREE,
        path=directory / "distortion_reference.ecsv",
        directory=directory,
    )


def pointing(t: int, directory: pathlib.Path) -> None:
    """Fit the pointing of one frame and save it as a one-row ECSV."""
    parameters = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    )
    _fits.fit_distortion_pointing(
        instrument=parameters.to_instrument(_fits._base(None)),
        device=DEVICE,
        merit=MERIT,
        frames=(t,),
        path=directory / f"pointing_{t:02d}.ecsv",
        directory=directory,
    )


def gather(directory: pathlib.Path) -> None:
    """Gather the per-frame pointing rows into the committed table."""
    tables = [
        astropy.table.QTable.read(p, format="ascii.ecsv")
        for p in sorted(directory.glob("pointing_*.ecsv"))
    ]
    table = astropy.table.vstack(tables, metadata_conflicts="silent")
    table.sort("frame")
    table.meta.update(tables[0].meta)
    table = _fits.pointing_relative(table, frame=15)
    table.write(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv", overwrite=True
    )
    print(f"{len(table)} frames -> {directory / 'distortion_pointing.ecsv'}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "channel":
        channel(int(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "combine":
        combine(pathlib.Path(sys.argv[2]))
    elif command == "pointing":
        pointing(int(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "gather":
        gather(pathlib.Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)

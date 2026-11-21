import csv
import pathlib
import functools
import numpy as np
import astropy.time
import named_arrays as na
from .._fits._fits import _path_cache, _retrieve

__all__ = [
    "path_sumer",
]

_url = "https://github.com/esis-mission/esis-data-2019/releases/download/sumer-v1.0"
"""
The release of ``esis-data-2019`` which holds the level-1 files of SUMER,
one asset for each file, beside a manifest of them.
"""

_hash_manifest = (
    "sha256:87208ea7eb567d302d531fe7b04c52d2007668485c326dfc6bb00f81077f8f06"
)
"""The hash of the manifest of the release, which holds the hash of each file."""


def _path_directory() -> pathlib.Path:
    """
    Return the directory which the files of SUMER are downloaded into.

    It is under :func:`~esis.flights.f1.data._fits._fits._path_cache`.
    """
    return _path_cache() / "sumer"


@functools.cache
def _manifest() -> tuple[dict[str, str], ...]:
    """
    Return the rows of the manifest of the release, one for each file.

    Each row gives the name, size, and checksum of a file, and its observation
    in ESA's catalogue of SOHO, in order of the start of the observations.
    """
    path = _retrieve(
        url=f"{_url}/manifest.csv",
        known_hash=_hash_manifest,
        fname="manifest.csv",
        path=_path_directory(),
        extract_dir=None,
    )
    with open(path, newline="") as f:
        rows = csv.DictReader(f)
        return tuple(sorted(rows, key=lambda row: row["begin_date"]))


def path_sumer(
    time_start: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    axis_time: str = "time",
) -> na.ScalarArray:
    """
    Return the paths of the level-1 files of SUMER which observed a time range.

    The files are those of the Solar Ultraviolet Measurements of Emitted
    Radiation (SUMER) spectrograph whose observations overlap the time range,
    and are downloaded if needed.

    The files are those whose rasters of O V 629.7 Angstroms passed the
    survey of the SOHO archive for the training set of the neural-network
    inversion of ESIS.
    They are downloaded from the
    `esis-data-2019 <https://github.com/esis-mission/esis-data-2019/releases/tag/sumer-v1.0>`_
    repository the first time they are needed, one by one,
    and each is checked against the checksum in the manifest of the release,
    so this does not depend on the SOHO archive.
    Load them with :meth:`~soho.sumer.SpectrographObservation.from_level1`.

    Parameters
    ----------
    time_start
        The start of the time range.
    time_stop
        The end of the time range.
        If :obj:`None`, the files whose observations include `time_start`
        are returned.
    axis_time
        The logical axis along which the paths are arranged,
        in order of the start of their observations.

    Raises
    ------
    ValueError
        If no file of the release observed the time range.
    """
    start = astropy.time.Time(time_start)
    stop = start if time_stop is None else astropy.time.Time(time_stop)

    rows = [
        row
        for row in _manifest()
        if astropy.time.Time(row["begin_date"]) <= stop
        and astropy.time.Time(row["end_date"]) >= start
    ]
    if not rows:
        raise ValueError(
            f"No file of the release observed {start.isot} to {stop.isot}."
        )

    paths = [
        _retrieve(
            url=f"{_url}/{row['file']}",
            known_hash=f"sha256:{row['sha256']}",
            fname=row["file"],
            path=_path_directory(),
            extract_dir=None,
        )
        for row in rows
    ]
    return na.ScalarArray(np.array(paths, dtype=object), axes=axis_time)

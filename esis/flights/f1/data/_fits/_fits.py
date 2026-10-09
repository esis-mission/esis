import os
import functools
import pathlib
import numpy as np
import numpy.typing as npt
import pooch
import named_arrays as na

__all__ = [
    "path_directory",
    "path_fits",
]


doi = "10.5281/zenodo.21997280"
"""
The DOI of the Zenodo record holding the Level-0 data from the 2019 flight.

The archive is served from GitHub while that record is being prepared, but it
is the same file, so this is the DOI to cite.
"""

_url = (
    "https://github.com/esis-mission/esis-data-2019/releases/download"
    "/v1.0/esis-2019-level0-fits.tar.gz"
)
_hash = "sha256:eeb60413f905ef30e832be6a70e2fc03e9cf08df2ee471c9d3828eff2bc1fd7f"


def _path_cache() -> pathlib.Path:
    """
    Return the directory which downloaded data is unpacked into.

    Defaults to ``~/.esis/data``, and can be moved by setting the
    ``ESIS_DATA_DIR`` environment variable.
    """
    return pathlib.Path(
        os.environ.get(
            "ESIS_DATA_DIR",
            pathlib.Path.home() / ".esis/data",
        )
    )


@functools.cache
def _retrieve(
    url: str,
    known_hash: str,
    fname: str,
    path: pathlib.Path,
    extract_dir: str,
) -> pathlib.Path:
    """
    Download a tar archive into `path` and return the directory it was unpacked into.

    If the archive is already in `path`, it is not downloaded again,
    but checking it means hashing the whole file,
    so the result is kept for the rest of the process.

    Parameters
    ----------
    url
        Where to download the archive from.
    known_hash
        The hash of the archive, such as ``"sha256:..."``.
    fname
        The name of the downloaded archive in `path`.
    path
        The directory to download the archive into.
    extract_dir
        The name of the directory in `path` to unpack the archive into.
    """
    files = pooch.retrieve(
        url=url,
        known_hash=known_hash,
        fname=fname,
        path=path,
        processor=pooch.Untar(extract_dir=extract_dir),
    )

    return pathlib.Path(files[0]).parent


def path_directory() -> pathlib.Path:
    """
    Return the directory containing the FITS files captured during the flight.

    The files are too large to distribute with this package, so they are
    downloaded from `Zenodo <https://doi.org/10.5281/zenodo.21997280>`_ the
    first time they are needed and unpacked into :func:`_path_cache`.
    """
    return _retrieve(
        url=_url,
        known_hash=_hash,
        fname="esis-2019-level0-fits.tar.gz",
        path=_path_cache(),
        extract_dir="fits",
    )


def path_fits(
    axis_time: str,
    axis_channel: str,
) -> na.ScalarArray[npt.NDArray[pathlib.Path]]:
    """
    Construct an array of paths to all the FITS files captured during the flight.

    Parameters
    ----------
    axis_time
        The name of the logical axis representing time.
    axis_channel
        The name of the logical axis representing the different channels.
    """
    path = path_directory().glob("*.fit.gz")
    path = sorted(list(path))
    path = np.array(path)
    path = na.ScalarArray(
        ndarray=path.reshape(4, -1),
        axes=(axis_channel, axis_time),
    )
    return path

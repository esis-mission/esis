import csv
import pathlib
import numpy as np
import numpy.typing as npt
import astropy.units as u
import astropy.time
import pooch
import named_arrays as na
from .._fits._fits import _path_cache

__all__ = [
    "path_aia",
]

_url = (
    "https://github.com/esis-mission/esis-data-2019/releases/download"
    "/aia-v1.0/esis-2019-aia-lev1.tar"
)
_hash = "sha256:48e5489aed4c4373d85df5530d25e39045f94f1378ccaaa8ecaa5924cc53e5b6"


def _path_directory() -> pathlib.Path:
    """
    Return the directory containing the AIA images.

    They are downloaded and unpacked into
    :func:`esis.flights.f1.data._fits._fits._path_cache` the first time they
    are needed.
    """
    files = pooch.retrieve(
        url=_url,
        known_hash=_hash,
        fname="esis-2019-aia-lev1.tar",
        path=_path_cache(),
        processor=pooch.Untar(extract_dir="aia"),
    )

    return pathlib.Path(files[0]).parent


def path_aia(
    wavelength: u.Quantity | na.AbstractScalarArray,
    axis_time: str,
    time_start: None | astropy.time.Time = None,
    time_stop: None | astropy.time.Time = None,
    limit: None | int = None,
) -> na.ScalarArray[npt.NDArray[pathlib.Path]]:
    """
    Construct an array of paths to the AIA images captured during the flight.

    The images are the Level 1 files of the 193 and 304 Angstrom channels of
    the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory
    in the JSOC's time slots from 18:06 to 18:12 UTC, 30 for each channel,
    which span the Level-1 observations of ESIS.
    They are too large to distribute with this package, so they are
    downloaded from the
    `esis-data-2019 <https://github.com/esis-mission/esis-data-2019/releases/tag/aia-v1.0>`_
    repository the first time they are needed.
    The archive also holds ``manifest.csv``, which lists the JSOC record,
    times, exposure, quality flag, and checksum of each image.

    This takes the place of :func:`sdo.aia.urls` and :func:`sdo.aia.download`
    for these images, so they can be used while the JSOC is unavailable,
    and the result can be passed to :func:`sdo.aia.prep`.

    Parameters
    ----------
    wavelength
        The AIA channel of each image, 193 or 304 Angstroms.
        A channel may appear more than once.
    axis_time
        The name of the logical axis representing time.
    time_start
        If not :obj:`None`, only images whose exposure began at or after this
        time are used.
    time_stop
        If not :obj:`None`, only images whose exposure began at or before this
        time are used.
    limit
        If not :obj:`None`, at most this many images of each channel are used,
        spread evenly over the time range.

    Raises
    ------
    ValueError
        If the archive does not hold a channel in `wavelength`, or holds no
        images in the time range.

    Notes
    -----
    The channels are imaged a few seconds apart, so along `axis_time` the
    images of every channel are used or left out together: an index along
    `axis_time` is used only if the exposure of each image there began
    within the time range.
    """
    directory = _path_directory()

    with open(directory / "manifest.csv", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: row["DATE-OBS"])

    rows_channel: dict[int, list[dict[str, str]]] = dict()
    for row in rows:
        rows_channel.setdefault(int(row["WAVELNTH"]), []).append(row)

    channel = sorted(rows_channel)
    axis_channel = f"_{axis_time}_channel"

    path = na.ScalarArray(
        ndarray=np.array(
            [[directory / row["file"] for row in rows_channel[c]] for c in channel],
            dtype=object,
        ),
        axes=(axis_channel, axis_time),
    )
    time = na.ScalarArray(
        ndarray=astropy.time.Time(
            [[row["DATE-OBS"] for row in rows_channel[c]] for c in channel],
        ),
        axes=(axis_channel, axis_time),
    )

    wavelength = na.as_named_array(wavelength)
    is_channel = wavelength == na.ScalarArray(channel * u.AA, axis_channel)
    if not np.all(np.any(is_channel, axis=axis_channel)).ndarray:
        raise ValueError(
            f"The archive holds the {channel} Angstrom channels, "
            f"not all of {wavelength=}."
        )
    index = np.argmax(is_channel, axis=axis_channel)
    path = path[index]
    time = time[index]

    if time_start is None:
        time_start = time.ndarray.min()
    if time_stop is None:
        time_stop = time.ndarray.max()

    is_inside = (time >= time_start) & (time <= time_stop)
    axes_other = tuple(a for a in is_inside.axes if a != axis_time)
    if axes_other:
        is_inside = np.all(is_inside, axis=axes_other)
    path = path[np.nonzero(is_inside)]

    num = path.shape[axis_time]
    if num == 0:
        raise ValueError(
            f"The archive holds no images between {time_start} and {time_stop}."
        )

    if limit is not None and limit < num:
        index_time = (na.arange(0, limit, axis=axis_time) + 1 / 2) * num // limit
        path = path[{axis_time: index_time.astype(int)}]

    return path

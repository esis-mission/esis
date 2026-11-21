import pathlib
import struct
import numpy as np
import astropy.units as u
import named_arrays as na
from esis.flights.f1.data._fits._fits import _path_cache, _retrieve

_url = (
    "https://github.com/esis-mission/esis-data-f2/releases/download"
    "/metrology-v1.0/esis-f2-metrology.tar"
)
_hash = "sha256:32488ca9a4e0d5a6a562cf97a37e97f76ebd43668e71baf11b5441be8bfedd5b"

_serial_numbers = ("SN2", "SN3")
"""The serial numbers of the primary mirrors built for the second flight."""


def _path_directory() -> pathlib.Path:
    """
    Return the directory containing the measurements of the optics of the second flight.

    They are downloaded and unpacked into
    :func:`esis.flights.f1.data._fits._fits._path_cache` the first time they
    are needed.
    """
    _retrieve(
        url=_url,
        known_hash=_hash,
        fname="esis-f2-metrology.tar",
        path=_path_cache(),
        extract_dir="metrology-f2",
    )
    return _path_cache() / "metrology-f2"


def _read_metropro(path: pathlib.Path) -> tuple[u.Quantity, u.Quantity]:
    """
    Read the map in a MetroPro file with a header of format 1.

    Return the height of the surface, NaN where the map has no data, and the
    distance between the centers of neighboring pixels.

    Parameters
    ----------
    path
        The path to the file.

    Raises
    ------
    ValueError
        If the file does not begin with a MetroPro header of format 1.
    """
    data = path.read_bytes()

    size_header = 834
    if struct.unpack(">IhI", data[:10]) != (0x881B036F, 1, size_header):
        raise ValueError(f"{path} does not begin with a MetroPro header of format 1")

    # The intensity, if there is any, comes between the header and the map.
    (size_intensity,) = struct.unpack(">i", data[60:64])
    num_x, num_y = struct.unpack(">hh", data[68:72])
    (
        scale_factor,
        wavelength,
        _,
        obliquity_factor,
        _,
        spacing,
    ) = struct.unpack(">ffffff", data[164:188])
    (phase_res,) = struct.unpack(">h", data[218:220])

    counts = np.frombuffer(
        buffer=data,
        dtype=">i4",
        count=num_x * num_y,
        offset=size_header + size_intensity,
    ).reshape(num_y, num_x)

    # Each count is a fraction of a wave, and the largest counts mark the
    # pixels without data.
    resolution = {0: 4096, 1: 32768, 2: 131072}[phase_res]
    height = counts * (scale_factor * obliquity_factor * wavelength / resolution)
    height = np.where(counts >= 2147483640, np.nan, height)

    return height * u.m, spacing * u.m


def _map(
    path: pathlib.Path,
    axis_x: str,
    axis_y: str,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    Load the map in a MetroPro file as a function of position.

    The heights are in nanometers, and the positions are those of the centers
    of the pixels, in millimeters, about the center of the map, with :math:`x`
    to the right and :math:`y` up as Zygo's software shows the map.

    Parameters
    ----------
    path
        The path to the file.
    axis_x
        The name of the logical axis along the columns of the map.
    axis_y
        The name of the logical axis along the rows of the map.
    """
    data, spacing = _read_metropro(path)

    # Zygo's software shows the first row of the map at the top.
    data = data[::-1]
    height = na.ScalarArray(
        ndarray=data.to(u.nm),
        axes=(axis_y, axis_x),
    )

    num_y, num_x = data.shape
    x = na.arange(0, num_x, axis=axis_x) - (num_x - 1) / 2
    y = na.arange(0, num_y, axis=axis_y) - (num_y - 1) / 2

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=(x * spacing).to(u.mm),
            y=(y * spacing).to(u.mm),
        ),
        outputs=height,
    )

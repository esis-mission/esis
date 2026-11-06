import pathlib
import numpy as np
import astropy.units as u
import h5py
import named_arrays as na
from esis.flights.f1.data._fits._fits import _path_cache, _retrieve

__all__ = [
    "figure_measurement",
]

_url = (
    "https://github.com/esis-mission/esis-data-2019/releases/download"
    "/metrology-v1.0/esis-2019-metrology.tar"
)
_hash = "sha256:0b28d90c5cbbcb27e27dd6455a54d5801ea5af4ca2b83892bf8e4506a7db7195"


def _path_directory() -> pathlib.Path:
    """
    Return the directory containing our measurements of the optics.

    They are downloaded and unpacked into
    :func:`esis.flights.f1.data._fits._fits._path_cache` the first time they
    are needed.
    """
    _retrieve(
        url=_url,
        known_hash=_hash,
        fname="esis-2019-metrology.tar",
        path=_path_cache(),
        extract_dir="metrology",
    )
    return _path_cache() / "metrology"


def figure_measurement(
    axis_x: str = "primary_x",
    axis_y: str = "primary_y",
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    Load our measurement of the figure of the primary mirror which flew.

    We measured the figure of the primary mirror, serial number SN001, at
    Montana State University in June 2018, with an interferometer at the
    center of curvature of the mirror.
    The map is the mean of two measurements, with the mirror turned 180
    degrees about its axis between them.
    Piston, tilt, power, and the spherical aberration which a paraboloid
    shows at its center of curvature were removed from it.

    The map is downloaded from the
    `esis-data-2019 <https://github.com/esis-mission/esis-data-2019/releases/tag/metrology-v1.0>`_
    repository the first time it is needed, as the file which the software
    of the interferometer saved.

    The result is the height of the surface of the mirror, as a function of
    the position of the center of each pixel.
    Pixels which were not measured, a disk at the center of the mirror and
    the rim beyond its clear aperture, are NaN.
    The positions are about the centroid of the pixels which were measured,
    along the rows and the columns of the map.
    Which way the mirror was turned about its axis during the measurement,
    compared to the instrument which flew, is not known.

    Parameters
    ----------
    axis_x
        The name of the logical axis along the columns of the map.
    axis_y
        The name of the logical axis along the rows of the map.

    Examples
    --------
    Plot the map.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        from esis.flights.f1.optics import primaries

        figure = primaries.figures.figure_measurement()

        fig, ax = plt.subplots(constrained_layout=True)
        mappable = na.plt.pcolormesh(
            figure.inputs.x.to(u.mm).value,
            figure.inputs.y.to(u.mm).value,
            C=figure.outputs.to(u.nm).value,
            ax=ax,
            cmap="RdBu_r",
            vmin=-50,
            vmax=50,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (mm)")
        ax.set_ylabel("$y$ (mm)")
        fig.colorbar(mappable.ndarray.item(), ax=ax, label="height (nm)")
    """
    path = _path_directory() / "primary/PM_001_Figure_Promoted.h5"
    with h5py.File(path, "r") as file:
        group = file["measurement0/genraw"]
        data = group["data"][...]
        wavelength = u.Quantity(group.attrs["wavelength"].decode())
        spacing = u.Quantity(group.attrs["xpix"].decode())

    # The map is in waves, and the wedge of the measurement makes it the
    # height of the surface. Masked pixels hold the largest 32-bit number.
    data = np.where(data == np.finfo(np.float32).max, np.nan, data)
    height = na.ScalarArray(
        ndarray=(data.astype(float) * wavelength).to(u.nm),
        axes=(axis_y, axis_x),
    )

    measured = np.isfinite(height.ndarray)
    rows, columns = np.nonzero(measured)
    num_y, num_x = measured.shape
    x = na.arange(0, num_x, axis=axis_x) - columns.mean()
    y = na.arange(0, num_y, axis=axis_y) - rows.mean()

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=(x * spacing).to(u.mm),
            y=(y * spacing).to(u.mm),
        ),
        outputs=height,
    )

import pathlib
import numpy as np
import scipy.ndimage
import scipy.optimize
import astropy.units as u
import h5py
import named_arrays as na
import optika
import esis
from esis import _caching
from ... import _metrology

__all__ = [
    "figure_measurement",
]

_packages = ("esis", "numpy", "scipy", "h5py")
"""
The packages which work out :func:`_figure_measurement`.

This one is among them, since :mod:`joblib` keys the cache on the source of
:func:`_figure_measurement` but not on the source of the functions it calls,
which a new version of this package can change.
"""

_width_reference = 1 * u.mm
"""
The width of the Gaussian which smooths the reference before it is
subtracted from each map.

Smoothed this much, the reference keeps the structure which the maps share
with it, the errors of the interferometer, and loses most of the structure
which they do not: the noise of its measurement, the finish of the reference
sphere, and the rings of a speck of dust.
Smoothed by a quarter of a millimeter, the dust would leave a spot a few
nanometers deep in three of the maps, at the same place on the detector.
"""

_width_edge = 0.25 * u.mm
"""
The width of the border inside the clear aperture of each grating which is
left out of its map.

It is the rim where the ruled area of the grating meets the border of the
blank, which steps the phase of the light the interferometer measured.
"""

_angle_search = 8 * u.deg
"""
How far each grating is searched for turned about its axis, either way, in
the frame of the interferometer.
"""

_num_edge = 300
"""How many points along each edge of the clear aperture its fit samples."""

_sigma_fit = (3, 2, 1)
"""
The widths, in pixels, of the Gaussians which smooth each interferogram
before its edges are found, from the widest to the narrowest.

The clear aperture is found in the widest, and then fitted to each of the
others in turn.
"""


def figure_measurement(
    manufacturing_number: na.AbstractScalar,
    aperture: optika.apertures.AbstractPolygonalAperture,
    axis_x: str = "grating_x",
    axis_y: str = "grating_y",
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    Load our measurement of the figure of the gratings which flew.

    We measured each of the four gratings three times at Montana State
    University in June 2018, with an interferometer at the center of
    curvature of the grating, while we measured its radius of curvature,
    and aligned it again before each measurement.
    The maps are downloaded from the
    `esis-data-2019 <https://github.com/esis-mission/esis-data-2019/releases/tag/metrology-v1.1>`_
    repository the first time they are needed, as the files which the
    software of the interferometer saved, and are made into the height of
    the surface of each grating as follows.

    #.  The interferometer measured the difference in the optical path to the
        grating and back.
        Its software subtracts a measurement of its reference sphere when it
        analyzes a map, but it saved the maps before that, so the reference
        is subtracted here, smoothed by a Gaussian :data:`_width_reference`
        wide.
        Half of the difference is the height of the surface.
    #.  The clear aperture of the grating is a bright trapezoid in the
        interferogram saved with each map, and the grating moved and turned a
        little between its measurements.
        ``aperture`` is fitted to the edges of the trapezoid, with the size of
        the pixels which the software saved, and the map is interpolated
        onto a grid in the coordinates of the grating with the same spacing.
    #.  The map is cut to ``aperture``, less a border :data:`_width_edge`
        wide, where the ruled area of the grating meets the border of its
        blank.
    #.  Piston, tilt, and power are removed from each map, since they are
        how the grating was aligned rather than its figure.
        Its radius of curvature was measured on its own.
    #.  The three maps of each grating are averaged where all three of them
        measured it, and piston, tilt, and power are removed from the mean.

    The clear aperture is symmetric about the :math:`x` axis, so the fit
    cannot tell which way the grating faced the interferometer about it.
    The map is placed as the interferometer saw the grating, with the rows
    of its pixels along :math:`x` and its columns along :math:`y`.

    Parameters
    ----------
    manufacturing_number
        The number of the blank of each grating, such as ``"UBO-16-024"``.
        The result has the axes of this array, as well as ``axis_x`` and
        ``axis_y``.
    aperture
        The clear aperture of the gratings, in the coordinates of their
        surfaces, which must be convex and the same for every grating.
    axis_x
        The name of the logical axis along :math:`x`, from the inner edge of
        the clear aperture to its outer edge.
    axis_y
        The name of the logical axis along :math:`y`.

    Raises
    ------
    ValueError
        If the clear aperture differs from one grating to another, or if
        there is no measurement of one of the gratings.

    Examples
    --------
    Plot the map of each grating.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import esis

        grating = esis.flights.f1.optics.as_built_unfocused().grating
        figure = esis.flights.f1.optics.gratings.figures.figure_measurement(
            manufacturing_number=grating.manufacturing_number,
            aperture=grating.surface.aperture,
        )

        fig, axs = na.plt.subplots(
            axis_cols="channel",
            ncols=figure.outputs.shape["channel"],
            sharex=True,
            sharey=True,
            constrained_layout=True,
            figsize=(10, 3),
        )
        na.plt.pcolormesh(
            figure.inputs.y.to(u.mm).value,
            figure.inputs.x.to(u.mm).value,
            C=figure.outputs.to(u.nm).value,
            ax=axs,
            cmap="RdBu_r",
            vmin=-8,
            vmax=8,
        )
        for ax in axs.ndarray:
            ax.set_aspect("equal")
            ax.set_xlabel("$y$ (mm)")
        axs.ndarray[0].set_ylabel("$x$ (mm)")
    """
    vertices = aperture.vertices
    if aperture.transformation is not None:
        vertices = aperture.transformation(vertices)
    vertices = vertices.broadcast_to(vertices.shape)
    if tuple(vertices.shape) != ("vertex",):
        raise ValueError(
            "the clear aperture must be the same for every grating, "
            f"but its vertices have the shape {vertices.shape}."
        )
    corners = tuple(
        (float(x), float(y))
        for x, y in zip(
            vertices.x.ndarray.to_value(u.mm),
            vertices.y.ndarray.to_value(u.mm),
        )
    )

    numbers = na.as_named_array(manufacturing_number)
    height = []
    for number in numbers.ndarray.flat:
        x, y, height_number = _figure_measurement(
            number=str(number),
            corners=corners,
            width_reference=_width_reference.to_value(u.mm),
            width_edge=_width_edge.to_value(u.mm),
            known_hash=_metrology._hash,
            versions=_caching.versions(*_packages),
        )
        height.append(height_number)
    height = np.stack(height)

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=na.ScalarArray(x * u.mm, axes=axis_x),
            y=na.ScalarArray(y * u.mm, axes=axis_y),
        ),
        outputs=na.ScalarArray(
            ndarray=height.reshape(numbers.ndarray.shape + height.shape[1:]) * u.nm,
            axes=numbers.axes + (axis_x, axis_y),
        ),
    )


@esis.memory.cache
def _figure_measurement(
    number: str,
    corners: tuple[tuple[float, float], ...],
    width_reference: float,
    width_edge: float,
    known_hash: str,
    versions: dict[str, None | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute :func:`figure_measurement` for one grating, and remember it.

    Finding the clear aperture in its three interferograms takes a few
    seconds.

    ``known_hash`` and ``versions`` are not read here. They are arguments so
    that they are part of the key the cache is looked up by, so that a new
    release of the measurements, or of any of :data:`_packages`, works out
    the maps again rather than returning the old ones.

    Parameters
    ----------
    number
        The number of the blank of the grating.
    corners
        The corners of the clear aperture, in millimeters, in order around it.
    width_reference
        :data:`_width_reference`, in millimeters.
    width_edge
        :data:`_width_edge`, in millimeters.
    known_hash
        The hash of the archive of the measurements.
    versions
        The versions of :data:`_packages`.

    Returns
    -------
    x
        The :math:`x` coordinate of each row of the grid, in millimeters.
    y
        The :math:`y` coordinate of each column of the grid, in millimeters.
    height
        The height of the surface of the grating, in nanometers, along
        :math:`x` and :math:`y`.

    Raises
    ------
    ValueError
        If there is no measurement of the grating, or if the pixels of its
        maps are not all the same size.
    """
    directory = _metrology._path_directory() / "gratings"
    corners = np.array(corners)

    blank = number.split("-")[-1]
    paths = sorted(directory.glob(f"EUV_{blank}_Confocal_*.h5"))
    if not paths:
        raise ValueError(f"there is no measurement of the grating {number}.")

    with h5py.File(paths[0], "r") as file:
        xpix = file["measurement0/genraw"].attrs["xpix"]
    spacing = u.Quantity(_text(xpix)).to_value(u.mm)
    x = _axis(corners[:, 0], spacing)
    y = _axis(corners[:, 1], spacing)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    inside = _distance(corners, x_grid, y_grid) <= -width_edge

    maps = []
    for path in paths:
        height, intensity, spacing_path = _height(path, width_reference)
        if spacing_path != spacing:
            raise ValueError(
                f"the pixels of {path.name} are {spacing_path} mm apart, "
                f"but those of {paths[0].name} are {spacing} mm apart."
            )
        params = _register(intensity, corners, spacing)
        column, row = _pixel(params, spacing, x_grid, y_grid)
        sampled = scipy.ndimage.map_coordinates(
            input=height,
            coordinates=[row, column],
            order=1,
            mode="constant",
            cval=np.nan,
        )
        sampled[~inside] = np.nan
        maps.append(_remove(sampled, x_grid, y_grid))

    maps = np.stack(maps)
    measured = np.all(np.isfinite(maps), axis=0)
    mean = np.where(measured, np.mean(maps, axis=0), np.nan)
    return x, y, _remove(mean, x_grid, y_grid)


def _text(value: bytes | str | float | np.generic) -> str:
    """Return an attribute saved by 4Sight as text."""
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip()


def _height(
    path: pathlib.Path,
    width_reference: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read one map, and make it into the height of the surface.

    Parameters
    ----------
    path
        The file saved by 4Sight.
    width_reference
        :data:`_width_reference`, in millimeters.

    Returns
    -------
    height
        The height of the surface on each pixel, in nanometers, NaN where it
        was masked.
    intensity
        The interferogram.
    spacing
        The size of the pixels, in millimeters.

    Raises
    ------
    ValueError
        If any pixel of the reference was masked.
    """
    with h5py.File(path, "r") as file:
        measurement = file["measurement0"]
        group = measurement["genraw"]
        group_reference = measurement["analyzed/opts/refdata"]
        data = group["data"][...].astype(float)
        reference = group_reference["data"][...].astype(float)
        intensity = measurement["reserve_interferogram/frame1/data"][...]
        wavelength = u.Quantity(_text(group.attrs["wavelength"]))
        wavelength_reference = u.Quantity(_text(group_reference.attrs["wavelength"]))
        wedge = float(_text(group.attrs["wedge"]))
        wedge_reference = float(_text(group_reference.attrs["wedge"]))
        spacing = u.Quantity(_text(group.attrs["xpix"])).to_value(u.mm)

    # Masked pixels hold the largest 32-bit number. The data are in waves,
    # multiplied by the wedge, so dividing by it gives the difference in the
    # optical path which the interferometer measured.
    masked = np.finfo(np.float32).max
    data = np.where(data == masked, np.nan, data)
    if np.any(reference == masked):
        raise ValueError(f"the reference saved in {path.name} is masked.")
    path_difference = data / wedge * wavelength.to_value(u.nm)
    path_reference = reference / wedge_reference * wavelength_reference.to_value(u.nm)

    path_reference = scipy.ndimage.gaussian_filter(
        input=path_reference,
        sigma=width_reference / spacing,
    )

    # The light crossed the gap above the surface twice.
    height = (path_difference - path_reference) / 2

    return height, intensity, spacing


def _axis(
    coordinate: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """
    Return the coordinates of a grid which covers the corners.

    The grid is on the multiples of its spacing.

    Parameters
    ----------
    coordinate
        One coordinate of each corner of the clear aperture.
    spacing
        The spacing of the grid.
    """
    start = np.floor(coordinate.min() / spacing)
    stop = np.ceil(coordinate.max() / spacing)
    return np.arange(start, stop + 1) * spacing


def _distance(
    corners: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    How far each position is outside a convex polygon, negative inside it.

    It is exact within the polygon, and near its edges outside it.

    Parameters
    ----------
    corners
        The corners of the polygon, in order around it.
    x
        The :math:`x` coordinate of each position.
    y
        The :math:`y` coordinate of each position.
    """
    center = corners.mean(axis=0)
    result = np.full(np.shape(x), -np.inf)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        tangent = (b - a) / np.linalg.norm(b - a)
        normal = np.array([tangent[1], -tangent[0]])
        if np.dot(center - a, normal) > 0:
            normal = -normal
        result = np.maximum(result, (x - a[0]) * normal[0] + (y - a[1]) * normal[1])
    return result


def _edges(corners: np.ndarray) -> np.ndarray:
    """
    Points spread along each edge of a polygon, short of its corners.

    The corners are left out because there the edges of the trapezoid in the
    interferogram are rounded off.

    Parameters
    ----------
    corners
        The corners of the polygon, in order around it.
    """
    t = (np.arange(_num_edge)[:, np.newaxis] + 0.5) / _num_edge
    t = 0.05 + 0.9 * t
    result = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        result.append(a + t * (b - a))
    return np.concatenate(result)


def _pixel(
    params: np.ndarray,
    spacing: float,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the column and the row of the pixel at each position on the grating.

    Parameters
    ----------
    params
        The column and the row of the origin of the grating, and the angle,
        in radians, by which the grating is turned in the interferogram.
    spacing
        The size of the pixels.
    x
        The :math:`x` coordinate of each position, along the rows.
    y
        The :math:`y` coordinate of each position, along the columns.
    """
    column, row, angle = params
    cos, sin = np.cos(angle), np.sin(angle)
    return (
        column + (sin * x + cos * y) / spacing,
        row + (cos * x - sin * y) / spacing,
    )


def _gradient(intensity: np.ndarray, sigma: float) -> np.ndarray:
    """
    Return the magnitude of the gradient of the smoothed interferogram.

    The edges of the clear aperture are where it is largest, whichever side
    of them is brighter.

    Parameters
    ----------
    intensity
        The interferogram.
    sigma
        The width of the Gaussian which smooths it, in pixels.
    """
    result = scipy.ndimage.gaussian_gradient_magnitude(
        input=intensity.astype(float),
        sigma=sigma,
    )
    return result / result.mean()


def _register(
    intensity: np.ndarray,
    corners: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """
    Find the clear aperture in an interferogram.

    The outline of the clear aperture is first correlated with the edges of
    the interferogram at each whole degree within :data:`_angle_search`, and
    the best match is then refined by maximizing the mean of the edges along
    the outline, on the interferogram smoothed less and less.

    Parameters
    ----------
    intensity
        The interferogram.
    corners
        The corners of the clear aperture, in order around it.
    spacing
        The size of the pixels.

    Returns
    -------
    The column and the row of the origin of the grating, and the angle, in
    radians, by which the grating is turned.

    Raises
    ------
    RuntimeError
        If the fit does not converge.
    """
    points = _edges(corners)

    gradient = _gradient(intensity, _sigma_fit[0])
    spectrum = np.fft.rfft2(gradient)
    limit = _angle_search.to_value(u.deg)
    best = -np.inf
    params = np.zeros(3)
    for angle in np.deg2rad(np.arange(-limit, limit + 0.5, 1)):
        column, row = _pixel((0, 0, angle), spacing, points[:, 0], points[:, 1])
        outline = np.zeros_like(gradient)
        index = (
            np.round(row).astype(int) % gradient.shape[0],
            np.round(column).astype(int) % gradient.shape[1],
        )
        np.add.at(outline, index, 1)
        correlation = np.fft.irfft2(
            spectrum * np.conj(np.fft.rfft2(outline)),
            s=gradient.shape,
        )
        index_max = np.unravel_index(np.argmax(correlation), correlation.shape)
        if correlation[index_max] > best:
            best = correlation[index_max]
            params = np.array([index_max[1], index_max[0], angle], dtype=float)

    for sigma in _sigma_fit:
        # The edges are interpolated with cubic splines, since linear
        # interpolation peaks only on the pixels themselves, which pulls an
        # edge along a row or a column of the pixels onto one of them.
        coefficients = scipy.ndimage.spline_filter(
            input=_gradient(intensity, sigma),
            order=3,
            mode="nearest",
        )

        def merit(p: np.ndarray, coefficients: np.ndarray = coefficients) -> float:
            column, row = _pixel(p, spacing, points[:, 0], points[:, 1])
            edge = scipy.ndimage.map_coordinates(
                input=coefficients,
                coordinates=[row, column],
                order=3,
                mode="nearest",
                prefilter=False,
            )
            return -edge.mean()

        step = np.array([2, 2, np.deg2rad(0.5)])
        fit = scipy.optimize.minimize(
            fun=merit,
            x0=params,
            method="Nelder-Mead",
            options={
                "initial_simplex": np.vstack([params, params + np.diag(step)]),
                "xatol": 1e-4,
                "fatol": 1e-9,
                "maxiter": 4000,
            },
        )
        if not fit.success:
            raise RuntimeError(f"the clear aperture was not found: {fit.message}")
        params = fit.x

    return params


def _remove(
    height: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Remove piston, tilt, and power from a map, fitted where it was measured.

    Parameters
    ----------
    height
        The map, NaN where it was not measured.
    x
        The :math:`x` coordinate of each pixel.
    y
        The :math:`y` coordinate of each pixel.
    """
    measured = np.isfinite(height)
    basis = np.stack([np.ones_like(x), x, y, x**2 + y**2], axis=-1)
    coefficients, *_ = np.linalg.lstsq(basis[measured], height[measured], rcond=None)
    return height - basis @ coefficients

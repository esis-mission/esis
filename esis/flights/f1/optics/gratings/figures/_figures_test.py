import pytest
import numpy as np
import scipy.ndimage
import astropy.units as u
import named_arrays as na
import optika
import esis
from . import _figures


def _aperture() -> optika.apertures.AbstractPolygonalAperture:
    """Return the clear aperture of the gratings, as designed."""
    return esis.flights.f1.optics.design(num_distribution=0).grating.surface.aperture


def _corners() -> np.ndarray:
    """Return the corners of the clear aperture of the gratings, in millimeters."""
    aperture = _aperture()
    vertices = aperture.transformation(aperture.vertices)
    vertices = vertices.broadcast_to(vertices.shape)
    return np.stack(
        [
            vertices.x.ndarray.to_value(u.mm),
            vertices.y.ndarray.to_value(u.mm),
        ],
        axis=-1,
    )


def test_figure_measurement() -> None:
    """
    The map of each grating is the height of its surface inside its aperture.

    The map is on a grid in the coordinates of the grating. It was measured
    over most of the clear aperture, less the border at its edge, and piston,
    tilt, and power are removed from it. What is left is a nanometer or two.
    """
    numbers = na.ScalarArray(
        ndarray=np.array(["UBO-16-024", "UBO-16-017", "UBO-16-019", "UBO-16-014"]),
        axes="c",
    )
    figure = _figures.figure_measurement(
        manufacturing_number=numbers,
        aperture=_aperture(),
        axis_x="a",
        axis_y="b",
    )
    assert isinstance(figure, na.FunctionArray)

    position = figure.inputs
    height = figure.outputs
    assert isinstance(position, na.Cartesian2dVectorArray)
    assert height.axes == ("c", "a", "b")
    assert height.unit == u.nm

    spacing = 0.03760814120043994700 * u.mm
    for component, axis in [(position.x, "a"), (position.y, "b")]:
        assert component.axes == (axis,)
        step = component[{axis: slice(1, None)}] - component[{axis: slice(None, -1)}]
        assert np.all(np.isclose(step, spacing, rtol=1e-12, atol=0 * u.mm))

    corners = _corners()
    x = position.x.ndarray.to_value(u.mm)[:, np.newaxis]
    y = position.y.ndarray.to_value(u.mm)[np.newaxis, :]
    distance = _figures._distance(corners, x, y)
    # the grid covers the corners of the clear aperture, and no more
    assert x.min() <= corners[:, 0].min() < x.min() + spacing.to_value(u.mm)
    assert x.max() - spacing.to_value(u.mm) < corners[:, 0].max() <= x.max()

    inside = distance <= -_figures._width_edge.to_value(u.mm)
    measured = np.isfinite(height.ndarray)
    assert np.all(measured <= inside)
    fraction = measured.sum(axis=(1, 2)) / inside.sum()
    assert np.all(fraction > 0.75)

    x_grid, y_grid = np.broadcast_arrays(x, y)
    for h in height.ndarray.to_value(u.nm):
        basis = np.stack([np.ones_like(x_grid), x_grid, y_grid, x_grid**2 + y_grid**2])
        coefficients = [np.nanmean(b * h) for b in basis]
        assert np.allclose(coefficients, 0, atol=1e-9)

    rms = np.sqrt(np.nanmean(np.square(height.ndarray.to_value(u.nm)), axis=(1, 2)))
    assert np.all((1 < rms) & (rms < 2.5))


def test_figure_measurement_axes() -> None:
    """The maps have the axes of the numbers of the gratings they are maps of."""
    numbers = na.ScalarArray(
        ndarray=np.array([["UBO-16-019", "UBO-16-014"]]),
        axes=("d", "c"),
    )
    figure = _figures.figure_measurement(numbers, _aperture())
    assert figure.outputs.axes == ("d", "c", "grating_x", "grating_y")
    assert figure.outputs.shape == dict(d=1, c=2, grating_x=452, grating_y=459)

    each = _figures.figure_measurement(
        na.ScalarArray(np.array(["UBO-16-014", "UBO-16-019"]), axes="c"),
        _aperture(),
    )
    flipped = each.outputs[dict(c=slice(None, None, -1))].add_axes("d")
    assert np.all(np.isfinite(figure.outputs) == np.isfinite(flipped))
    assert np.all(np.nan_to_num(figure.outputs) == np.nan_to_num(flipped))


def test_figure_measurement_unknown() -> None:
    """A grating which was not measured has no map."""
    with pytest.raises(ValueError, match="UBO-16-099"):
        _figures.figure_measurement(
            na.ScalarArray(np.array(["UBO-16-099"]), axes="c"),
            _aperture(),
        )


def test_figure_measurement_aperture() -> None:
    """The clear aperture must be the same for every grating."""
    aperture = _aperture()
    aperture.x_left = na.ScalarArray([1, 2] * u.mm, axes="c")
    with pytest.raises(ValueError, match="same for every grating"):
        _figures.figure_measurement(
            na.ScalarArray(np.array(["UBO-16-014", "UBO-16-019"]), axes="c"),
            aperture,
        )


def test_distance() -> None:
    """The distance outside a square, negative inside it."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    x = np.array([0, 0.5, 1, 1.5, 0, -3])
    y = np.array([0, 0, 0, 0, -1.25, 0.5])
    result = _figures._distance(corners, x, y)
    assert np.allclose(result, [-1, -0.5, 0, 0.5, 0.25, 2], rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    argnames="params",
    argvalues=[
        (500.0, 480.0, 0.0),
        (463.25, 517.75, np.deg2rad(2.3)),
        (540.5, 455.5, np.deg2rad(-1.4)),
    ],
)
def test_register(params: tuple[float, float, float]) -> None:
    """
    The clear aperture is found in a synthetic interferogram.

    There it is a bright trapezoid on a speckled background, and it is found
    to within a tenth of a pixel and a hundredth of a degree.
    """
    corners = _corners()
    spacing = 0.0376
    shape = (1004, 999)
    row, column = np.indices(shape)

    # the position of each pixel on the grating, inverting _figures._pixel
    column0, row0, angle = params
    dc = (column - column0) * spacing
    dr = (row - row0) * spacing
    cos, sin = np.cos(angle), np.sin(angle)
    x = cos * dr + sin * dc
    y = -sin * dr + cos * dc
    distance = _figures._distance(corners, x, y)

    rng = np.random.default_rng(42)
    face = 0.5 * (1 - np.tanh(distance / (2 * spacing)))
    intensity = 250 + 700 * face + 50 * rng.standard_normal(shape)
    intensity = scipy.ndimage.gaussian_filter(intensity, 1)

    result = _figures._register(intensity, corners, spacing)
    assert np.allclose(result[:2], params[:2], rtol=0, atol=0.1)
    assert np.isclose(result[2], params[2], rtol=0, atol=np.deg2rad(0.01))

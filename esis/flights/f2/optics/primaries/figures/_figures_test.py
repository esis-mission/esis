import pathlib
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis


@pytest.mark.parametrize("serial_number", ["SN2", "SN3"])
def test_figure_measurement(serial_number: str) -> None:
    """
    The map is the height of the surface of a primary mirror.

    It is on a grid of square pixels about the center of the map. Turned by
    22.5 degrees, it lies inside the clear aperture of the design, and it was
    measured everywhere inside it, center included, except within three
    pixels of its edge.
    """
    figures = esis.flights.f2.optics.primaries.figures
    figure = figures.figure_measurement(serial_number, axis_x="a", axis_y="b")
    assert isinstance(figure, na.FunctionArray)

    position = figure.inputs
    height = figure.outputs
    assert isinstance(position, na.Cartesian2dVectorArray)
    assert position.x.shape == dict(a=576)
    assert position.y.shape == dict(b=576)
    assert height.shape == dict(b=576, a=576)
    assert height.unit == u.nm

    spacing = 0.25 * u.mm
    for component, axis in [(position.x, "a"), (position.y, "b")]:
        step = component[{axis: slice(1, None)}] - component[{axis: slice(None, -1)}]
        assert np.all(np.isclose(step, spacing, rtol=1e-7, atol=0 * u.mm))
        assert np.isclose(component.mean(), 0 * u.mm, rtol=0, atol=1e-9 * u.mm)

    angle = 22.5 * u.deg
    turned = na.Cartesian3dVectorArray(
        x=position.x * np.cos(angle) - position.y * np.sin(angle),
        y=position.x * np.sin(angle) + position.y * np.cos(angle),
        z=0 * u.mm,
    )
    primary = esis.flights.f2.optics.design(num_distribution=0).primary_mirror
    clear = primary.surface.aperture(turned)
    measured = np.isfinite(height)
    assert not np.any(measured & ~clear)
    unmeasured = clear & ~measured
    assert np.any(unmeasured)

    # How far each pixel is inside the edge of the clear aperture. On the grid
    # of the map, its flats are along x, y, and the diagonals.
    x, y = position.x, position.y
    reach = np.maximum(
        np.maximum(np.abs(x), np.abs(y)),
        np.maximum(np.abs(x + y), np.abs(x - y)) / np.sqrt(2),
    )
    depth = primary.width_clear / 2 - reach
    assert np.all(np.where(unmeasured, depth, 0 * u.mm) < 3 * spacing)

    rms = np.sqrt(np.nanmean(np.square(height)))
    assert 2.5 * u.nm < rms < 3 * u.nm


def test_figure_measurement_orientation() -> None:
    """The first row of the map in the file is at the top, where y is largest."""
    figures = esis.flights.f2.optics.primaries.figures
    path = figures._figures._path_directory() / "primary/SN2_3219CC ASP_IRR_UC.dat"
    data, _ = figures._figures._read_metropro(path)

    figure = figures.figure_measurement("SN2", axis_x="a", axis_y="b")
    assert figure.outputs.axes == ("b", "a")
    assert np.all(figure.inputs.x[dict(a=slice(1, None))] > figure.inputs.x[dict(a=0)])
    assert np.all(figure.inputs.y[dict(b=slice(1, None))] > figure.inputs.y[dict(b=0)])
    assert np.array_equal(
        figure.outputs.ndarray.to_value(u.nm),
        data[::-1].to_value(u.nm),
        equal_nan=True,
    )


def test_figure_measurement_serial_number() -> None:
    """Only the serial numbers of the mirrors which were measured are accepted."""
    with pytest.raises(ValueError, match="SN1"):
        esis.flights.f2.optics.primaries.figures.figure_measurement("SN1")


def test_read_metropro_format(tmp_path: pathlib.Path) -> None:
    """A file whose header is of another format than 1 is refused."""
    figures = esis.flights.f2.optics.primaries.figures
    path = figures._figures._path_directory() / "primary/SN2_3219CC ASP_IRR_UC.dat"
    data = bytearray(path.read_bytes())
    data[4:6] = (3).to_bytes(2, "big")
    path_format = tmp_path / "format_3.dat"
    path_format.write_bytes(data)
    with pytest.raises(ValueError, match="format 1"):
        figures._figures._read_metropro(path_format)

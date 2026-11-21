import pathlib
import pytest
import numpy as np
import astropy.units as u
from esis.flights.f2.optics.primaries import _metrology

_files = [
    "SN2_3219CC ASP_IRR_UC.dat",
    "SN2_00a_MET01_NXV1_1.4X_1XZOOM.dat",
    "SN2_00a_MET01_NXV1_20X_1XZOOM.dat",
    "SN3_3219CC ASP_IRR_UC.dat",
    "SN3_00a_MET01_NXV1_1.4X_1XZOOM.dat",
    "SN3_00a_MET01_NXV1_20X_1XZOOM.dat",
]


@pytest.mark.parametrize("name", _files)
def test_map(name: str) -> None:
    """
    The map is the file's heights, with its first row at the top, where y is largest.

    The positions are on a grid of square pixels about the center of the map.
    The maps look much the same flipped, so a test of what they show could
    not tell which way up they are.
    """
    path = _metrology._path_directory() / "primary" / name
    data, spacing = _metrology._read_metropro(path)
    num_y, num_x = data.shape

    result = _metrology._map(path, axis_x="a", axis_y="b")
    position = result.inputs
    height = result.outputs
    assert height.axes == ("b", "a")
    assert height.unit == u.nm
    assert np.array_equal(
        height.ndarray.to_value(u.nm),
        data[::-1].to_value(u.nm),
        equal_nan=True,
    )

    assert position.x.shape == dict(a=num_x)
    assert position.y.shape == dict(b=num_y)
    for component, axis in [(position.x, "a"), (position.y, "b")]:
        step = component[{axis: slice(1, None)}] - component[{axis: slice(None, -1)}]
        assert np.all(np.isclose(step, spacing, rtol=1e-7, atol=0 * u.mm))
        assert np.isclose(component.mean(), 0 * u.mm, rtol=0, atol=1e-9 * u.mm)


def test_read_metropro_format(tmp_path: pathlib.Path) -> None:
    """A file whose header is of another format than 1 is refused."""
    path = _metrology._path_directory() / "primary" / _files[0]
    data = bytearray(path.read_bytes())
    data[4:6] = (3).to_bytes(2, "big")
    path_format = tmp_path / "format_3.dat"
    path_format.write_bytes(data)
    with pytest.raises(ValueError, match="format 1"):
        _metrology._read_metropro(path_format)

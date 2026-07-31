import pytest
import astropy.units as u
import named_arrays as na
import esis


def test_aia_context():
    try:
        result = esis.flights.f1.data.aia_context(
            wavelength=[304] * u.AA,
            limit=1,
        )
    except OSError:
        pytest.skip("JSOC is unreachable, skipping live-network test")

    assert isinstance(result, dict)
    assert "AIA 304" in result

    function = result["AIA 304"]
    assert isinstance(function, na.FunctionArray)

    x = function.inputs.position.x
    y = function.inputs.position.y
    assert len(x.axes) == 1
    assert len(y.axes) == 1
    assert x.shape[x.axes[0]] == function.outputs.shape[x.axes[0]] + 1
    assert y.shape[y.axes[0]] == function.outputs.shape[y.axes[0]] + 1

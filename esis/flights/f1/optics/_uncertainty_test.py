import pytest
import astropy.units as u
import named_arrays as na
from esis.flights.f1.optics._uncertainty import uniform


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_uniform(num_distribution: int):
    nominal = 5 * u.mm
    result = uniform(nominal=nominal, width=1 * u.mm, num_distribution=num_distribution)
    if num_distribution == 0:
        assert result is nominal
    else:
        assert isinstance(result, na.AbstractUncertainScalarArray)
        assert result.nominal == nominal
        assert result.num_distribution == num_distribution


def test_uniform_width() -> None:
    """The samples span the nominal value plus or minus the width."""
    nominal = 5 * u.mm
    width = 1 * u.mm
    result = uniform(nominal=nominal, width=width, num_distribution=101)
    distribution = result.distribution
    assert distribution.min() >= nominal - width
    assert distribution.max() <= nominal + width
    # a full width would keep every sample within half of it of the nominal
    # value, so the samples could span at most the width
    assert distribution.max() - distribution.min() > width

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

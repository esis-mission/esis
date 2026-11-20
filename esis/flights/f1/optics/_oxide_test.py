import pytest
import astropy.units as u
import astropy.time
from esis.flights.f1.optics import _oxide

time_coating = astropy.time.Time("2018-01-01")


def test_rate_growth() -> None:
    """The rate is a fraction of a nanometer per decade, as for a native oxide."""
    result = _oxide.rate_growth()
    assert result.unit.is_equivalent(u.nm)
    assert 0.1 * u.nm < result < 0.3 * u.nm


def test_thickness_growth_decade() -> None:
    """One decade of time since the coating grows one rate of oxide."""
    result = _oxide.thickness_growth(
        time_coating=time_coating,
        time_start=time_coating + 3 * u.day,
        time_stop=time_coating + 30 * u.day,
    )
    assert u.isclose(result, _oxide.rate_growth())


def test_thickness_growth_zero() -> None:
    """No oxide grows in an empty interval."""
    time = time_coating + 1 * u.day
    result = _oxide.thickness_growth(time_coating, time, time)
    assert result == 0 * u.nm


def test_thickness_growth_additive() -> None:
    """The growth over an interval is the sum of the growth over its parts."""
    times = [time_coating + t for t in (1, 40, 600) * u.day]
    first = _oxide.thickness_growth(time_coating, times[0], times[1])
    second = _oxide.thickness_growth(time_coating, times[1], times[2])
    whole = _oxide.thickness_growth(time_coating, times[0], times[2])
    assert u.isclose(first + second, whole)


@pytest.mark.parametrize(
    argnames="time_start,time_stop",
    argvalues=[
        (time_coating, time_coating + 1 * u.day),
        (time_coating + 2 * u.day, time_coating + 1 * u.day),
    ],
)
def test_thickness_growth_order(
    time_start: astropy.time.Time,
    time_stop: astropy.time.Time,
) -> None:
    """The interval must start after the coating and stop after it starts."""
    with pytest.raises(ValueError):
        _oxide.thickness_growth(time_coating, time_start, time_stop)

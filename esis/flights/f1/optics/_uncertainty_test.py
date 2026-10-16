import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from esis.flights.f1.optics._uncertainty import spawn_seeds, uniform


def test_spawn_seeds() -> None:
    """The same seed spawns the same seeds, which are all different."""
    seeds = spawn_seeds(0)
    result = [next(seeds) for _ in range(10)]

    seeds_again = spawn_seeds(0)
    assert [next(seeds_again) for _ in range(10)] == result

    assert len(set(result)) == len(result)

    seeds_other = spawn_seeds(1)
    assert set(next(seeds_other) for _ in range(10)).isdisjoint(result)


def test_spawn_seeds_none() -> None:
    """Without a seed, the seeds are different every time."""
    assert next(spawn_seeds(None)) != next(spawn_seeds(None))


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_uniform(num_distribution: int):
    nominal = 5 * u.mm
    result = uniform(
        nominal=nominal,
        width=1 * u.mm,
        num_distribution=num_distribution,
        seed=0,
    )
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
    result = uniform(nominal=nominal, width=width, num_distribution=101, seed=0)
    distribution = result.distribution
    assert distribution.min() >= nominal - width
    assert distribution.max() <= nominal + width
    # a full width would keep every sample within half of it of the nominal
    # value, so the samples could span at most the width
    assert distribution.max() - distribution.min() > width


def test_uniform_seed() -> None:
    """The same seed draws the same samples, and another seed draws others."""
    nominal = 5 * u.mm
    width = 1 * u.mm
    result = uniform(nominal=nominal, width=width, num_distribution=11, seed=0)
    again = uniform(nominal=nominal, width=width, num_distribution=11, seed=0)
    other = uniform(nominal=nominal, width=width, num_distribution=11, seed=1)
    assert np.all(again == result)
    assert np.all(other.distribution != result.distribution)

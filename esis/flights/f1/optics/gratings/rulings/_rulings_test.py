import collections.abc
import pytest
import numpy as np
import astropy.units as u
import optika
import esis


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_ruling_design(num_distribution: int) -> None:
    r = esis.flights.f1.optics.gratings.rulings.ruling_design(
        num_distribution=num_distribution,
    )
    assert isinstance(r, optika.rulings.AbstractRulings)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_ruling_measurement(num_distribution: int) -> None:
    r = esis.flights.f1.optics.gratings.rulings.ruling_measurement(
        num_distribution=num_distribution,
    )
    assert isinstance(r, optika.rulings.AbstractRulings)


@pytest.mark.parametrize(
    argnames="ruling",
    argvalues=[
        esis.flights.f1.optics.gratings.rulings.ruling_design,
        esis.flights.f1.optics.gratings.rulings.ruling_measurement,
    ],
)
def test_width_density(
    ruling: collections.abc.Callable[..., optika.rulings.AbstractRulings],
) -> None:
    """The tolerance on the spacing is that of 1 per millimeter on the density."""
    spacing = ruling(num_distribution=11).spacing.coefficients[0]
    density = 1 / spacing.nominal
    width_density = spacing.width * np.square(density)
    assert np.isclose(width_density, 1 / u.mm)

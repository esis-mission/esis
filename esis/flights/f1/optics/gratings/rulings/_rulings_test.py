import pytest
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

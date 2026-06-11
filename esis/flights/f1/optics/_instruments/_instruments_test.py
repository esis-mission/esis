import pytest
import numpy as np
import named_arrays as na
import esis


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_full(num_distribution: int):
    result = esis.flights.f1.optics.design_full(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.f1.optics.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.f1.optics.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built(num_distribution: int):
    result = esis.flights.f1.optics.as_built(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_distortion_fit(num_distribution: int):
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


def test_distortion_fit_bounds():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)

    assert isinstance(lower, esis.optics.DistortionParameters)
    assert isinstance(upper, esis.optics.DistortionParameters)

    x = na.pack(parameters).ndarray
    x_lower = na.pack(lower).ndarray
    x_upper = na.pack(upper).ndarray

    assert x_lower.shape == x.shape
    assert x_upper.shape == x.shape
    assert np.all(x_lower <= x)
    assert np.all(x <= x_upper)

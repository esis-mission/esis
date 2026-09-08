import numpy as np
import named_arrays as na
import esis
from . import _fit


def _quadratic(x: np.ndarray) -> float:
    return float(np.sum(np.square(x - 0.3)))


def test_polish():
    x0 = np.zeros(3)
    x, fun, num = _fit.polish(
        _quadratic, x0, -np.ones(3), np.ones(3), scale=0.1, num_round=2
    )
    assert np.allclose(x, 0.3, atol=1e-2)
    assert fun < 1e-3
    assert num > 0


def test_polish_start_on_bound():
    # the design start sits on the upper bound of two parameters; the simplex
    # must step into the interior there rather than be clipped onto the start
    x0 = np.array([1.0, 1.0, 0.0])
    x, fun, _ = _fit.polish(
        _quadratic, x0, -np.ones(3), np.ones(3), scale=0.1, num_round=2
    )
    assert np.allclose(x, 0.3, atol=1e-2)


def test_fit_distortion(tmp_path):
    instrument = esis.flights.f1.optics.design(num_distribution=0)[dict(channel=1)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)
    target = na.pack(parameters).ndarray + 0.1 * (
        na.pack(upper).ndarray - na.pack(lower).ndarray
    )

    def objective(x):
        return float(
            np.mean(
                np.square(
                    (x - target) / (na.pack(upper).ndarray - na.pack(lower).ndarray)
                )
            )
        )

    messages = []
    result = _fit.fit_distortion(
        objective=objective,
        parameters=parameters,
        bounds=(lower, upper),
        popsize=2,
        maxiter=3,
        log=messages.append,
    )
    assert isinstance(result, esis.optics.DistortionParameters)
    assert objective(na.pack(result).ndarray) < objective(na.pack(parameters).ndarray)
    assert any("capture" in m for m in messages)
    assert any("polish" in m for m in messages)

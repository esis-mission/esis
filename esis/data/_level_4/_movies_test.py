import functools
import pytest
import numpy as np
import matplotlib.animation
import astropy.units as u
import named_arrays as na
import esis

ctis = pytest.importorskip("ctis")

if not hasattr(ctis.instruments, "OptikaInstrument"):  # pragma: nocover
    pytest.skip(
        reason="ctis is missing OptikaInstrument (needs ctis PR #18)",
        allow_module_level=True,
    )


@functools.cache
def _level_4() -> esis.data.Level_4:
    """Load the coarse-grid flight product used by the loader tests."""
    return esis.flights.f1.data.level_4(
        pitch_scene=16 * u.arcsec,
        pitch_velocity=200 * u.km / u.s,
        num_iteration=2,
    )


def _context() -> dict[str, na.FunctionArray]:
    """Build a synthetic context-image function on the scene coordinates."""
    a = _level_4()
    x = a.inputs.position.x
    y = a.inputs.position.y
    num_x = x.shape[a.axis_x] - 1
    num_y = y.shape[a.axis_y] - 1
    num_time = a.shape[a.axis_time]
    outputs = na.ScalarArray(
        ndarray=np.random.default_rng(42).random((num_time, num_x, num_y)),
        axes=(a.axis_time, a.axis_x, a.axis_y),
    )
    return {
        "synthetic": na.FunctionArray(
            inputs=na.TemporalPositionalVectorArray(
                time=a.inputs.time,
                position=na.Cartesian2dVectorArray(x=x, y=y),
            ),
            outputs=outputs,
        ),
    }


def test_locate_event():
    a = _level_4()
    position = a.locate_event()
    assert position.shape == (2,)
    assert position.unit.is_equivalent(u.arcsec)
    x = a.inputs.position.x.ndarray
    y = a.inputs.position.y.ndarray
    assert x.min() <= position[0] <= x.max()
    assert y.min() <= position[1] <= y.max()


def test_animate_event():
    a = _level_4()
    position = a.locate_event()
    result = a.animate_event(
        position=position,
        halfwidth=64 * u.arcsec,
        context=_context(),
    )
    assert isinstance(result, matplotlib.animation.FuncAnimation)

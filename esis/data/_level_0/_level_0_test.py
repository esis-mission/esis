import pytest
import numpy as np
import astropy.units as u
import astropy.time
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis
from ..abc._channel_data_test import AbstractTestAbstractChannelData

_level_0 = esis.flights.f1.data.level_0()


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        _level_0[dict(time=slice(None, None, 8))],
    ],
)
class TestLevel_0(
    AbstractTestAbstractSensorData,
    AbstractTestAbstractChannelData,
):
    def test_timeline(self, a: esis.data.Level_0):
        result = a.timeline
        if result is not None:
            assert isinstance(result, esis.nsroc.Timeline)

    def test_despiked(self, a: esis.data.Level_0):
        a = a[{a.axis_time: slice(0, 1)}]
        super().test_despiked(a)

    def test_time_mission_start(self, a: esis.data.Level_0):
        result = a.time_mission_start
        assert isinstance(result, astropy.time.Time)
        assert result < a.inputs.time.ndarray.min()

    def test_lights(self, a: esis.data.Level_0):
        if a.axis_time in a.shape:
            result = a.lights
            assert isinstance(result, type(a))

    def test_darks(self, a: esis.data.Level_0):
        if a.axis_time in a.shape:
            result = a.darks
            assert isinstance(result, type(a))

    def test_dark(self, a: esis.data.Level_0):
        index = {
            a.axis_x: slice(0, 100),
            a.axis_y: slice(0, 100),
        }
        result = a[index].dark
        assert isinstance(result, type(a))
        assert np.all(result.outputs.std((a.axis_x, a.axis_y)) < 10 * u.DN)
        assert result.outputs.shape[a.axis_time] == 1

    def test_dark_subtracted(self, a: esis.data.Level_0):
        index = {
            a.axis_x: slice(0, 100),
            a.axis_y: slice(0, 100),
        }
        result = a[index].dark_subtracted
        assert isinstance(result, type(a))
        assert np.all(result.darks.outputs.mean() < 1 * u.DN)


def _sequence_numbers(a: esis.data.Level_0) -> np.ndarray:
    """
    Return the sequence numbers of the images in `a`.

    Parameters
    ----------
    a
        The images.
    """
    return a.inputs.sequence_number[{a.axis_channel: 0}].ndarray


@pytest.mark.parametrize(
    argnames="selection,expected",
    argvalues=[
        ("darks_up", np.arange(98, 104)),
        ("lights", np.arange(109, 139)),
        ("darks_down", np.arange(153, 167)),
    ],
)
def test_selection_whole_flight(selection: str, expected: np.ndarray) -> None:
    """
    Check which images of the whole flight are taken as the darks and the lights.

    The frames of :func:`esis.flights.f1.data.level_1`, and the fits indexed
    by them, depend on these.

    Parameters
    ----------
    selection
        The name of the property which selects the images.
    expected
        The sequence numbers of the images it should select.
    """
    result = _sequence_numbers(getattr(_level_0, selection))
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="index",
    argvalues=[
        slice(1, None),
        slice(10, None),
        slice(20, 60),
        slice(None, 50),
        slice(None, 30),
        slice(None, None, 8),
    ],
)
@pytest.mark.parametrize(
    argnames="selection",
    argvalues=[
        "darks_up",
        "lights",
        "darks_down",
    ],
)
def test_selection_subset(index: slice, selection: str) -> None:
    """
    Check that a subset of the images selects the same images as the whole flight.

    Parameters
    ----------
    index
        The images of the flight in the subset.
    selection
        The name of the property which selects the images.
    """
    a = _level_0
    b = a[{a.axis_time: index}]
    result = _sequence_numbers(getattr(b, selection))
    expected = np.intersect1d(
        _sequence_numbers(getattr(a, selection)),
        _sequence_numbers(b),
    )
    assert np.array_equal(result, expected)

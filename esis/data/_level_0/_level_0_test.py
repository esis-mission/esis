import pytest
import numpy as np
import astropy.units as u
import astropy.time
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis
from ..abc._channel_data_test import AbstractTestAbstractChannelData


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.data.level_0(),
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

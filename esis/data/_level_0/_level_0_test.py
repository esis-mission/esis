import pytest
import astropy.time
import named_arrays as na
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.data.level_0(),
    ],
)
class TestLevel_0(
    AbstractTestAbstractSensorData,
):
    def test_timeline(self, a: esis.data.Level_0):
        result = a.timeline
        if result is not None:
            assert isinstance(result, esis.nsroc.Timeline)

    def test_channel(self, a: esis.data.Level_0):
        result = a.channel
        assert isinstance(result, na.ScalarArray)

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

import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import msfc_ccd
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis

_timeline = esis.nsroc.Timeline(
    timedelta_esis_start=1 * u.s,
    timedelta_sparcs_rlg_enable=60 * u.s,
    timedelta_sparcs_rlg_disable=120 * u.s,
)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.data.Level_0.from_fits(
            path=msfc_ccd.samples.path_dark_esis1,
            camera=msfc_ccd.Camera(),
            timeline=_timeline,
        ),
        esis.data.Level_0.from_fits(
            path=na.ScalarArray(
                ndarray=np.array(
                    [
                        msfc_ccd.samples.path_dark_esis1,
                        msfc_ccd.samples.path_dark_esis3,
                    ]
                ),
                axes="channel",
            ),
            camera=msfc_ccd.Camera(),
            timeline=_timeline,
        ),
        esis.data.Level_0.from_fits(
            path=na.ScalarArray(
                ndarray=np.array(
                    [
                        [
                            msfc_ccd.samples.path_dark_esis1,
                            msfc_ccd.samples.path_dark_esis3,
                        ],
                        [
                            msfc_ccd.samples.path_fe55_esis1,
                            msfc_ccd.samples.path_fe55_esis3,
                        ]
                    ]
                ),
                axes=("time", "channel"),
            ),
            camera=msfc_ccd.Camera(),
            timeline=_timeline,
        ),
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

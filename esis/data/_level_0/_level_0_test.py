import pytest
import astropy.units as u
import astropy.time
import named_arrays as na
import msfc_ccd
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.data.Level_0(
            inputs=msfc_ccd.ImageHeader(
                pixel=na.Cartesian2dVectorArrayRange(
                    start=0,
                    stop=10,
                    axis=na.Cartesian2dVectorArray("x", "y"),
                ),
                time=astropy.time.Time.now(),
                timedelta=10 * u.s,
                timedelta_requested=10 * u.s,
            ),
            outputs=na.random.uniform(0, 100, dict(x=10, y=10)),
            axis_x="x",
            axis_y="y",
            camera=esis.optics.Camera(),
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

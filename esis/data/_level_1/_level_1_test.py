import pytest
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.data.level_1(),
    ],
)
class TestLevel_1(
    AbstractTestAbstractSensorData,
):
    pass

import pytest
from msfc_ccd._tests.test_sensors import AbstractTestAbstractSensor
import esis


@pytest.mark.parametrize(
    argnames="sensor",
    argvalues=[
        esis.optics.Sensor(),
    ],
)
class TestSensor(AbstractTestAbstractSensor):
    pass

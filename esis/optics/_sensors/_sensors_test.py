import pytest
from msfc_ccd._tests.test_sensors import AbstractTestAbstractSensor
from optika._tests import test_mixins
import esis
from ..mixins import _mixins_test


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Sensor(),
    ],
)
class TestSensor(
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
    test_mixins.AbstractTestTranslatable,
    _mixins_test.AbstractTestCylindricallyTransformable,
    AbstractTestAbstractSensor,
):
    pass

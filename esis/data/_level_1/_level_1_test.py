import pytest
from msfc_ccd._images._tests.test_images import AbstractTestAbstractImageData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.data.level_1(),
    ],
)
class TestLevel_1(
    AbstractTestAbstractImageData,
):
    pass

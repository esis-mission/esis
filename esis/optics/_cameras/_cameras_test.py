import pytest
import optika
from msfc_ccd._tests.test_cameras import AbstractTestAbstractSensor
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Camera(),
    ],
)
class TestCameras(
    AbstractTestAbstractSensor,
):
    def test_surface(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.surface, optika.surfaces.AbstractSurface)

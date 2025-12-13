import pytest
import optika
from msfc_ccd._tests.test_cameras import AbstractTestAbstractCamera
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Camera(),
    ],
)
class TestCameras(
    AbstractTestAbstractCamera,
):
    def test_surface(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.surface, optika.surfaces.AbstractSurface)

import pytest
import astropy.units as u
import optika
from msfc_ccd._tests.test_cameras import AbstractTestAbstractCamera
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Camera(),
        esis.optics.Camera(
            sensor=esis.optics.Sensor(readout_noise=6 * u.electron),
        ),
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

    def test_surface_read_noise(
        self,
        a: esis.optics.Camera,
    ):
        result = a.surface.read_noise
        assert result == a.sensor.readout_noise
        assert result > 0 * u.electron

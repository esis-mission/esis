import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
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
    optika._tests.test_mixins.AbstractTestReplaceable,
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


def test_surface_read_noise_tap() -> None:
    """
    Check the read noise of the surface when it varies from tap to tap.

    It reaches the surface as its quadrature mean over the taps of each
    channel.
    """
    camera = esis.optics.Camera(
        sensor=esis.optics.Sensor(
            readout_noise=na.ScalarArray(
                ndarray=[[[1, 2], [3, 4]], [[2, 2], [2, 2]]] * u.electron,
                axes=("channel", "tap_y", "tap_x"),
            ),
        ),
    )

    expected = na.ScalarArray(
        ndarray=np.sqrt([(1 + 4 + 9 + 16) / 4, 4]) * u.electron,
        axes="channel",
    )

    result = camera.surface.read_noise
    assert result.shape == expected.shape
    assert np.allclose(result, expected)

    # and the system the surface belongs to gains no axes from it
    assert camera.surface.shape == {"channel": 2}

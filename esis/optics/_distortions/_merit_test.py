import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from . import _merit


def _channel() -> esis.optics.Instrument:
    from esis.flights.f1.optics._fits._fits import _idealized, _wavelength_lines

    instrument = _idealized(esis.flights.f1.optics.design(num_distribution=0))
    instrument.wavelength = _wavelength_lines()
    return instrument[dict(channel=1)]


def _scene(num: int = 41, seed: int = 0) -> na.FunctionArray:
    """Make a random scene on the vertices of a coarse field grid, three lines."""
    from esis.flights.f1.optics._fits._fits import _wavelength_lines

    wavelength = _wavelength_lines()
    position = na.Cartesian2dVectorLinearSpace(
        start=-400 * u.arcsec,
        stop=400 * u.arcsec,
        axis=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        num=num,
    )
    # a smooth random field, so that a shift of a few pixels is visible
    rng = np.random.default_rng(seed)
    values = rng.uniform(0, 1, size=(3, 1, num - 1, num - 1))
    import scipy.ndimage

    values = scipy.ndimage.gaussian_filter(values, sigma=(0, 0, 1.5, 1.5))
    radiance = u.photon / u.s / u.cm**2 / u.arcsec**2 / u.AA
    return na.FunctionArray(
        inputs=na.SpectralPositionalVectorArray(
            # one velocity cell per line, so two vertices along the velocity axis
            wavelength=wavelength
            + na.ScalarArray(np.array([-0.05, 0.05]) * u.AA, axes="velocity"),
            position=position,
        ),
        outputs=na.ScalarArray(
            values * 1e3 * radiance,
            axes=("wavelength", "velocity", "detector_x", "detector_y"),
        ),
    )


@pytest.fixture(scope="module")
def merit() -> tuple[_merit.LinearMerit, esis.optics.DistortionParameters]:
    instrument = _channel()
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    scene = _scene()
    # the observation is the model's own image of the scene at the truth
    truth = na.unpack(na.pack(parameters).ndarray.copy(), parameters)
    truth.pitch = 4 * u.arcsec
    truth.yaw = -3 * u.arcsec
    probe = _merit.LinearMerit(
        instrument,
        parameters,
        scene,
        observation=0 * scene.outputs.sum(("wavelength", "velocity")),
    )
    observation = probe.image(truth)
    return (
        _merit.LinearMerit(instrument, parameters, scene, observation=observation),
        truth,
    )


class TestLinearMerit:
    def test_truth(self, merit):
        m, truth = merit
        assert m.correlation(truth) > 0.999

    def test_deterministic(self, merit):
        # no sampling noise: repeated evaluations agree to the rounding of a
        # threaded reduction, which is a hundred million times below the
        # differences the fit resolves
        m, truth = merit
        x = na.pack(truth).ndarray
        assert np.isclose(m(x), m(x), rtol=0, atol=1e-12)

    def test_offset_is_worse(self, merit):
        m, truth = merit
        x = na.pack(truth).ndarray
        y = x.copy()
        names = [f.name for f in __import__("dataclasses").fields(truth)]
        y[names.index("yaw")] += 20  # arcsec, a few pixels
        assert m(y) > m(x)

    def test_start_is_worse_than_truth(self, merit):
        m, truth = merit
        assert m.correlation(m.parameters) < m.correlation(truth)

    def test_off_sensor_is_worst(self, merit):
        m, truth = merit
        x = na.pack(truth).ndarray.copy()
        names = [f.name for f in __import__("dataclasses").fields(truth)]
        x[names.index("pitch")] += 3000  # arcsec: off the sensor
        assert m(x) == 1.0
        assert m.num_failed >= 1

    def test_mask(self, merit):
        m, truth = merit
        mask = m.mask_field_stop(truth.to_instrument(m.instrument))
        fraction = float(np.asarray(mask.ndarray).mean())
        assert 0.3 < fraction < 0.9

    def test_linearize_pins_area(self, merit):
        m, truth = merit
        linear = m.linearize(truth.to_instrument(m.instrument))
        area = linear.area_effective.area
        assert np.allclose(na.value(area).ndarray, na.value(area).ndarray[0])


def test_correlation():
    a = na.ScalarArray(np.arange(12.0).reshape(3, 4), axes=("x", "y"))
    assert np.isclose(float(_merit.correlation(a, a).ndarray), 1)
    assert np.isclose(float(_merit.correlation(a, -a).ndarray), -1)
    b = na.ScalarArray(np.ones((3, 4)), axes=("x", "y"))
    assert float(_merit.correlation(a, b).ndarray) == 0


def test_correlation_raytraced(merit):
    m, truth = merit
    pupil = na.Cartesian2dVectorLinearSpace(
        -0.25, 0.25, axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"), num=2
    )
    result = _merit.correlation_raytraced(
        instrument=m.instrument,
        parameters=truth,
        scene=m.scene,
        observation=m.observation,
        pupil=pupil,
        sigma_psf=2.0,
    )
    assert 0 < result <= 1

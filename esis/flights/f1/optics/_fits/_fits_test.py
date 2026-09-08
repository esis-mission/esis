import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis
from . import _fits


def test_idealized():
    instrument = esis.flights.f1.optics.as_built(num_distribution=0)
    result = _fits._idealized(instrument)

    assert isinstance(result.grating.material, optika.materials.Mirror)
    assert isinstance(result.primary_mirror.material, optika.materials.Mirror)
    assert result.filter.material is None
    coefficients = result.grating.rulings.spacing.coefficients
    for k in coefficients:
        assert not isinstance(coefficients[k], na.AbstractUncertainScalarArray)


def test_wavelength_lines():
    result = _fits._wavelength_lines()
    assert na.shape(result) == dict(wavelength=3)
    assert np.all(np.diff(result.ndarray) > 0)


def test_wavelengths_alignment():
    result = _fits._wavelengths_alignment()
    assert set(result) == set(_fits._LINES_ALIGNMENT)
    for wavelength in result.values():
        assert wavelength.unit.is_equivalent(u.AA)


def test_pupil():
    result = _fits._pupil()
    assert na.shape(result) == dict(pupil_x=2, pupil_y=2)


def test_enforce_shared():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = [
        esis.optics.DistortionParameters.from_instrument(instrument[dict(channel=c)])
        for c in range(4)
    ]
    for c, p in enumerate(parameters):
        p.displacement_primary = -c * u.mm
        p.pitch = c * u.arcsec
        p.yaw_grating = p.yaw_grating + c * u.arcmin
    result = _fits.enforce_shared(parameters)
    for p in result:
        assert p.displacement_primary == -1.5 * u.mm
        assert p.pitch == 1.5 * u.arcsec
    # a channel's own terms are untouched, and the inputs are not modified
    for c, (p, q) in enumerate(zip(parameters, result)):
        assert q.yaw_grating == p.yaw_grating
        assert p.displacement_primary == -c * u.mm


def test_stack():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = [
        esis.optics.DistortionParameters.from_instrument(instrument[dict(channel=c)])
        for c in range(4)
    ]
    result = _fits._stack(parameters, axis="channel")
    assert na.shape(result) == dict(channel=4)
    for c in range(4):
        assert result.roll[dict(channel=c)] == parameters[c].roll

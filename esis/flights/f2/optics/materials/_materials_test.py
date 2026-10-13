import pytest
import astropy.units as u
import optika
import esis


def test_multilayer_AlSc():
    result = esis.flights.f2.optics.materials.multilayer_AlSc()
    assert isinstance(result, optika.materials.MultilayerMirror)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        esis.flights.f2.wavelength_Ne_VII,
        esis.flights.f2.wavelength_Si_XII,
    ],
)
def test_multilayer_AlSc_oxide(wavelength: u.Quantity):
    oxide = esis.flights.f2.optics.materials.multilayer_AlSc().layers[0]
    n = oxide.n(wavelength).ndarray
    assert n.real < 1
    assert n.imag > 0


def test_multilayer_SiSc():
    result = esis.flights.f2.optics.materials.multilayer_SiSc()
    assert isinstance(result, optika.materials.MultilayerMirror)

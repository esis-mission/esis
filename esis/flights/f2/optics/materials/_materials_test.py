from typing import Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis
from esis.flights.f2.optics.materials import _materials

materials = esis.flights.f2.optics.materials


def _layers(
    multilayer: optika.materials.MultilayerMirror,
) -> list[optika.materials.Layer]:
    """Every layer of a multilayer stack, top first, and its substrate last."""
    result = []
    for layer in multilayer.layers:
        if isinstance(layer, optika.materials.PeriodicLayerSequence):
            result += list(layer.layers)
        else:
            result.append(layer)
    return result + [multilayer.substrate]


def _reflectivity(
    multilayer: optika.materials.MultilayerMirror,
    wavelength: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """Compute the reflectivity of a multilayer stack at normal incidence."""
    return multilayer.efficiency(
        rays=optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        ),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )


def test_multilayer_AlSc():
    result = materials.multilayer_AlSc()
    assert isinstance(result, optika.materials.MultilayerMirror)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        esis.flights.f2.wavelength_Ne_VII,
        esis.flights.f2.wavelength_Si_XII,
    ],
)
def test_multilayer_AlSc_oxide(wavelength: u.Quantity):
    oxide = materials.multilayer_AlSc().layers[0]
    n = oxide.n(wavelength).ndarray
    assert n.real < 1
    assert n.imag > 0


def test_multilayer_SiSc():
    """The design has the interfaces fitted to the sample."""
    result = materials.multilayer_SiSc()
    assert isinstance(result, optika.materials.MultilayerMirror)
    for layer, layer_sample in zip(
        result.layers[~0].layers,
        materials.multilayer_SiSc_sample_fit().layers[~0].layers,
    ):
        assert layer.chemical == layer_sample.chemical
        assert layer.interface.width == layer_sample.interface.width


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        esis.flights.f2.wavelength_Ne_VII,
        esis.flights.f2.wavelength_Si_XII,
    ],
)
@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        materials.multilayer_SiSc,
        materials.multilayer_SiSc_sample_fit,
    ],
)
def test_multilayer_SiSc_layers(
    function: Callable[[], optika.materials.MultilayerMirror],
    wavelength: u.Quantity,
):
    """Every layer has a physical index of refraction in the passband."""
    for layer in _layers(function()):
        n = layer.n(wavelength).ndarray
        assert 0.6 < n.real < 1.1
        assert 0 < n.imag < 0.5


def test_multilayer_SiSc_amorphous():
    """The Si layers, the oxide, and the glass substrate are all amorphous."""
    for layer in _layers(materials.multilayer_SiSc()):
        chemical = layer.chemical
        if isinstance(chemical, optika.chemicals.Chemical) and "Si" in chemical.formula:
            assert chemical.is_amorphous


def test_multilayer_SiSc_reflectivity():
    """The design reflects about 0.30 at Ne VII and 0.25 at Si XII."""
    wavelength = na.ScalarArray(
        u.Quantity(
            [esis.flights.f2.wavelength_Ne_VII, esis.flights.f2.wavelength_Si_XII]
        ),
        axes="wavelength",
    )
    r = _reflectivity(materials.multilayer_SiSc(), wavelength).ndarray
    assert np.all(r > 0.24)
    assert np.all(r < 0.32)


def test_time_coating_sample() -> None:
    """The sample was coated on the date in its name."""
    path = _materials._path_directory() / "multilayer/mul080572.abs"
    header = path.read_text().splitlines()[0]
    assert header == f"# CX{materials.time_coating_sample.strftime('%y%m%d')}B"


def test_multilayer_SiSc_sample_measured():
    result = materials.multilayer_SiSc_sample_measured()
    assert isinstance(result, optika.materials.MeasuredMirror)
    measurement = result.efficiency_measured
    wavelength = measurement.inputs.wavelength
    assert wavelength.shape == dict(wavelength=250)
    assert np.isclose(wavelength.min().ndarray, 25.09783 * u.nm)
    assert np.isclose(wavelength.max().ndarray, 50.00188 * u.nm)
    index = np.argmax(measurement.outputs.ndarray)
    assert 46 * u.nm < wavelength.ndarray[index] < 47 * u.nm
    assert 0.36 < measurement.outputs.ndarray[index] < 0.38


def test_multilayer_SiSc_sample_fit():
    """The fit follows the measurement, without reaching the bounds of the fit."""
    result = materials.multilayer_SiSc_sample_fit()
    assert isinstance(result, optika.materials.MultilayerMirror)

    measurement = materials.multilayer_SiSc_sample_measured().efficiency_measured
    index = dict(wavelength=slice(None, -1))
    angle = measurement.inputs.direction
    r, _ = optika.materials.multilayer_efficiency(
        wavelength=measurement.inputs.wavelength[index],
        direction=np.cos(angle),
        layers=result.layers,
        substrate=result.substrate,
        num_interpolation=result.num_interpolation,
    )
    f = _materials._fraction_s
    r = f * r.s + (1 - f) * r.p
    ratio = measurement.outputs[index] / r
    assert np.sqrt(np.mean(np.square(np.log(ratio)))).ndarray < 0.1

    Si, Sc = result.layers[~0].layers
    assert 25 * u.nm < Si.thickness + Sc.thickness < 27 * u.nm
    for layer in (Si, Sc):
        assert 0.1 * u.nm < layer.interface.width < 3.9 * u.nm

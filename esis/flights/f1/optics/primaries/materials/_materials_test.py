import pathlib
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis


def test_multilayer_design():
    r = esis.flights.f1.optics.primaries.materials.multilayer_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_multilayer_witness_measured():
    r = esis.flights.f1.optics.primaries.materials.multilayer_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_time_coating() -> None:
    """The witness was coated on the date in the name of its sample."""
    materials = esis.flights.f1.optics.primaries.materials
    path = pathlib.Path(materials.__file__).parent / "_data/mul063931.abs"
    header = path.read_text().splitlines()[0]
    time = materials.time_coating
    assert header.startswith(f"# CX{time.strftime('%y%m%d')}")
    assert time < materials.time_measurement


def test_multilayer_witness_fit():
    r = esis.flights.f1.optics.primaries.materials.multilayer_witness_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)


def test_multilayer_fit():
    r = esis.flights.f1.optics.primaries.materials.multilayer_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)


def test_multilayer_fit_time() -> None:
    """
    Oxide grown since the measurement tops the coating and lowers its reflectivity.

    When the witness was measured, the coating is the witness fit on glass.
    """
    materials = esis.flights.f1.optics.primaries.materials
    witness = materials.multilayer_witness_fit()
    measured = materials.multilayer_fit(time=materials.time_measurement)
    flight = materials.multilayer_fit()

    thickness = [layer.thickness for layer in witness.layers]
    assert [layer.thickness for layer in measured.layers] == thickness
    assert [layer.thickness for layer in flight.layers[1:]] == thickness
    assert 0 * u.nm < flight.layers[0].thickness < 1 * u.nm

    angle = 4 * u.deg
    rays = optika.rays.RayVectorArray(
        wavelength=esis.flights.f1.spectrum.O_V.wavelength,
        direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
    )
    normal = na.Cartesian3dVectorArray(0, 0, -1)
    assert flight.efficiency(rays, normal) < measured.efficiency(rays, normal)


def test_multilayer_fit_before_measurement() -> None:
    """The oxide the witness was measured with cannot be taken off the fit."""
    materials = esis.flights.f1.optics.primaries.materials
    with pytest.raises(ValueError):
        materials.multilayer_fit(time=materials.time_measurement - 1 * u.day)

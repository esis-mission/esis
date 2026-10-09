import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis


def test_multilayer_design():
    r = esis.flights.f1.optics.gratings.materials.multilayer_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_multilayer_witness_measured():
    r = esis.flights.f1.optics.gratings.materials.multilayer_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_multilayer_witness_measured_efficiency() -> None:
    """Each witness is evaluated on its own wavelength samples."""
    r = esis.flights.f1.optics.gratings.materials.multilayer_witness_measured()
    measurement = r.efficiency_measured
    wavelength = esis.flights.f1.spectrum.O_V.wavelength
    angle = measurement.inputs.direction
    rays = optika.rays.RayVectorArray(
        wavelength=wavelength,
        direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
    )
    result = r.efficiency(rays, na.Cartesian3dVectorArray(0, 0, -1))
    expected = na.interp(
        wavelength,
        measurement.inputs.wavelength,
        measurement.outputs,
        axis="wavelength",
    )
    assert result.shape == r.shape == dict(channel=3)
    assert np.all(result == expected)
    assert np.all(result > 0.3 * u.dimensionless_unscaled)


def test_time_coating() -> None:
    """Each witness was coated on the date in the name of its sample."""
    materials = esis.flights.f1.optics.gratings.materials
    serial_number = materials.multilayer_witness_measured().serial_number
    directory = pathlib.Path(materials.__file__).parent / "_data"
    assert materials.time_coating.shape == serial_number.shape
    for index in na.ndindex(serial_number.shape):
        number = serial_number[index].ndarray[-2:]
        path = directory / f"Witness_g{number}.txt"
        header = path.read_text().splitlines()[0]
        time = materials.time_coating[index].ndarray
        assert header.startswith(f"# CX{time.strftime('%y%m%d')}")
    assert np.all(materials.time_coating < materials.time_measurement)


def test_multilayer_witness_fit():
    r = esis.flights.f1.optics.gratings.materials.multilayer_witness_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)


def test_multilayer_fit():
    r = esis.flights.f1.optics.gratings.materials.multilayer_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)

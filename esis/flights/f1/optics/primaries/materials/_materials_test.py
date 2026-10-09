import pathlib
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

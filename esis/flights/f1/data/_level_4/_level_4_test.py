import pytest
import astropy.units as u
import esis

ctis = pytest.importorskip("ctis")

if not hasattr(ctis.instruments, "OptikaInstrument"):  # pragma: nocover
    pytest.skip(
        reason="ctis is missing OptikaInstrument (needs ctis PR #18)",
        allow_module_level=True,
    )


def test_level_4():
    result = esis.flights.f1.data.level_4(
        pitch_scene=16 * u.arcsec,
        pitch_velocity=200 * u.km / u.s,
        num_iteration=2,
    )
    assert isinstance(result, esis.data.Level_4)

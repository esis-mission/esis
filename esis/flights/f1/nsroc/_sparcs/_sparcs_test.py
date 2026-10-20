import dataclasses
import astropy.units as u
import esis


def test_sparcs() -> None:
    result = esis.flights.f1.nsroc.sparcs()

    assert isinstance(result, esis.nsroc.Sparcs)

    for f in dataclasses.fields(result):
        assert getattr(result, f.name) > 0

    assert result.jitter_pointing.unit.is_equivalent(u.arcsec)
    assert result.drift_pointing.unit.is_equivalent(u.arcsec / u.s)
    assert result.jitter_roll.unit.is_equivalent(u.arcsec)
    assert result.drift_roll.unit.is_equivalent(u.arcsec / u.s)

import dataclasses
import astropy.units as u
import astropy.time
import esis


def test_timeline() -> None:
    result = esis.flights.f1.nsroc.timeline()

    assert isinstance(result, esis.nsroc.Timeline)
    assert isinstance(result.time_mission_start, astropy.time.Time)

    for f in dataclasses.fields(result):
        if f.name.startswith("timedelta_"):
            assert getattr(result, f.name) > 0 * u.s

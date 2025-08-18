import esis


def test_requirements():
    req = esis.flights.f1.optics.requirements()
    assert isinstance(req, esis.optics.Requirements)

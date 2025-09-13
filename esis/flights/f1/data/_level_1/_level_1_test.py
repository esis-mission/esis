import esis


def test_level_0():

    a = esis.flights.f1.data.level_1()

    assert isinstance(a, esis.data.Level_1)

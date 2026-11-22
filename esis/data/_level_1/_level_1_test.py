import pytest
import numpy as np
from ..abc._channel_data_test import AbstractTestAbstractChannelData
import esis

# A copy, so that the rest of the flight's images are not kept alive for the
# whole session by this view of them.
_level_0 = esis.flights.f1.data.level_0()[dict(time=slice(None, None, 8))].copy()

_level_1 = esis.data.Level_1.from_level_0(_level_0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        _level_1,
    ],
)
class TestLevel_1(
    AbstractTestAbstractChannelData,
):
    pass


def test_from_level_0_whole_sequence() -> None:
    """
    Check that converting the dark and solar images separately changes nothing.

    :meth:`esis.data.Level_1.from_level_0` converts them to electrons
    separately, which gives the same images as converting the whole
    sequence only while each step before the dark subtraction acts on each
    image on its own.
    """
    a = _level_0
    taps = a.taps.unbiased.active.electrons
    b = a.from_taps(taps).dark_subtracted.lights.despiked
    b = b.outputs[{b.axis_y: slice(None, None, -1)}]
    assert np.all(_level_1.outputs == b)

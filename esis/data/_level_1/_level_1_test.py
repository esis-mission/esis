import pytest
import numpy as np
from ..abc._channel_data_test import AbstractTestAbstractChannelData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.data.Level_1.from_level_0(
            esis.flights.f1.data.level_0()[dict(time=slice(None, None, 8))],
        ),
    ],
)
class TestLevel_1(
    AbstractTestAbstractChannelData,
):
    def test_where_shadow(self, a: esis.data.Level_1):
        result = a.where_shadow()
        assert result.dtype == bool
        assert set(result.shape) == {a.axis_channel, a.axis_x}
        num = result.sum(a.axis_x)
        assert np.all(num.ndarray <= 55)
        assert np.any(result)

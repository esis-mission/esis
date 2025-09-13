import pytest
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
    pass

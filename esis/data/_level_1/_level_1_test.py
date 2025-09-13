import pytest
from ..abc._channel_data_test import AbstractTestAbstractChannelData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.data.level_1()[dict(time=slice(None, None, 8))],
    ],
)
class TestLevel_1(
    AbstractTestAbstractChannelData,
):
    pass

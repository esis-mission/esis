import esis
from ... import optics
from .. import level_0

__all__ = [
    "level_1",
]


@esis.memory.cache
def level_1() -> esis.data.Level_1:
    """
    The ESIS data processed to the Level-1 stage.
    """
    return esis.data.Level_1.from_level_0(
        a=level_0(),
        instrument=optics.as_built(),
    )

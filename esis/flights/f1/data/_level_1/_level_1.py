import esis
from ... import optics
from .. import level_0

__all__ = [
    "level_1",
]


@esis.memory.cache
def level_1() -> esis.data.Level_1:
    """
    Load the ESIS images and process them to the :doc:`reports/level_1` stage.

    This function takes the result of :func:`level_0`,
    creates a new instance of :class:`esis.data.Level_1` using the
    :meth:`~esis.data.Level_1.from_level_0` classmethod,
    and then caches the result for future use.
    """
    return esis.data.Level_1.from_level_0(
        a=level_0(),
        instrument=optics.as_built(),
    )

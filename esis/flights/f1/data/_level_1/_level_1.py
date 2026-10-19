import esis
from esis import _caching
from ... import optics
from .. import level_0

__all__ = [
    "level_1",
]

_packages = (
    "euv-snapshot-imaging-spectrograph",
    "msfc-ccd",
    "optika",
    "named-arrays",
)
"""
The packages whose versions the cache of :func:`level_1` is keyed on.

They read the images and calibrate them (this package and :mod:`msfc_ccd`),
and they define the instrument model and the arrays the result holds.
"""


@esis.memory.cache
def _level_1(versions: dict[str, None | str]) -> esis.data.Level_1:
    """
    Compute :func:`level_1`, and remember it.

    ``versions`` is not read here. It is an argument so that it is part of
    the key the cache is looked up by, so that a new release of any of
    :data:`_packages` computes the result again rather than returning one
    computed by the old release. The serial number of camera SN10 went on
    reading ``"1"`` from a cache computed before :mod:`msfc_ccd` 1.1.0
    fixed it, for example.

    Parameters
    ----------
    versions
        The versions of :data:`_packages`.
    """
    return esis.data.Level_1.from_level_0(
        a=level_0(),
        instrument=optics.as_built(),
    )


def level_1() -> esis.data.Level_1:
    """
    Load the ESIS images and process them to the Level-1 stage.

    This function takes the result of :func:`level_0`,
    creates a new instance of :class:`esis.data.Level_1` using the
    :meth:`~esis.data.Level_1.from_level_0` classmethod,
    and then caches the result for future use.

    The cache is kept for each combination of versions of this package and
    of the packages it depends on to read and calibrate the images, so
    installing a new release of any of them computes the result again.
    """
    return _level_1(versions=_caching.versions(*_packages))

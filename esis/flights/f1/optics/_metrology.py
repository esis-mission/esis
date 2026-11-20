"""Our own measurements of the optics which flew, from the esis-data-2019 repository."""

import pathlib
from esis.flights.f1.data._fits._fits import _path_cache, _retrieve

__all__ = []

_url = (
    "https://github.com/esis-mission/esis-data-2019/releases/download"
    "/metrology-v1.1/esis-2019-metrology.tar"
)
_hash = "sha256:5621162bfef7542610a2ddaf1f7f925964900c08cb3d59da51419752f88242a8"


def _path_directory() -> pathlib.Path:
    """
    Return the directory containing our measurements of the optics.

    They are downloaded from the ``metrology-v1.1`` release of the
    `esis-data-2019 <https://github.com/esis-mission/esis-data-2019/releases/tag/metrology-v1.1>`_
    repository, and unpacked into
    :func:`esis.flights.f1.data._fits._fits._path_cache`, the first time they
    are needed.
    The maps of the primary mirror are in its ``primary`` directory, and the
    maps of the gratings in its ``gratings`` directory.
    """
    _retrieve(
        url=_url,
        known_hash=_hash,
        fname="esis-2019-metrology-v1.1.tar",
        path=_path_cache(),
        extract_dir="metrology-v1.1",
    )
    return _path_cache() / "metrology-v1.1"

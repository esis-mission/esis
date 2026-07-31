import os
import pathlib
import joblib

__all__ = [
    "memory",
]

_path_cache = pathlib.Path(
    os.environ.get(
        "ESIS_CACHE_DIR",
        pathlib.Path.home() / ".esis/cache",
    )
)

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""
A representation of the cache which stores intermediate results.

The cache lives in ``~/.esis/cache`` by default; set the ``ESIS_CACHE_DIR``
environment variable before importing :mod:`esis` to relocate it (for
example, to a scratch filesystem on a cluster).
"""

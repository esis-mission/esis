import importlib.metadata
import pathlib
import joblib

__all__ = [
    "memory",
]

_path_cache = pathlib.Path.home() / ".esis/cache"

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""A representation of the cache which stores intermediate results."""


def versions(*packages: str) -> dict[str, None | str]:
    """
    Return the installed version of each package, or :obj:`None` if absent.

    Passing these to a function cached by :obj:`memory` makes them part of
    the key its results are looked up by. :mod:`joblib` hashes the arguments
    and the source of the cached function, neither of which changes when a
    new release of a package it calls changes the answer.

    Parameters
    ----------
    packages
        The names of the packages, as they are installed.
    """
    result = dict()
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

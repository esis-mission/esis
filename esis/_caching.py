import inspect
import pathlib
import joblib

__all__ = [
    "memory",
]

f = inspect.isfunction
inspect.isfunction = lambda x: f(x) or isinstance(x, joblib.memory.MemorizedFunc)

_path_cache = pathlib.Path(__file__).parent.parent / ".cache"

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""A representation of the cache which stores intermediate results."""

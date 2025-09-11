import inspect
import pathlib
import joblib

__all__ = [
    "memory",
]

class MemorizedFunc(
    joblib.memory.MemorizedFunc,
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f =  self.func
        self.__code__ = f.__code__
        self.__name__ = f.__name__
        self.__doc__ = f.__doc__
        self.__annotations__ = f.__annotations__
        self.__defaults__ = f.__defaults__
        self.__kwdefaults__ = f.__kwdefaults__


joblib.memory.MemorizedFunc = MemorizedFunc

f = inspect.isfunction
inspect.isfunction = lambda x: f(x) or isinstance(x, joblib.memory.MemorizedFunc)

_path_cache = pathlib.Path.home() / ".esis/cache"

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""A representation of the cache which stores intermediate results."""

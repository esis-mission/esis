import pathlib
import joblib

_path_cache = pathlib.Path(__file__).parent.parent / ".cache"

memory = joblib.Memory(
    location=_path_cache,
    mmap_mode="r",
    verbose=0,
)
"""A representation of the cache which stores intermediate results."""

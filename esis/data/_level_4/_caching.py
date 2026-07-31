"""
Joblib-cached wrappers for the expensive steps of the Level-4 MART inversion.

The linearization raytrace and the regridding-weights build take minutes and
tens of gigabytes of memory, but they are pure functions of the instrument
model, the scene grid, and the code that computes them.  The wrappers here
persist their results to the ESIS cache (``~/.esis/cache``) with a key built
from everything the result depends on:

- the optical system, fingerprinted with :func:`key_system` immediately after
  construction, so a change to the fitted instrument model invalidates the
  cache;
- the explicit grid arguments, so changing the scene grid computes and stores
  a new entry instead of ever returning a stale one; and
- :func:`code_state`, the installed versions (and git state, for editable
  installs) of the libraries that do the actual computation, so pulling new
  ``optika``/``named-arrays``/``regridding`` code invalidates the cache.

Since ``regridding`` PR #44, the sparse weights are tuples of flat arrays
``(indices_input, indices_output, values)`` with unique, coalesced index
pairs — a form that pickles and memory-maps cleanly — so the cached values
are exactly what :mod:`optika` computes.  The public accessors additionally
downcast each element to ``int32`` indices and ``float32`` values, which
halves the memory traffic of every application; that traffic is what sets
the MART iteration time.

Old entries are never evicted automatically; delete ``~/.esis/cache`` (or
call ``esis.memory.clear()``) to reclaim disk space.
"""

import ctypes
import importlib.metadata
import pathlib
import subprocess
import sys

import joblib
import named_arrays as na
import numpy as np
import optika

import esis

__all__ = [
    "code_state",
    "key_system",
    "linear_system",
    "weights",
    "weights_transpose",
]


def code_state() -> str:
    """
    Fingerprint the code that linearizes the system and builds the weights.

    The result includes the version of every library in the raytrace/regrid
    stack and, if ``optika`` is an editable install inside a git repository,
    its current commit, with a ``-dirty`` suffix if there are uncommitted
    changes.  Note that the fingerprint cannot distinguish between two
    different dirty states of the same commit; commit (or stash) changes to
    ``optika`` to be safe.
    """
    parts = [
        f"{package} {importlib.metadata.version(package)}"
        for package in ("optika", "named-arrays", "regridding")
    ]
    repo = pathlib.Path(optika.__file__).parent.parent
    if (repo / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(repo), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            parts.append(f"optika-git {result.stdout.strip()}")
    return "; ".join(parts)


def key_system(system: optika.systems.AbstractSequentialSystem) -> str:
    """
    Fingerprint a freshly-constructed optical system for use as a cache key.

    Call this immediately after constructing the system, before anything
    raytraces it: lazy attributes (like the default rayfunction) land in the
    instance ``__dict__`` when first evaluated, and hashing after that point
    would make the fingerprint depend on evaluation order instead of only on
    the model parameters.

    Parameters
    ----------
    system
        The optical system to fingerprint.
    """
    return joblib.hash(system)


@esis.memory.cache(ignore=["system"])
def linear_system(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    code: str,
) -> optika.systems.LinearSystem:
    """
    Linearize the given system, caching the result in the ESIS cache.

    Parameters
    ----------
    system
        The sequential system to linearize.  Excluded from the cache key
        (hashing it here would be sensitive to its lazy attributes); `key`
        stands in for it instead.
    key
        The fingerprint of `system` computed by :func:`key_system`.
    wavelength
        The vertices of the wavelength grid to linearize on.
    degree
        The degree of the polynomial distortion and vignetting models.
    code
        The result of :func:`code_state`.
    """
    return system.linearize(wavelength=wavelength, degree=degree)


@esis.memory.cache(ignore=["system"])
def _weights(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    coordinates_scene: na.AbstractSpectralPositionalVectorArray,
    axis_wavelength: str,
    axis_field: tuple[str, str],
    code: str,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Build the scene-to-sensor weights, caching them at full precision.

    Parameters
    ----------
    system
        The sequential system whose linearization maps the scene onto the
        sensor.
    key
        The fingerprint of `system` computed by :func:`key_system`.
    wavelength
        The vertices of the wavelength grid the system was linearized on.
    degree
        The degree of the polynomial distortion and vignetting models.
    coordinates_scene
        The vertices of the scene grid to compute weights for.
    axis_wavelength
        The logical axis of `coordinates_scene` corresponding to changing
        wavelength.
    axis_field
        The logical axes of `coordinates_scene` corresponding to changing
        position on the object plane.
    code
        The result of :func:`code_state`.
    """
    linear = linear_system(system, key, wavelength, degree, code)
    return linear.weights(
        coordinates=coordinates_scene,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
    )


@esis.memory.cache(ignore=["system"])
def _weights_transposed(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    coordinates_scene: na.AbstractSpectralPositionalVectorArray,
    axis_wavelength: str,
    axis_field: tuple[str, str],
    code: str,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Build the conservative transpose weights, caching them at full precision.

    The transpose is computed from the cached forward weights by
    :meth:`optika.systems.AbstractLinearSystem.weights_transposed`, whose
    radiometric ``weights_input`` (vignetting times effective area) varies
    across the field — the case fixed by ``regridding`` PR #47.

    Parameters
    ----------
    system
        The sequential system whose linearization maps the scene onto the
        sensor.
    key
        The fingerprint of `system` computed by :func:`key_system`.
    wavelength
        The vertices of the wavelength grid the system was linearized on.
    degree
        The degree of the polynomial distortion and vignetting models.
    coordinates_scene
        The vertices of the scene grid to compute weights for.
    axis_wavelength
        The logical axis of `coordinates_scene` corresponding to changing
        wavelength.
    axis_field
        The logical axes of `coordinates_scene` corresponding to changing
        position on the object plane.
    code
        The result of :func:`code_state`.
    """
    forward = _weights(
        system,
        key,
        wavelength,
        degree,
        coordinates_scene,
        axis_wavelength,
        axis_field,
        code,
    )
    linear = linear_system(system, key, wavelength, degree, code)
    return linear.weights_transposed(
        weights=forward,
        coordinates=coordinates_scene,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
    )


def _release_working_set(array: np.ndarray) -> None:
    """
    Drop a memory-mapped array's pages from the process working set (Windows).

    Streaming through a large memory-mapped cache entry leaves its clean
    pages resident faster than the operating system evicts them, which can
    crowd out the arrays being materialized alongside and exhaust physical
    memory.  ``VirtualUnlock`` on an unlocked region is documented to remove
    the pages from the working set without any pagefile traffic.  No-op for
    arrays that are not memory maps and for non-Windows platforms.

    Parameters
    ----------
    array
        The array whose pages should be released.
    """
    if not isinstance(array, np.memmap):
        return
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.VirtualUnlock(
        ctypes.c_void_p(array.ctypes.data),
        ctypes.c_size_t(array.nbytes),
    )


def _compact(
    weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Downcast each weights element to ``int32`` indices and ``float32`` values.

    The ``float32`` values sacrifice ~1e-7 relative precision on the overlap
    areas but halve the memory traffic of every regrid application, which is
    what sets the MART iteration time.  The indices are intra-element and
    never exceed the number of sensor pixels (~2.1e6), so ``int32`` is exact.

    Parameters
    ----------
    weights
        The weights, and their input and output shapes, as loaded from the
        cache.
    """
    array, shape_input, shape_output = weights
    flat = array.ndarray.reshape(-1)
    compact = np.empty(flat.size, dtype=object)
    for k in range(flat.size):
        indices_input, indices_output, values = flat[k]
        compact[k] = (
            np.ascontiguousarray(indices_input, dtype=np.int32),
            np.ascontiguousarray(indices_output, dtype=np.int32),
            np.ascontiguousarray(values, dtype=np.float32),
        )
        _release_working_set(indices_input)
        _release_working_set(indices_output)
        _release_working_set(values)
    compact = na.ScalarArray(compact.reshape(array.ndarray.shape), axes=array.axes)
    return compact, shape_input, shape_output


def weights(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    coordinates_scene: na.AbstractSpectralPositionalVectorArray,
    axis_wavelength: str,
    axis_field: tuple[str, str],
    code: str,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Build the scene-to-sensor regridding weights, caching them in the ESIS cache.

    The scene coordinates are part of the cache key, so every distinct scene
    grid gets its own cache entry.  The result is downcast by :func:`_compact`.

    Parameters
    ----------
    system
        The sequential system whose linearization maps the scene onto the
        sensor.  Excluded from the cache key (hashing it here would be
        sensitive to its lazy attributes); `key` stands in for it instead.
    key
        The fingerprint of `system` computed by :func:`key_system`.
    wavelength
        The vertices of the wavelength grid the system was linearized on.
    degree
        The degree of the polynomial distortion and vignetting models.
    coordinates_scene
        The vertices of the scene grid to compute weights for.
    axis_wavelength
        The logical axis of `coordinates_scene` corresponding to changing
        wavelength.
    axis_field
        The logical axes of `coordinates_scene` corresponding to changing
        position on the object plane.
    code
        The result of :func:`code_state`.
    """
    return _compact(
        _weights(
            system,
            key,
            wavelength,
            degree,
            coordinates_scene,
            axis_wavelength,
            axis_field,
            code,
        )
    )


def weights_transpose(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    coordinates_scene: na.AbstractSpectralPositionalVectorArray,
    axis_wavelength: str,
    axis_field: tuple[str, str],
    code: str,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Build the conservative transpose weights, caching them in the ESIS cache.

    The result maps sensor pixels back to scene cells and is downcast by
    :func:`_compact`.  Assign it to the ``weights_transpose`` attribute of a
    :class:`ctis.instruments.OptikaInstrument` to pre-empt the (much more
    memory-hungry) upstream computation.

    Parameters
    ----------
    system
        The sequential system whose linearization maps the scene onto the
        sensor.  Excluded from the cache key (hashing it here would be
        sensitive to its lazy attributes); `key` stands in for it instead.
    key
        The fingerprint of `system` computed by :func:`key_system`.
    wavelength
        The vertices of the wavelength grid the system was linearized on.
    degree
        The degree of the polynomial distortion and vignetting models.
    coordinates_scene
        The vertices of the scene grid to compute weights for.
    axis_wavelength
        The logical axis of `coordinates_scene` corresponding to changing
        wavelength.
    axis_field
        The logical axes of `coordinates_scene` corresponding to changing
        position on the object plane.
    code
        The result of :func:`code_state`.
    """
    return _compact(
        _weights_transposed(
            system,
            key,
            wavelength,
            degree,
            coordinates_scene,
            axis_wavelength,
            axis_field,
            code,
        )
    )

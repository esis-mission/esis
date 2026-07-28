"""
Joblib-cached wrappers for the expensive steps of the MART inversion report.

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

The sparse regridding weights computed by :mod:`regridding` are ragged
``numba.typed.List`` objects of ``(input, output, weight)`` triples.  Holding
the weights and their conservative transpose in that form costs ~50 bytes per
triple across two full copies, which exceeds this machine's memory for scene
pitches near the plate scale.  This module therefore converts each list into
three flat arrays ``(indices_input, indices_output, values)`` — the form the
cache stores — and patches :func:`regridding.regrid_from_weights` with a
dispatching wrapper so the flat-array form can be applied directly by an
equivalent numba kernel, without ever rebuilding the typed lists.  The
conservative transpose is computed one ``(channel, wavelength)`` element at a
time with :func:`named_arrays.regridding.transpose_weights_conservative`
(the same upstream routine :mod:`optika` uses), so its peak memory is a few
gigabytes instead of two full copies of the weights.

Old entries are never evicted automatically; delete ``~/.esis/cache`` (or call
``esis.memory.clear()``) to reclaim disk space.
"""

import ctypes
import importlib.metadata
import pathlib
import subprocess
import sys

import joblib
import named_arrays as na
import numba
import numpy as np
import optika
import regridding

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


@numba.njit(cache=True)
def _triples_to_arrays(
    triples: numba.typed.List,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(triples)
    indices = np.empty((n, 2), dtype=np.int64)
    values = np.empty(n, dtype=np.float64)
    for k in range(n):
        i, j, w = triples[k]
        indices[k, 0] = i
        indices[k, 1] = j
        values[k] = w
    return indices, values


@numba.njit(cache=True)
def _arrays_to_triples(
    indices: np.ndarray,
    values: np.ndarray,
) -> numba.typed.List:
    triples = numba.typed.List()
    triples.append((numba.int64(0), numba.int64(0), 0.0))
    triples.pop()
    for k in range(values.shape[0]):
        triples.append((indices[k, 0], indices[k, 1], values[k]))
    return triples


def _pack_weights(
    weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
) -> tuple:
    """
    Convert regridding weights into a picklable form for the joblib cache.

    The sparse weights computed by :mod:`regridding` are ragged
    ``numba.typed.List`` objects, which cannot be pickled; this converts each
    list of ``(input, output, weight)`` triples into a pair of flat
    :class:`numpy.ndarray` instances.

    Parameters
    ----------
    weights
        The weights, and their input and output shapes, as computed by
        :meth:`optika.systems.AbstractLinearSystem.weights`.
    """
    array, shape_input, shape_output = weights
    flat = array.ndarray.reshape(-1)
    packed = []
    for k in range(flat.size):
        packed.append(_triples_to_arrays(flat[k]))
        # free each element's typed list as it is packed, so the peak memory
        # is one full copy plus the growing arrays instead of two full copies
        flat[k] = None
    return packed, array.ndarray.shape, array.axes, shape_input, shape_output


@esis.memory.cache(ignore=["system"])
def _weights_packed(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    coordinates_scene: na.AbstractSpectralPositionalVectorArray,
    axis_wavelength: str,
    axis_field: tuple[str, str],
    code: str,
) -> tuple:
    """
    Build the scene-to-sensor weights and return them in picklable form.

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
    result = linear.weights(
        coordinates=coordinates_scene,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
    )
    return _pack_weights(result)


@esis.memory.cache(ignore=["system"])
def _weights_transposed_packed(
    system: optika.systems.AbstractSequentialSystem,
    key: str,
    wavelength: na.AbstractScalar,
    degree: int,
    coordinates_scene: na.AbstractSpectralPositionalVectorArray,
    axis_wavelength: str,
    axis_field: tuple[str, str],
    code: str,
) -> tuple:
    """
    Build the conservative transpose weights and return them in picklable form.

    This replicates the radiometric preamble of
    :meth:`optika.systems.AbstractLinearSystem.weights_transposed` and then
    calls :func:`named_arrays.regridding.transpose_weights_conservative` one
    ``(channel, wavelength)`` element at a time, rebuilding only that
    element's ``numba.typed.List`` from the packed forward weights.  Peak
    memory is therefore a few gigabytes instead of two full copies of the
    weights.

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
    packed = _weights_packed(
        system,
        key,
        wavelength,
        degree,
        coordinates_scene,
        axis_wavelength,
        axis_field,
        code,
    )
    packed_lists, shape, axes, shape_input, shape_output = packed

    linear = linear_system(system, key, wavelength, degree, code)

    coordinates = na.SpectralPositionalVectorArray(
        wavelength=coordinates_scene.wavelength,
        position=coordinates_scene.position,
    )
    coordinates = coordinates.cell_centers(axis_wavelength)

    position_sensor = linear.distortion.distort(coordinates).position

    coordinates_cell = coordinates.cell_centers(axis_field)

    vignetting = linear.vignetting
    if vignetting is not None:
        weights_vignetting = vignetting(coordinates_cell)
    else:
        weights_vignetting = 1

    weights_area = linear.area_effective(coordinates_cell.wavelength)

    weights_input = weights_vignetting * weights_area.value

    axis_pixel = linear.sensor.axis_pixel
    axis_output = (axis_pixel.x, axis_pixel.y)
    coordinates_sensor = linear.coordinates_sensor

    def _isel(a, item: dict):
        item = {ax: i for ax, i in item.items() if ax in a.shape}
        return a[item] if item else a

    packed_transposed = []
    for k, index in enumerate(np.ndindex(*shape)):
        item = {ax: slice(i, i + 1) for ax, i in zip(axes, index)}

        element = np.empty((1,) * len(shape), dtype=object)
        element[(0,) * len(shape)] = _arrays_to_triples(
            np.ascontiguousarray(packed_lists[k][0]),
            np.ascontiguousarray(packed_lists[k][1]),
        )
        element = na.ScalarArray(element, axes=axes)

        shape_input_k = {ax: (1 if ax in axes else n) for ax, n in shape_input.items()}
        shape_output_k = {
            ax: (1 if ax in axes else n) for ax, n in shape_output.items()
        }

        result_k = na.regridding.transpose_weights_conservative(
            weights=(element, shape_input_k, shape_output_k),
            coordinates_input=_isel(position_sensor, item),
            coordinates_output=_isel(coordinates_sensor, item),
            axis_input=axis_field,
            axis_output=axis_output,
            weights_input=_isel(weights_input, item),
        )

        array_k = result_k[0].ndarray.reshape(-1)[0]
        indices_k, values_k = _triples_to_arrays(array_k)

        # int32 halves the accumulated index storage, which is what sets the
        # peak memory of this function; the indices are intra-element and
        # never exceed the number of sensor pixels (~2.1e6).
        packed_transposed.append((indices_k.astype(np.int32), values_k))

        _release_working_set(packed_lists[k][0])
        _release_working_set(packed_lists[k][1])

    return packed_transposed, shape, axes, shape_output, shape_input


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


def _arrays_from_packed(
    packed: tuple,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Rebuild weights from the cache as flat-array elements.

    Each element of the result is a ``(indices_input, indices_output,
    values)`` tuple of flat arrays instead of a ``numba.typed.List``; the
    dispatching wrapper installed on :func:`regridding.regrid_from_weights`
    recognizes this form and applies it directly.

    Parameters
    ----------
    packed
        The result of :func:`_pack_weights`.
    """
    packed_lists, shape, axes, shape_input, shape_output = packed
    flat = np.empty(len(packed_lists), dtype=object)
    for k, (indices, values) in enumerate(packed_lists):
        # float32 values sacrifice ~1e-7 relative precision on the overlap
        # areas but cut the memory traffic of every regrid application, which
        # is what sets the MART iteration time.
        flat[k] = (
            np.ascontiguousarray(indices[:, 0], dtype=np.int32),
            np.ascontiguousarray(indices[:, 1], dtype=np.int32),
            np.ascontiguousarray(values, dtype=np.float32),
        )
        _release_working_set(indices)
        _release_working_set(values)
    array = na.ScalarArray(flat.reshape(shape), axes=axes)
    return array, shape_input, shape_output


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
    grid gets its own cache entry.  The result is returned in the flat-array
    form described in :func:`_arrays_from_packed`.

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
    packed = _weights_packed(
        system,
        key,
        wavelength,
        degree,
        coordinates_scene,
        axis_wavelength,
        axis_field,
        code,
    )
    return _arrays_from_packed(packed)


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

    The result maps sensor pixels back to scene cells and is returned in the
    flat-array form described in :func:`_arrays_from_packed`.  Assign it to
    the ``weights_transpose`` attribute of a
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
    packed = _weights_transposed_packed(
        system,
        key,
        wavelength,
        degree,
        coordinates_scene,
        axis_wavelength,
        axis_field,
        code,
    )
    return _arrays_from_packed(packed)


@numba.njit(cache=True, parallel=True)
def _regrid_from_weights_arrays_numba(
    weights: numba.typed.List,
    values_input: np.ndarray,
    values_output: np.ndarray,
) -> None:
    for d in numba.prange(len(weights)):
        d = numba.types.int64(d)
        idx_input, idx_output, vals = weights[d]
        values_input_d = values_input[d].reshape(-1)
        values_output_d = values_output[d].reshape(-1)
        for k in range(vals.shape[0]):
            values_output_d[idx_output[k]] += vals[k] * values_input_d[idx_input[k]]


def _regrid_from_weights_arrays(
    weights: np.ndarray,
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray = None,
    axis_input=None,
    axis_output=None,
) -> np.ndarray:
    """
    Apply flat-array weights to an array of values.

    This mirrors the axis bookkeeping of
    :func:`regridding.regrid_from_weights` line for line; only the element
    container (tuples of flat arrays instead of ``numba.typed.List``) and the
    numba kernel differ.

    Parameters
    ----------
    weights
        Object array whose elements are ``(indices_input, indices_output,
        values)`` tuples of flat arrays.
    shape_input
        Broadcasted shape of the input coordinates.
    shape_output
        Broadcasted shape of the output coordinates.
    values_input
        Input array of values to be resampled.
    values_output
        Optional array in which to place the output.
    axis_input
        Logical axes of the input array to resample.
    axis_output
        Logical axes of the output array corresponding to the resampled axes
        of the input array.

    Raises
    ------
    ValueError
        If `values_output` does not have the expected shape.
    """
    from regridding import _util

    unit = getattr(values_input, "unit", None)

    ndim_input = len(shape_input)
    ndim_output = len(shape_output)

    axis_input = _util._normalize_axis(axis_input, ndim=ndim_input)
    axis_output = _util._normalize_axis(axis_output, ndim=ndim_output)

    shape_input_orthogonal = tuple(
        shape_input[i]
        for i in _util._normalize_axis(None, ndim=len(shape_input))
        if i not in axis_input
    )
    shape_output_orthogonal = tuple(
        shape_output[i]
        for i in _util._normalize_axis(None, ndim=len(shape_output))
        if i not in axis_output
    )
    if np.ndim(values_input) > 0:
        shape_values_orthogonal = tuple(
            values_input.shape[i]
            for i in _util._normalize_axis(None, ndim=values_input.ndim)
            if i not in axis_input
        )
    else:
        shape_values_orthogonal = ()

    shape_orthogonal = np.broadcast_shapes(
        shape_input_orthogonal,
        shape_output_orthogonal,
        shape_values_orthogonal,
    )

    axis_input = tuple(sorted(axis_input))
    axis_output = tuple(sorted(axis_output))

    shape_input_new = list(reversed(shape_orthogonal))
    for ax in reversed(axis_input):
        shape_input_new.insert(~ax, shape_input[ax])
    shape_input = tuple(reversed(shape_input_new))

    shape_output_new = list(reversed(shape_orthogonal))
    for ax in reversed(axis_output):
        shape_output_new.insert(~ax, shape_output[ax])
    shape_output = tuple(reversed(shape_output_new))

    weights = np.broadcast_to(np.array(weights), shape_orthogonal, subok=True)
    values_input = np.broadcast_to(values_input, shape_input, subok=True)

    if values_output is None:
        values_output = np.zeros_like(values_input, shape=shape_output, dtype=float)
    else:
        if values_output.shape != shape_output:  # pragma: nocover
            raise ValueError(
                f"{values_output.shape=} should be equal to {shape_output}"
            )
        values_output.fill(0)

    axis_input_numba = ~np.arange(len(axis_input))[::-1]
    axis_output_numba = ~np.arange(len(axis_output))[::-1]

    shape_input_numba = tuple(shape_input[ax] for ax in axis_input)
    shape_output_numba = tuple(shape_output[ax] for ax in axis_output)

    values_input = np.moveaxis(values_input, axis_input, axis_input_numba)
    values_output = np.moveaxis(values_output, axis_output, axis_output_numba)

    shape_output_tmp = values_output.shape

    values_input = values_input.reshape(-1, *shape_input_numba)
    values_output = values_output.reshape(-1, *shape_output_numba)

    weights_list = numba.typed.List()
    for element in weights.reshape(-1):
        weights_list.append(element)

    values_input = np.ascontiguousarray(values_input)
    values_output = np.ascontiguousarray(values_output)

    _regrid_from_weights_arrays_numba(
        weights=weights_list,
        values_input=values_input,
        values_output=values_output,
    )

    values_output = values_output.reshape(*shape_output_tmp)

    values_output = np.moveaxis(values_output, axis_output_numba, axis_output)

    if unit is not None:
        values_output = values_output << unit

    return values_output


_regrid_from_weights_upstream = regridding.regrid_from_weights


def _regrid_from_weights_dispatch(
    weights,
    shape_input,
    shape_output,
    values_input,
    values_output=None,
    axis_input=None,
    axis_output=None,
):
    """
    Dispatch between the upstream and flat-array weight representations.

    Elements that are tuples of flat arrays (produced by
    :func:`_arrays_from_packed`) are applied by
    :func:`_regrid_from_weights_arrays`; anything else falls through to the
    original :func:`regridding.regrid_from_weights`.

    Parameters
    ----------
    weights
        Ragged array of weights.
    shape_input
        Broadcasted shape of the input coordinates.
    shape_output
        Broadcasted shape of the output coordinates.
    values_input
        Input array of values to be resampled.
    values_output
        Optional array in which to place the output.
    axis_input
        Logical axes of the input array to resample.
    axis_output
        Logical axes of the output array corresponding to the resampled axes
        of the input array.
    """
    flat = np.asarray(weights).reshape(-1)
    if flat.size and isinstance(flat[0], tuple):
        return _regrid_from_weights_arrays(
            weights=weights,
            shape_input=shape_input,
            shape_output=shape_output,
            values_input=values_input,
            values_output=values_output,
            axis_input=axis_input,
            axis_output=axis_output,
        )
    return _regrid_from_weights_upstream(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )


if not getattr(regridding.regrid_from_weights, "_mart_caching_patch", False):
    _regrid_from_weights_dispatch._mart_caching_patch = True
    regridding.regrid_from_weights = _regrid_from_weights_dispatch

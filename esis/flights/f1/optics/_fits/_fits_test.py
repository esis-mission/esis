import dataclasses
import numpy as np
import named_arrays as na
import optika
import esis
from . import _fits

_fields = {field.name for field in dataclasses.fields(esis.optics.DistortionParameters)}


def _check_grids(grids: list[dict]):
    assert len(grids) > 0
    for grids_round in grids:
        for key, offsets in grids_round.items():
            names = key if isinstance(key, tuple) else (key,)
            assert set(names) <= _fields
            if not isinstance(offsets, tuple):
                offsets = (offsets,)
            assert len(names) == len(offsets)
            for grid in offsets:
                assert grid.ndim == 1
                assert len(grid) >= 3
                assert np.all(np.diff(grid.value) > 0)
                # every scan must include the current best as a sample
                assert grid.min() <= 0 <= grid.max()


def test_grids_reference():
    _check_grids(_fits._grids_reference())


def test_grids_pointing():
    _check_grids(_fits._grids_pointing())


def test_idealized():
    instrument = esis.flights.f1.optics.as_built(num_distribution=0)
    result = _fits._idealized(instrument)

    assert isinstance(result.grating.material, optika.materials.Mirror)
    assert result.filter.material is None

    # the parameters of the idealized model must pack as plain scalars
    parameters = esis.optics.DistortionParameters.from_instrument(result)
    x = na.pack(parameters).ndarray
    assert np.all(np.isfinite(x))

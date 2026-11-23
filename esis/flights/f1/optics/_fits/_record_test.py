import pathlib
import shutil
import numpy as np
import pytest
import astropy.table
import esis
from esis.flights.f1.optics._fits import _plots, _record

_data = pathlib.Path(esis.__file__).parent / "flights/f1/optics/_instruments/_data"


def _measurement(tilt: float, offset: float, scale: float) -> astropy.table.QTable:
    table = astropy.table.QTable(dict(x=[1.0], y=[0.0]))
    table.meta.update(
        tilt_deg=[tilt, -tilt],
        offset_um=[offset, -offset],
        spread_tilt_deg=0.03,
        spread_offset_um=0.45,
        rotation=[0.0, 0.0],
        scale=[scale, scale],
    )
    return table


def test_field_stop_change():
    result = _record.field_stop_change(_measurement(0.02, 0.5, 1e-4))
    assert result["shape_tilt_deg"] == pytest.approx(0.02)
    assert result["shape_offset_um"] == pytest.approx(0.5)
    assert result["scale"] == [1e-4, 1e-4]


@pytest.mark.parametrize(
    "tilt, offset, scale, expected",
    [
        # the passes of the validation run: settled, then the regular octagon
        (0.007, 0.08, 4e-4, True),
        (0.038, 1.13, -3e-4, True),
        (0.49, 1.20, -3.4e-3, False),
        # the offsets and scales do not end the iteration, the tilts do
        (0.007, 2.0, 2e-3, True),
        (0.08, 0.08, 0.0, False),
    ],
)
def test_field_stop_converged(tilt, offset, scale, expected):
    assert _record.field_stop_converged(_measurement(tilt, offset, scale)) is expected


def test_registration_rms():
    acceptance = astropy.table.QTable.read(_data / "acceptance.ecsv")
    drift = astropy.table.QTable.read(_data / "window_drift.ecsv")
    result = _record.registration_rms(acceptance, drift)
    assert set(result) == {"He I", "O V"}
    assert all(0 < v < 1 for v in result.values())


def _pass(directory: pathlib.Path, tilt: float, stop: bool) -> pathlib.Path:
    directory.mkdir()
    # the committed tables, those of the data release among them
    names = {fname: name for name, fname in _plots._tables.items()}
    for fname in _record.TABLES:
        table = esis.flights.f1.optics.distortion_fit_table(names[fname])
        table.write(directory / fname, format="ascii.ecsv")
    measured = _measurement(tilt, 0.1, 0.0)
    measured.write(directory / "field_stop.ecsv", format="ascii.ecsv")
    if stop:
        shutil.copyfile(_data / "field_stop.ecsv", directory / "field_stop_input.ecsv")
    return directory


def test_record_distortion_fit(tmp_path: pathlib.Path):
    passes = [
        _pass(tmp_path / "pass_0", 0.5, stop=False),
        _pass(tmp_path / "pass_1", 0.005, stop=True),
    ]
    clocking = dict(north_deg=[-0.4] * 4, rotation_deg=-0.38)
    result = _record.record_distortion_fit(
        passes=passes,
        directory=tmp_path / "committed",
        clocking=clocking,
    )
    vertices = astropy.table.QTable.read(_data / "field_stop.ecsv")
    assert np.allclose(result["x"].value, vertices["x"].value)
    written = astropy.table.QTable.read(tmp_path / "committed/field_stop.ecsv")
    history = written.meta["history"]["passes"]
    assert [p["field_stop"] for p in history] == ["regular", "pass_0"]
    assert [p["converged"] for p in history] == [False, True]
    assert len(history[0]["outline_edges_px"]) == 4
    assert written.meta["clocking"] == clocking
    assert written.meta["parker2022"]["crota2_deg"] == _record.PARKER2022["crota2_deg"]
    assert written.meta["history"]["tolerance_tilt_deg"] == 0.05
    for name in _record.TABLES:
        assert (tmp_path / "committed" / name).exists()


def test_record_distortion_fit_one_pass(tmp_path: pathlib.Path):
    with pytest.raises(ValueError):
        _record.record_distortion_fit(
            passes=[tmp_path / "pass_0"], directory=tmp_path / "out", clocking={}
        )


def test_record_distortion_fit_unsettled(tmp_path: pathlib.Path):
    passes = [
        _pass(tmp_path / "pass_0", 0.5, stop=False),
        _pass(tmp_path / "pass_1", 0.3, stop=True),
    ]
    with pytest.raises(ValueError, match="moved"):
        _record.record_distortion_fit(
            passes=passes, directory=tmp_path / "out", clocking={}
        )
    assert not (tmp_path / "out").exists()

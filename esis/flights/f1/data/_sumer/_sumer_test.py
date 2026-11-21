from typing import Any
import pathlib
import hashlib
import pytest
import astropy.time
import named_arrays as na
import esis
from esis.flights.f1.data._sumer import _sumer

_time_test = "2005-04-15T15:43"
"""A time during the observation of the smallest file of the release, 4 MB."""

_name_test = "sum_20050415_153008_l1.tgz"
"""The file which observed :data:`_time_test`."""


def test_manifest() -> None:
    rows = _sumer._manifest()
    assert len(rows) == 178
    begin = [astropy.time.Time(row["begin_date"]) for row in rows]
    assert all(a <= b for a, b in zip(begin, begin[1:]))
    for row in rows:
        assert row["file"].startswith("sum_")
        assert row["file"].endswith("_l1.tgz")
        assert len(row["sha256"]) == 64


@pytest.mark.parametrize("axis_time", ["time", "t"])
def test_path_sumer(axis_time: str) -> None:
    result = esis.flights.f1.data.path_sumer(_time_test, axis_time=axis_time)
    assert isinstance(result, na.ScalarArray)
    assert result.shape == {axis_time: 1}
    path = result.ndarray[0]
    assert isinstance(path, pathlib.Path)
    assert path.name == _name_test
    assert path.is_file()
    (row,) = [row for row in _sumer._manifest() if row["file"] == _name_test]
    assert path.stat().st_size == int(row["size"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize(
    argnames="time_start,time_stop",
    argvalues=[
        ("1996-01-01", "1997-01-01"),
        ("1996-09-16T18:00", "1996-09-16T18:00"),
        ("2005-04-15T15:43", "2005-10-12T00:00"),
    ],
)
def test_path_sumer_time_range(
    monkeypatch: pytest.MonkeyPatch,
    time_start: str,
    time_stop: str,
) -> None:
    """
    The files are those whose observations overlap the time range.

    Each is checked against its own checksum.
    The files are not downloaded: :func:`_retrieve` is replaced by a stand-in
    which returns where each file would be.
    """
    _sumer._manifest()
    calls = []

    def retrieve(**kwargs: Any) -> pathlib.Path:
        calls.append(kwargs)
        return kwargs["path"] / kwargs["fname"]

    monkeypatch.setattr(_sumer, "_retrieve", retrieve)

    start = astropy.time.Time(time_start)
    stop = astropy.time.Time(time_stop)
    result = esis.flights.f1.data.path_sumer(start, stop)

    expected = [
        row
        for row in _sumer._manifest()
        if astropy.time.Time(row["begin_date"]) <= stop
        and astropy.time.Time(row["end_date"]) >= start
    ]
    assert expected
    assert [p.name for p in result.ndarray] == [row["file"] for row in expected]
    assert len(calls) == len(expected)
    for call, row in zip(calls, expected):
        assert call["url"] == f"{_sumer._url}/{row['file']}"
        assert call["known_hash"] == f"sha256:{row['sha256']}"
        assert call["extract_dir"] is None


def test_path_sumer_empty() -> None:
    with pytest.raises(ValueError, match="No file"):
        esis.flights.f1.data.path_sumer("2019-09-30T18:00")

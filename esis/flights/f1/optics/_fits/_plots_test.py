import pytest
import numpy as np
import astropy.table
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import esis  # noqa: E402
from . import _plots  # noqa: E402


@pytest.mark.parametrize("name", sorted(_plots._tables))
def test_distortion_fit_table(name: str):
    result = esis.flights.f1.optics.distortion_fit_table(name)
    assert isinstance(result, astropy.table.QTable)
    assert len(result) > 0


def test_distortion_fit_table_unknown():
    with pytest.raises(ValueError):
        esis.flights.f1.optics.distortion_fit_table("nonsense")


def test_plot_distortion_flight():
    axes = esis.flights.f1.optics.plot_distortion_flight()
    assert axes.shape == (5,)
    # the last panel holds the channels before (dashed) and after (solid)
    # the co-registration, three of each
    styles = [line.get_linestyle() for line in axes[4].get_lines()]
    assert styles.count("--") == 3
    assert styles.count("-") == 3
    plt.close("all")


def test_plot_distortion_flight_axes():
    _, axes = plt.subplots(5, 1)
    result = esis.flights.f1.optics.plot_distortion_flight(axes=axes)
    assert result is not None
    assert np.all(result == axes)
    plt.close("all")


def test_plot_coalignment_tiles():
    axes = esis.flights.f1.optics.plot_coalignment_tiles(frames=(15,))
    # two lines at one frame, three channels against the anchor
    assert axes.shape == (2, 3)
    assert "frame 15" in axes[0, 0].get_title()
    plt.close("all")

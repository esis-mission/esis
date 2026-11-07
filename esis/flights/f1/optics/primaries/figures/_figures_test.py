import numpy as np
import astropy.units as u
import named_arrays as na
import esis


def test_figure_measurement() -> None:
    """
    The map is the height of the surface of the primary mirror.

    It is on a grid of square pixels, about the centroid of the pixels which
    were measured, and it was measured everywhere inside the clear aperture
    except a disk at the center.
    """
    figures = esis.flights.f1.optics.primaries.figures
    figure = figures.figure_measurement(axis_x="a", axis_y="b")
    assert isinstance(figure, na.FunctionArray)

    position = figure.inputs
    height = figure.outputs
    assert isinstance(position, na.Cartesian2dVectorArray)
    assert position.x.shape == dict(a=999)
    assert position.y.shape == dict(b=1004)
    assert height.shape == dict(b=1004, a=999)
    assert height.unit == u.nm

    spacing = 0.18346766654410920 * u.mm
    for component, axis in [(position.x, "a"), (position.y, "b")]:
        step = component[{axis: slice(1, None)}] - component[{axis: slice(None, -1)}]
        assert np.all(np.isclose(step, spacing, rtol=1e-12, atol=0 * u.mm))

    measured = np.isfinite(height)
    x = np.where(measured, position.x, np.nan)
    y = np.where(measured, position.y, np.nan)
    assert np.abs(np.nanmean(x)) < 1e-9 * u.mm
    assert np.abs(np.nanmean(y)) < 1e-9 * u.mm

    design = esis.flights.f1.optics.design(num_distribution=0)
    clear = design.primary_mirror.surface.aperture(
        na.Cartesian3dVectorArray(position.x, position.y, 0 * u.mm)
    )
    unmeasured = clear & ~measured
    assert np.any(unmeasured)
    assert np.all(np.where(unmeasured, position.length, 0 * u.mm) < 4 * u.mm)

    rms = np.sqrt(np.nanmean(np.square(np.where(clear, height, np.nan))))
    assert 15 * u.nm < rms < 17 * u.nm

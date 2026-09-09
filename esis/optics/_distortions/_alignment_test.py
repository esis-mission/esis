import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from . import _alignment


def test_shift_fft():
    rng = np.random.default_rng(0)
    import scipy.ndimage

    a = scipy.ndimage.gaussian_filter(rng.uniform(size=(96, 96)), 2)
    # the convention: b shows at r what a shows at r + s
    b = np.roll(a, (3, -2), axis=(0, 1))
    dx, dy = _alignment.shift_fft(a, b)
    assert np.isclose(dx, -3, atol=0.1)
    assert np.isclose(dy, 2, atol=0.1)


def test_shift_fft_empty():
    a = np.zeros((8, 8))
    assert all(np.isnan(v) for v in _alignment.shift_fft(a, a))


def test_sample_on_sky():
    frame = np.arange(20.0).reshape(4, 5)
    x = np.array([1.0, 1.5, 10.0])
    y = np.array([2.0, 2.5, 0.0])
    result = _alignment.sample_on_sky(frame, x, y)
    assert result[0] == frame[2, 1]
    assert np.isclose(
        result[1], np.mean([frame[2, 1], frame[2, 2], frame[3, 1], frame[3, 2]])
    )
    assert np.isnan(result[2])


def test_measure_shifts():
    rng = np.random.default_rng(1)
    import scipy.ndimage

    a = scipy.ndimage.gaussian_filter(rng.uniform(size=(240, 240)), 2)
    b = np.roll(a, (2, 1), axis=(0, 1))
    result = _alignment.measure_shifts([a, b], num_tile=3, anchor=0)
    assert set(result) == {1}
    rows = np.array(result[1])
    assert len(rows) == 9
    assert np.allclose(rows[:, 2], -2, atol=0.2)
    assert np.allclose(rows[:, 3], -1, atol=0.2)


def test_sky_grid():
    position = na.Cartesian2dVectorLinearSpace(
        start=-10 * u.arcsec,
        stop=10 * u.arcsec,
        axis=na.Cartesian2dVectorArray("a", "b"),
        num=5,
    )
    result = _alignment.sky_grid(position, num=7)
    assert na.shape(result) == dict(sky_x=7, sky_y=7)
    assert result.start.x == -10 * u.arcsec


def test_align_channels():
    """
    Align two copies of a channel, one with a known camera error.

    From a frame the anchor made, the alignment must recover the error to a
    fraction of a pixel.
    """
    from esis.flights.f1.optics._fits._fits import (
        _idealized,
        _wavelength_lines,
        _wavelengths_alignment,
    )
    from ._merit_test import _scene

    instrument = _idealized(esis.flights.f1.optics.design(num_distribution=0))
    instrument.wavelength = _wavelength_lines()
    channel = instrument[dict(channel=1)]
    p_anchor = esis.optics.DistortionParameters.from_instrument(channel)
    p_wrong = na.unpack(na.pack(p_anchor).ndarray.copy(), p_anchor)
    p_wrong.z_sensor = 2 * u.mm
    p_wrong.roll_sensor = 0.05 * u.deg

    # a frame of a smooth random scene, made by the anchor
    scene = _scene(num=81, seed=2)
    merit = esis.optics.LinearMerit(
        channel,
        p_anchor,
        scene,
        observation=0 * scene.outputs.sum(("wavelength", "velocity")),
    )
    frame = merit.image(p_anchor)
    frame = np.asarray(na.value(frame).ndarray_aligned(("detector_y", "detector_x")))

    aligned, medians = esis.optics.align_channels(
        instruments=[channel, channel],
        parameters=[p_anchor, p_wrong],
        frames=[frame, frame],
        scene_position=scene.inputs.position,
        wavelengths=_wavelengths_alignment(),
        anchor=0,
        num_sky=400,
        num_tile=4,
        num_pass=3,
    )
    assert medians[0] == 0
    assert medians[1] < 0.5
    # the anchor is untouched, and the shared optics of the other are too
    assert aligned[0].z_sensor == p_anchor.z_sensor
    assert aligned[1].pitch == p_wrong.pitch
    assert aligned[1].displacement_primary == p_wrong.displacement_primary

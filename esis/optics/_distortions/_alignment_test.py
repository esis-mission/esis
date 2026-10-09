import numpy as np
import scipy.ndimage
import astropy.units as u
import named_arrays as na
import esis
from . import _alignment
from ._merit_test import _scene


def test_shift_fft():
    rng = np.random.default_rng(0)
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
    # optika's convention: pixel i spans [i, i + 1), so (1.5, 2.5) is the
    # centre of pixel [2, 1] and (2.0, 3.0) the corner between four
    frame = np.arange(20.0).reshape(4, 5)
    x = np.array([1.5, 2.0, 10.0])
    y = np.array([2.5, 3.0, 0.0])
    result = _alignment.sample_on_sky(frame, x, y)
    assert result[0] == frame[2, 1]
    assert np.isclose(
        result[1], np.mean([frame[2, 1], frame[2, 2], frame[3, 1], frame[3, 2]])
    )
    assert np.isnan(result[2])


def test_measure_shifts():
    rng = np.random.default_rng(1)
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


def test_window_mask():
    """Mark exactly the pixels inside a square outline, and nothing outside."""
    # an outline through pixel edges: the pixels whose centres, at i + 1/2,
    # lie inside are columns 2 to 5 and rows 1 to 3
    footprint = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([2.0, 6.0, 6.0, 2.0]), axes="vertex"),
        y=na.ScalarArray(np.array([1.0, 1.0, 4.0, 4.0]), axes="vertex"),
    )
    mask = esis.optics.window_mask(footprint, shape=(8, 10))
    assert mask.shape == (8, 10)
    assert mask.sum() == 4 * 3
    assert mask[1:4, 2:6].all()
    assert not mask[0].any() and not mask[:, :2].any()
    assert not mask[4:].any() and not mask[:, 6:].any()


def test_median_shifts():
    """Reduce tile shifts to the length of the median, not the median length."""
    rng = np.random.default_rng(1)
    dx = 0.2 + rng.normal(0, 0.3, 36)
    dy = -0.1 + rng.normal(0, 0.3, 36)
    field = _alignment.ShiftField(584 * u.AA, np.zeros(36), np.zeros(36), dx, dy)
    result = esis.optics.median_shifts({0: [field], 2: []}, scales=[0.5, 0.5])
    assert np.allclose(result[0], 0.5 * np.array([np.median(dx), np.median(dy)]))
    assert np.all(np.isnan(result[2]))


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

    messages = []
    aligned, medians = esis.optics.align_channels(
        log=messages.append,
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
    assert any("pass 0" in m for m in messages)
    assert medians[0] == 0
    assert medians[1] < 0.5
    # the anchor is untouched, and the shared optics of the other are too
    assert aligned[0].z_sensor == p_anchor.z_sensor
    assert aligned[1].pitch == p_wrong.pitch
    assert aligned[1].displacement_primary == p_wrong.displacement_primary


def test_channel_shifts_agree_on_the_pixel_convention():
    """
    Two channels that disperse at right angles see the same sky in the same place.

    Each channel's frame is the model image of one scene, made by the
    regrid onto optika's pixel grid, and read back onto the sky by the
    sampler.  A half-pixel disagreement between the two conventions would
    be a common shift in sensor coordinates, which the rotation between
    the channels turns into a shift between them on the sky of most of a
    pixel; the shift measured must be a small fraction of a pixel.
    """
    from esis.flights.f1.optics._fits._fits import (
        _idealized,
        _wavelength_lines,
        _wavelengths_alignment,
    )

    instrument = _idealized(esis.flights.f1.optics.design(num_distribution=0))
    instrument.wavelength = _wavelength_lines()
    scene = _scene(num=81, seed=3)
    frames, linears = [], []
    for c in (0, 2):
        channel = instrument[dict(channel=c)]
        p = esis.optics.DistortionParameters.from_instrument(channel)
        merit = esis.optics.LinearMerit(
            channel,
            p,
            scene,
            observation=0 * scene.outputs.sum(("wavelength", "velocity")),
        )
        frames.append(
            np.asarray(
                na.value(merit.image(p)).ndarray_aligned(("detector_y", "detector_x"))
            )
        )
        linears.append(merit.linearize(p.to_instrument(channel)))
    # a sky sample of 0.7 px, so the correlation resolves a tenth of a pixel
    sky = esis.optics.sky_grid(scene.inputs.position, 1200)
    fields, scales = esis.optics.measure_channel_shifts(
        linears, frames, sky, _wavelengths_alignment(), anchor=0, num_tile=4
    )
    shift = esis.optics.median_shifts(fields, scales)[1]
    assert np.all(np.isfinite(shift))
    assert np.hypot(*shift) < 0.15


def test_shift_modes():
    """Recover a magnification and a rotation from tile shifts, and call noise noise."""
    num_sky = 401
    half = 0.5 * (num_sky - 1)
    centres = np.linspace(25, 375, 8)
    i, j = np.meshgrid(centres, centres, indexing="ij")
    i, j = i.ravel(), j.ravel()
    u_ = (i - half) / half
    v_ = (j - half) / half
    scale = 2.0  # px per sky sample
    # a translation of 1 sample, a magnification of 0.3 samples at the edge,
    # a rotation of 0.2 samples at the edge
    dx = 1.0 + 0.3 * u_ - 0.2 * v_
    dy = 0.5 + 0.3 * v_ + 0.2 * u_
    field = _alignment.ShiftField(584 * u.AA, i, j, dx, dy)
    modes = esis.optics.shift_modes([field], scale, num_sky)
    assert np.isclose(modes["translation"], np.hypot(1.0, 0.5) * scale, atol=0.05)
    assert np.isclose(modes["magnification"], 0.3 * scale, atol=0.02)
    assert np.isclose(modes["rotation"], 0.2 * scale, atol=0.02)
    assert modes["anisotropy"] < 0.02 and modes["shear"] < 0.02
    assert modes["residual"] < 0.02
    # pure noise: no modes to speak of, and uncorrelated neighbours
    rng = np.random.default_rng(0)
    noise = _alignment.ShiftField(
        584 * u.AA, i, j, rng.normal(0, 0.1, i.size), rng.normal(0, 0.1, i.size)
    )
    modes = esis.optics.shift_modes([noise], scale, num_sky)
    assert modes["magnification"] < 0.1 and modes["rotation"] < 0.1
    assert abs(modes["correlation"]) < 0.3
    assert np.isclose(modes["residual"], modes["scatter"], rtol=0.2)

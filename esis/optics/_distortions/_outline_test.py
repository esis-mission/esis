import dataclasses
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
import esis
from . import _outline
from ._merit_test import merit  # noqa: F401


def _octagon(cx: float, cy: float, radius: float) -> na.Cartesian2dVectorArray:
    """Make a flat-topped octagon of the given vertex radius, as a footprint."""
    phi = np.deg2rad(np.arange(8) * 45 + 22.5)
    return na.Cartesian2dVectorArray(
        x=na.ScalarArray(cx + radius * np.cos(phi), axes="wire") * u.pix,
        y=na.ScalarArray(cy + radius * np.sin(phi), axes="wire") * u.pix,
    )


def _frame(footprint: na.Cartesian2dVectorArray, shape=(400, 600), sigma=1.5):
    """Make a frame lit inside the footprint, with soft edges and a gradient."""
    px, py = _outline._polygon(footprint)
    y, x = np.mgrid[: shape[0], : shape[1]]
    inside = np.ones(shape, dtype=float)
    # the signed distance to each edge of the convex polygon
    distance = np.full(shape, np.inf)
    for k in range(len(px)):
        x0, y0 = px[k], py[k]
        x1, y1 = px[(k + 1) % len(px)], py[(k + 1) % len(px)]
        nx, ny = y1 - y0, x0 - x1
        norm = np.hypot(nx, ny)
        d = ((x - x0) * nx + (y - y0) * ny) / norm
        distance = np.minimum(distance, -d)
    inside = 0.5 * (1 + scipy.special.erf(distance / (np.sqrt(2) * sigma)))
    return 100 * inside * (1 + 0.001 * x) + 10


def test_measure_and_predict_edges():
    footprint = _octagon(300, 200, 120)
    frame = _frame(footprint)
    edges = _outline.measure_edges(frame, [footprint])
    assert len(edges) > 100
    assert set(np.asarray(edges["side"])) == {0, 1, 2, 3}
    model = _outline.predict_edges([footprint], edges)
    residual = model - edges["position"].to_value(u.pix)
    assert np.nanmedian(np.abs(residual)) < 0.1
    assert _outline.outline_residual([footprint], edges) < 0.2
    # a window moved by two pixels along both axes is two pixels off on
    # every side
    moved = _octagon(302, 202, 120)
    assert abs(_outline.outline_residual([moved], edges) - 2) < 0.3
    # and a window far away scores the clip
    assert _outline.outline_residual([_octagon(300, 900, 120)], edges) == 3.0


def test_width_residual():
    footprint = _octagon(300, 200, 120)
    edges = _outline.measure_edges(_frame(footprint), [footprint])
    assert _outline.width_residual([footprint], edges) < 0.2
    # a moved window has the same width, a larger one does not
    assert _outline.width_residual([_octagon(303, 202, 120)], edges) < 0.2
    larger = _outline.width_residual([_octagon(300, 200, 121)], edges)
    assert 1.5 < larger < 2.5
    assert _outline.width_residual([_octagon(900, 900, 120)], edges) == 3.0


def test_edges_of_a_model_image(merit):  # noqa: F811
    # the edges measured on a channel's own image sit on the footprint of
    # the parameters that made it, up to the staircase of the coarse scene
    m, truth = merit
    instrument = m.instrument
    linear = m.linearize(truth.to_instrument(instrument))
    wavelength = instrument.wavelength
    footprints = [
        linear.footprint(wavelength[dict(wavelength=i)])
        for i in range(na.shape(wavelength)["wavelength"])
    ]
    axes = tuple(m.axis_field)[::-1]
    frame = np.asarray(m.observation.ndarray_aligned(axes))
    edges = _outline.measure_edges(frame, footprints)
    assert len(edges) > 100
    residual = _outline.predict_edges(footprints, edges) - edges["position"].to_value(
        u.pix
    )
    assert abs(np.nanmedian(residual)) < 3


class _FakeLinear:
    def __init__(self, shift: float):
        self.shift = shift

    def footprint(self, wavelength):
        return _octagon(300 + self.shift, 200, 120)


class _FakeMerit:
    """
    Slide the windows with the grating yaw and prefer a pitch of two.

    A stand-in for :class:`esis.optics.LinearMerit`, so that the
    alternation can be tested without a raytrace.
    """

    def __init__(self, parameters: esis.optics.DistortionParameters, instrument):
        self.parameters = parameters
        self.instrument = instrument
        self.yaw_design = instrument.grating.yaw
        self.num_calls = 0

    def linearize(self, instrument):
        yaw = (instrument.grating.yaw - self.yaw_design).to_value(u.arcmin)
        return _FakeLinear(shift=4 * yaw)

    def __call__(self, x: np.ndarray) -> float:
        self.num_calls += 1
        p = na.unpack(np.asarray(x), self.parameters)
        return float((p.pitch.to_value(u.arcsec) - 2) ** 2) - 1


def test_fit_distortion_outline():
    instrument = esis.flights.f1.optics.design(num_distribution=0)[dict(channel=1)]
    truth = esis.optics.DistortionParameters.from_instrument(instrument)
    # the windows of the truth, the design, are the octagon at 300 px, and
    # the frame is lit inside it
    frame = _frame(_octagon(300, 200, 120))
    edges = _outline.measure_edges(frame, [_octagon(300, 200, 120)])
    start = dataclasses.replace(
        truth,
        yaw_grating=truth.yaw_grating + 0.5 * u.arcmin,
        pitch=7 * u.arcsec,
    )
    fake = _FakeMerit(start, instrument)
    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(start)
    log = []
    result = _outline.fit_distortion_outline(
        fake,
        edges,
        window=("yaw_grating",),
        sky=("pitch",),
        bounds=(lower, upper),
        num_pass=2,
        maxfev_window=200,
        maxfev_sky=200,
        log=log.append,
    )
    # the window fit brought the grating back, and the sky polish the pitch
    assert abs((result.yaw_grating - truth.yaw_grating).to_value(u.arcmin)) < 0.02
    assert abs(result.pitch.to_value(u.arcsec) - 2) < 0.1
    assert any("outline pass 1" in line for line in log)
    # nothing else moved
    assert result.z_sensor == start.z_sensor

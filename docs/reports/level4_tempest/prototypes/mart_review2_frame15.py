"""
Per-(channel, window) effective-area gain fit on top of the FS support
constraint and pedestal subtraction.

Same tied-doublet configuration as mart_tied_frame15.py, plus:
- the FS octagon polygon (from the instrument's RegularPolygonalAperture,
  orientation selected empirically against the reconstruction);
- weight tables filtered so no triple references a scene cell outside the
  polygon (forward: idx_in; transpose: idx_out) -> outside flux is
  structurally impossible and the weights shrink ~2x;
- the seed zeroed outside (multiplicative MART keeps exact zeros forever).

Usage: python mart_fsmask_frame15.py [--smoke]
The smoke run does 3 iterations and reports NaN/Inf diagnostics.
"""

import pathlib
import sys
import time

sys.path.insert(0, r"C:\Users\t26q518\Documents\git repos\esis\docs\reports")

import astropy.constants
import astropy.units as u
import matplotlib.path
import named_arrays as na
import numpy as np

import ctis
import esis
from esis.data._level_4 import _caching as mart_caching

SMOKE = "--smoke" in sys.argv

OUT = pathlib.Path(
    r"C:\Users\t26q518\Documents\git repos\esis\docs\reports\mart_movie_data_review2"
)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"{stamp} {msg}", flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(f"{stamp} {msg}\n")


log(f"loading instrument (smoke={SMOKE})")
instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
system = instrument.system
key_system = mart_caching.key_system(system)
code_state = mart_caching.code_state()

spectrum = esis.flights.f1.spectrum

RATIO_MGX = 0.52
# O IV 608.40/609.83 photon ratio: both lines decay from the SAME upper
# level (ground-term splitting = the 1.43 A separation), so this is a pure
# branching ratio, rigorously density- and temperature-independent
# (CHIANTI: 0.538 flat over n_e 1e8-1e12 cm-3, T 1e4-1e7 K; see oiv_ratio.py)
RATIO_OIV = 0.538
# members: (wavelength, photon ratio wrt primary, gain column in the
# six-window GAINS matrix measured by mart_eagain_frame15.py)
WINDOWS = [
    (
        "He I 584",
        [(spectrum.He_I.wavelength, 1.0, 0)],
        spectrum.He_I.width_doppler,
        0.70,
    ),
    ("O III 600", [(599.59 * u.AA, 1.0, 1)], 25 * u.km / u.s, 0.13),
    (
        "O IV 608+610",
        [(609.83 * u.AA, 1.0, 3), (608.40 * u.AA, RATIO_OIV, 2)],
        25 * u.km / u.s,
        0.17,
    ),
    (
        "Mg X 610+625",
        [(spectrum.Mg_X.wavelength, 1.0, 4), (624.94 * u.AA, RATIO_MGX, 4)],
        spectrum.Mg_X.width_doppler,
        0.25,
    ),
    (
        "O V 630",
        [(spectrum.O_V.wavelength, 1.0, 5)],
        spectrum.O_V.width_doppler,
        1.00,
    ),
]

num_velocity = 14
velocity = na.linspace(
    start=-210 * u.km / u.s,
    stop=210 * u.km / u.s,
    axis="wavelength",
    num=num_velocity + 1,
)

grids_member = {}
for name, members, _, _ in WINDOWS:
    for wavelength_0, _, _ in members:
        grid = (wavelength_0 * (1 + velocity / astropy.constants.c)).to(u.AA)
        grids_member[float(wavelength_0.to_value(u.AA))] = grid
union = np.sort(
    np.concatenate([g.ndarray.to_value(u.AA) for g in grids_member.values()])
)
wavelength_union = na.ScalarArray(union * u.AA, axes="wavelength")

pitch_scene = 1.5 * u.arcsec
factor_fov = 1.25
field = system.rayfunction_default.inputs.field
center = (field.max() + field.min()) / 2
halfwidth = factor_fov * (field.max() - field.min()) / 2
start, stop = center - halfwidth, center + halfwidth
extent = stop - start
ratio = (np.maximum(extent.x, extent.y) / pitch_scene).ndarray
num_field = int(np.ceil(ratio.to_value(u.dimensionless_unscaled)))
position = na.Cartesian2dVectorLinearSpace(
    start=start,
    stop=stop,
    axis=na.Cartesian2dVectorArray("field_x", "field_y"),
    num=num_field + 1,
)
log(f"num_field={num_field}")

# ---- FS octagon polygon in field coordinates ----
x_vertices_grid = position.x.ndarray.to_value(u.arcsec)
y_vertices_grid = position.y.ndarray.to_value(u.arcsec)
x_cell = (x_vertices_grid[:-1] + x_vertices_grid[1:]) / 2
y_cell = (y_vertices_grid[:-1] + y_vertices_grid[1:]) / 2
XX, YY = np.meshgrid(x_cell, y_cell, indexing="ij")
cx = float(((center.x).ndarray if hasattr(center.x, "ndarray") else center.x).to_value(u.arcsec))
cy = float(((center.y).ndarray if hasattr(center.y, "ndarray") else center.y).to_value(u.arcsec))

half_x = float((extent.x / 2 / factor_fov).ndarray.to_value(u.arcsec))

# reference bright mask from the tied reconstruction, for orientation choice
ref = np.load(
    pathlib.Path(
        r"C:\Users\t26q518\Documents\git repos\esis\docs\reports\mart_movie_data_tied"
    )
    / "frame_15.npz"
)["intensity"][5]
bright = ref > np.percentile(ref, 70)

# the rayfunction field extent is the octagon's edge-to-edge span, so the
# vertex radius is half_x / cos(22.5 deg); phase candidates cover the
# aperture-to-field orientation ambiguity
candidates = {
    "vertex-on-axis small": (half_x, 0.0),
    "edge-on-axis": (half_x / np.cos(np.pi / 8), np.pi / 8),
    "vertex-on-axis": (half_x / np.cos(np.pi / 8), 0.0),
}
best = None
for label, (radius, phase) in candidates.items():
    ang = phase + np.arange(8) * np.pi / 4
    poly = matplotlib.path.Path(
        np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=-1)
    )
    inside = poly.contains_points(
        np.stack([XX.ravel(), YY.ravel()], axis=-1)
    ).reshape(XX.shape)
    score = bright[inside].mean() - bright[~inside].mean()
    log(f"orientation {label}: score {score:.4f}")
    if best is None or score > best[1]:
        best = (label, score, inside)
label_best, _, inside = best
log(f"chosen orientation: {label_best}; inside fraction {inside.mean():.3f}")
inside_flat = inside.ravel()  # (field_x, field_y) C-order, matching flat cell index

args = dict(
    key=key_system,
    wavelength=wavelength_union,
    degree=2,
    code=code_state,
)
linear = mart_caching.linear_system(system, **args)

weights_member = {}
for lam, grid in grids_member.items():
    coords_m = na.SpectralPositionalVectorArray(wavelength=grid, position=position)
    args_m = dict(
        coordinates_scene=coords_m,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        **args,
    )
    fw = mart_caching.weights(system, **args_m)
    tr = mart_caching.weights_transpose(system, **args_m)
    weights_member[lam] = (fw, tr)
log("member weights loaded (cached)")

num_window = len(WINDOWS)
num_cell = num_velocity
num_wavelength = num_window * num_cell

fw0, tr0 = next(iter(weights_member.values()))
num_channel = fw0[1]["channel"]

# ---- frame-transfer-mask shadow: dark leftmost detector columns ----
# the storage-region mask is misaligned (worst in ch 3, ~20 px opaque) and
# shades ACTIVE columns by the He I edge; those pixels read ~0 where the
# model predicts signal, so MART carves dark lanes through the scene.
# Measure the shadow width per channel from the data and remove the pixels
# from the weights, exactly like the FS support constraint.
l1 = esis.flights.f1.data.level_1()
_e0 = l1[dict(time=15)].outputs
_src = [_e0.axes.index(ax) for ax in ("channel", "detector_x", "detector_y")]
_ev = np.moveaxis(np.asarray(_e0.ndarray.value), _src, (0, 1, 2))
num_det_x, num_det_y = _ev.shape[1], _ev.shape[2]
x_shadow = []
for c in range(_ev.shape[0]):
    prof = np.median(_ev[c, :300, 150:900], axis=-1)
    plateau = np.median(prof[50:250])
    below = np.nonzero(prof[:50] < 0.8 * plateau)[0]
    edge = int(below.max()) + 1 if below.size else 0
    x_shadow.append(edge + 5)  # margin past the penumbra
log(f"frame-transfer shadow mask: leftmost columns per channel {x_shadow}")
det_ok = np.ones((_ev.shape[0], num_det_x, num_det_y), dtype=bool)
for c, nxs in enumerate(x_shadow):
    det_ok[c, :nxs, :] = False
det_ok_flat = det_ok.reshape(_ev.shape[0], -1)

n_before = 0
n_after = 0


def filter_forward(el, c):
    global n_before, n_after
    idx_in, idx_out, vals = el
    keep = inside_flat[idx_in] & det_ok_flat[c][idx_out]
    n_before += vals.shape[0]
    n_after += int(keep.sum())
    return (
        np.ascontiguousarray(idx_in[keep]),
        np.ascontiguousarray(idx_out[keep]),
        np.ascontiguousarray(vals[keep]),
    )


def filter_transpose(el, c):
    idx_in, idx_out, vals = el
    keep = inside_flat[idx_out] & det_ok_flat[c][idx_in]
    return (
        np.ascontiguousarray(idx_in[keep]),
        np.ascontiguousarray(idx_out[keep]),
        np.ascontiguousarray(vals[keep]),
    )


arr_fw = np.empty((num_channel, num_wavelength), dtype=object)
arr_tr = np.empty((num_channel, num_wavelength), dtype=object)

# gains measured by mart_eagain_frame15.py (round 3) on the SIX-window layout;
# applied per member via each member's gain column, so the merged O IV window
# keeps both lines' measured photometry
GAINS = np.array(
    [
        [1.0244, 1.0155, 1.0175, 0.9619, 1.0501, 1.0840],
        [1.0248, 1.0330, 0.9689, 0.9725, 0.9492, 0.9219],
        [0.9410, 0.9422, 0.9603, 0.9426, 0.9946, 1.0153],
        [1.0101, 1.0088, 1.0538, 1.1248, 1.0061, 0.9788],
    ]
)

for w, (name, members, _, _) in enumerate(WINDOWS):
    fws, trs, scales, gcols = [], [], [], []
    for wavelength_0, scale, gcol in members:
        fw, tr = weights_member[float(wavelength_0.to_value(u.AA))]
        fws.append(fw[0].ndarray)
        trs.append(tr[0].ndarray)
        scales.append(scale)
        gcols.append(gcol)
    flux = np.array(
        [
            s
            * sum(
                float(f[c, j][2].sum())
                for c in range(num_channel)
                for j in range(num_cell)
            )
            for f, s in zip(fws, scales)
        ]
    )
    alpha = flux / flux.sum()
    for c in range(num_channel):
        scales_c = [s * GAINS[c, g] for s, g in zip(scales, gcols)]
        for j in range(num_cell):
            k = w * num_cell + j
            fw_cat = tuple(
                np.concatenate([f[c, j][i] for f in fws])
                if i < 2
                else np.concatenate(
                    [(s * f[c, j][2]).astype(np.float32) for f, s in zip(fws, scales_c)]
                )
                for i in range(3)
            )
            tr_cat = tuple(
                np.concatenate([t[c, j][i] for t in trs])
                if i < 2
                else np.concatenate(
                    [(a * t[c, j][2]).astype(np.float32) for t, a in zip(trs, alpha)]
                )
                for i in range(3)
            )
            arr_fw[c, k] = filter_forward(fw_cat, c)
            arr_tr[c, k] = filter_transpose(tr_cat, c)

log(f"weights filtered: kept {n_after / n_before:.3f} of forward triples")

shape_in = dict(fw0[1])
shape_in["wavelength"] = num_wavelength
shape_out = dict(fw0[2])
shape_out["wavelength"] = num_wavelength
shape_in_t = dict(tr0[1])
shape_in_t["wavelength"] = num_wavelength
shape_out_t = dict(tr0[2])
shape_out_t["wavelength"] = num_wavelength

weights_full = (
    na.ScalarArray(arr_fw, axes=("channel", "wavelength")),
    shape_in,
    shape_out,
)
weights_transpose_full = (
    na.ScalarArray(arr_tr, axes=("channel", "wavelength")),
    shape_in_t,
    shape_out_t,
)

verts = None
for name, members, _, _ in WINDOWS:
    grid = grids_member[float(members[0][0].to_value(u.AA))].ndarray.to_value(u.AA)
    if verts is None:
        verts = list(grid)
    else:
        verts.extend(verts[-1] + np.cumsum(np.diff(grid)))
wavelength_scene = na.ScalarArray(np.array(verts) * u.AA, axes="wavelength")
coordinates_scene = na.SpectralPositionalVectorArray(
    wavelength=wavelength_scene,
    position=position,
)

l1 = esis.flights.f1.data.level_1()

mart_instrument = ctis.instruments.OptikaInstrument(
    system=linear,
    coordinates_scene=coordinates_scene,
    channel=l1[dict(time=0)].channel,
    axis_channel="channel",
    axis_wavelength="wavelength",
    axis_scene_xy=("field_x", "field_y"),
)
mart_instrument.weights = weights_full
mart_instrument.weights_transpose = weights_transpose_full

velocity_center = (
    velocity[dict(wavelength=slice(None, -1))]
    + velocity[dict(wavelength=slice(1, None))]
) / 2
velocity_center_kms = velocity_center.to(u.km / u.s)

shape_scene = {ax: n for ax, n in shape_in.items() if ax != "channel"}
uniform = na.ScalarArray(np.ones(tuple(shape_scene.values())), axes=tuple(shape_scene))

profile = np.empty(num_wavelength)
for w, (name, _, width, amplitude) in enumerate(WINDOWS):
    g = amplitude * np.exp(-np.square(velocity_center / width) / 2)
    profile[w * num_cell : (w + 1) * num_cell] = np.maximum(np.asarray(g.ndarray), 0.01)
profile = na.ScalarArray(profile, axes=("wavelength",))
support = na.ScalarArray(inside.astype(float), axes=("field_x", "field_y"))
seed = uniform * profile * support


def as_xy(a: na.AbstractScalar) -> np.ndarray:
    src = (a.axes.index("field_x"), a.axes.index("field_y"))
    return np.moveaxis(np.asarray(a.ndarray), src, (0, 1))


t = 15
frame = l1[dict(time=t)]
electrons = np.maximum(frame.outputs, 0)

# ---- pedestal subtraction, paired with the FS support constraint ----
# with the support enforced, the forward model can place NO flux on detector
# pixels outside the in-FS footprint, so the counts there are a direct
# per-channel measurement of the background pedestal (bias/dark residual +
# stray light); subtract it so MART is not asked to explain it
images_raw = na.FunctionArray(
    inputs=mart_instrument.coordinates_sensor,
    outputs=electrons,
)
unit_scene = mart_instrument.backproject(images_raw.outputs).outputs.unit
image_support = mart_instrument.image(seed * unit_scene, noise=False)


def _nd(a, order):
    src = [a.axes.index(ax) for ax in order]
    return np.moveaxis(a.ndarray, src, range(len(order)))


order_det = ("channel", "detector_x", "detector_y")
fp_vals = np.asarray(_nd(image_support.outputs, order_det).value)
footprint = fp_vals > 1e-4 * fp_vals.mean(axis=(-2, -1), keepdims=True)
e_vals = _nd(electrons, order_det)
# background = out-of-footprint AND unshaded (shadow pixels read ~0 and
# would bias the pedestal low)
pedestal = np.stack(
    [
        np.median(e_vals[c][~footprint[c] & det_ok[c]].value)
        for c in range(footprint.shape[0])
    ]
) * e_vals.unit
log(
    "pedestal per channel: "
    f"{np.array2string(pedestal.value, precision=2)} {pedestal.unit} "
    f"(background pixels: {np.array2string((~footprint).mean(axis=(-2, -1)), precision=3)})"
)
electrons = np.maximum(
    electrons - na.ScalarArray(pedestal, axes=("channel",)), 0
)
# zero the shaded pixels: the model can no longer reach them, and their
# penumbral counts would otherwise inflate chi2 with unmodelable signal
electrons = electrons * na.ScalarArray(
    det_ok.astype(float), axes=order_det
)

mean_channel = electrons.mean(("detector_x", "detector_y"))
mean_reference = mean_channel[dict(channel=0)]
factor_norm = mean_reference / mean_channel
electrons = electrons * factor_norm
images = na.FunctionArray(
    inputs=mart_instrument.coordinates_sensor,
    outputs=electrons,
)
guess_base = seed * unit_scene
image_guess = mart_instrument.image(guess_base, noise=False)
scale = images.outputs.sum() / image_guess.outputs.sum()
guess = guess_base * scale



import scipy.ndimage


class _Blur:
    """Per-channel detector-space Gaussian, applied symmetrically."""

    sigma = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def apply(cls, outputs):
        if max(cls.sigma) <= 0 or outputs is None:
            return outputs
        # uncertain arrays (nominal + width): blur the pieces consistently
        if hasattr(outputs, "nominal"):
            import copy

            result = copy.copy(outputs)
            result.nominal = cls.apply(outputs.nominal)
            if hasattr(outputs, "width"):
                result.width = cls.apply(outputs.width)
            return result
        if not hasattr(outputs, "axes") or "detector_x" not in outputs.axes:
            return outputs
        nd = outputs.ndarray
        axes = outputs.axes
        ax_x = axes.index("detector_x")
        ax_y = axes.index("detector_y")
        unit = getattr(nd, "unit", None)
        vals = (nd.value if unit is not None else nd).copy()
        if "channel" in axes:
            ax_c = axes.index("channel")
            view = np.moveaxis(vals, ax_c, 0)
            ax_x_s = ax_x - (1 if ax_c < ax_x else 0)
            ax_y_s = ax_y - (1 if ax_c < ax_y else 0)
            for c in range(view.shape[0]):
                sig = [0.0] * view[c].ndim
                sig[ax_x_s] = cls.sigma[c]
                sig[ax_y_s] = cls.sigma[c]
                view[c] = scipy.ndimage.gaussian_filter(view[c], sigma=sig, mode="nearest")
        else:
            sig = [0.0] * vals.ndim
            sig[ax_x] = max(cls.sigma)
            sig[ax_y] = max(cls.sigma)
            vals = scipy.ndimage.gaussian_filter(vals, sigma=sig, mode="nearest")
        if unit is not None:
            vals = vals << unit
        return na.ScalarArray(vals, axes=axes)


_image_upstream = mart_instrument.image
_backproject_upstream = mart_instrument.backproject


def _image_blurred(scene, **kwargs):
    result = _image_upstream(scene, **kwargs)
    result.outputs = _Blur.apply(result.outputs)
    return result


def _backproject_blurred(image, **kwargs):
    return _backproject_upstream(_Blur.apply(image), **kwargs)


mart_instrument.image = _image_blurred
mart_instrument.backproject = _backproject_blurred

# PSF paused per user direction 2026-07-30: no blur in the forward model
SIGMAS = [(0.0, 0.0, 0.0, 0.0)]
solution = None
for sigma in SIGMAS:
    _Blur.sigma = sigma
    if solution is None:
        guess_base = seed * unit_scene
    else:
        guess_base = solution
    image_guess = mart_instrument.image(guess_base, noise=False)
    scale = images.outputs.sum() / image_guess.outputs.sum()
    guess = guess_base * scale
    mart = ctis.inverters.MartInverter(
        instrument=mart_instrument,
        gamma=1,
        threshold_convergence=1e-2,
        num_iteration=100,
    )
    t0 = time.time()
    inversion = mart(images, guess=guess)
    solution = inversion.solutions[{mart.axis_iteration: ~0}].outputs
    chi2_final = np.asarray(
        inversion.mean_chi_squared[{mart.axis_iteration: ~0}].ndarray
    )
    log(
        f"sigma={sigma}: {inversion.num_iteration:3d} iters, "
        f"chi2 {np.array2string(chi2_final, precision=3)}, {time.time() - t0:4.0f} s"
    )

# ---- save the final configuration's science products ----
maps_I = []
maps_v = []
for w in range(num_window):
    jw = dict(wavelength=slice(w * num_cell, (w + 1) * num_cell))
    radiance = solution[jw]
    intensity = radiance.sum("wavelength")
    vel = (radiance * velocity_center_kms).sum("wavelength") / intensity
    maps_I.append(as_xy(intensity).astype(np.float32))
    maps_v.append(as_xy(vel.to(u.km / u.s)).astype(np.float32))
np.savez(
    OUT / "frame_15.npz",
    intensity=np.stack(maps_I),
    velocity=np.stack(maps_v),
    sigma=np.array(SIGMAS[-1]),
    inside=inside,
)

# ---- residual capture: data vs model at the converged solution ----
image_model = mart_instrument.image(solution, noise=False)
model_out = image_model.outputs
if hasattr(model_out, "nominal"):
    model_out = model_out.nominal
data_out = images.outputs
if hasattr(data_out, "nominal"):
    data_out = data_out.nominal
np.savez(
    OUT / "residual_15.npz",
    data=np.asarray(_nd(data_out, order_det).value),
    model=np.asarray(_nd(model_out, order_det).value),
    footprint=footprint,
    pedestal=pedestal.value,
)
log("residual saved")
log("DONE")

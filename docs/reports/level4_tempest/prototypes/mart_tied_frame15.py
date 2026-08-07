"""
Prototype: tied-line (multiplet) forward model, frame 15 at 1.5 arcsec.

One scene window may have several *members* — spectral lines (or, in future
instruments, diffraction orders) that share the same velocity structure with a
prescribed amplitude ratio.  The forward operator of a group is the sum of its
members' weights tables, member values scaled by the ratio; this works because
the weights elements are per (channel, wavelength-cell) with intra-element
field -> detector indices, so the tie is per-element triple concatenation.

Windows (all 14 x 30 km/s cells, +/-210 km/s):
  0  He I  584.33   free
  1  O III 599.59   free
  2  O IV  608.40   free
  3  O IV  609.83   free      <- explicit blend partner, de-blended by the tie
  4  Mg X  609.79 + 624.94 x 0.52   TIED (theoretical doublet ratio)
  5  O V   629.73   free

The weights are built per member (a side benefit: no seam cells between
windows).  The backprojection of a group is the convex combination of the
members' conservative transposes, weighted by each member's share of the
detected flux, which preserves the normalization scale of the free windows.
"""

import pathlib
import sys
import time

sys.path.insert(0, r"C:\Users\t26q518\Documents\git repos\esis\docs\reports")

import astropy.constants
import astropy.units as u
import named_arrays as na
import numpy as np

import ctis
import esis
import mart_caching

OUT = pathlib.Path(
    r"C:\Users\t26q518\Documents\git repos\esis\docs\reports\mart_movie_data_tied"
)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"{stamp} {msg}", flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(f"{stamp} {msg}\n")


log("loading instrument")
instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
system = instrument.system
key_system = mart_caching.key_system(system)
code_state = mart_caching.code_state()

spectrum = esis.flights.f1.spectrum

RATIO_MGX = 0.52  # Mg X 624.94 / 609.79 photon ratio (2P3/2:2P1/2, Table 1)

# (name, members [(wavelength, forward scale)], seed width, seed amplitude)
WINDOWS = [
    ("He I 584", [(spectrum.He_I.wavelength, 1.0)], spectrum.He_I.width_doppler, 0.70),
    ("O III 600", [(599.59 * u.AA, 1.0)], 25 * u.km / u.s, 0.13),
    ("O IV 608", [(608.40 * u.AA, 1.0)], 25 * u.km / u.s, 0.06),
    ("O IV 610", [(609.83 * u.AA, 1.0)], 25 * u.km / u.s, 0.11),
    (
        "Mg X 610+625",
        [(spectrum.Mg_X.wavelength, 1.0), (624.94 * u.AA, RATIO_MGX)],
        spectrum.Mg_X.width_doppler,
        0.25,
    ),
    ("O V 630", [(spectrum.O_V.wavelength, 1.0)], spectrum.O_V.width_doppler, 1.00),
]

num_velocity = 14
velocity = na.linspace(
    start=-210 * u.km / u.s,
    stop=210 * u.km / u.s,
    axis="wavelength",
    num=num_velocity + 1,
)

# one linearization on the sorted union of every member's vertex grid, so all
# member weights interpolate the same distortion/vignetting models
grids_member = {}
for name, members, _, _ in WINDOWS:
    for wavelength_0, _ in members:
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

args = dict(
    key=key_system,
    wavelength=wavelength_union,
    degree=2,
    code=code_state,
)
linear = mart_caching.linear_system(system, **args)

# ---- per-member weights and transposes (cached individually) ----
weights_member = {}
for lam, grid in grids_member.items():
    coords_m = na.SpectralPositionalVectorArray(wavelength=grid, position=position)
    args_m = dict(
        coordinates_scene=coords_m,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        **args,
    )
    t0 = time.time()
    fw = mart_caching.weights(system, **args_m)
    tr = mart_caching.weights_transpose(system, **args_m)
    weights_member[lam] = (fw, tr)
    log(f"member {lam:.2f}: weights + transpose in {time.time() - t0:4.0f} s")

num_window = len(WINDOWS)
num_cell = num_velocity
num_wavelength = num_window * num_cell

fw0, tr0 = next(iter(weights_member.values()))
num_channel = fw0[1]["channel"]

arr_fw = np.empty((num_channel, num_wavelength), dtype=object)
arr_tr = np.empty((num_channel, num_wavelength), dtype=object)

for w, (name, members, _, _) in enumerate(WINDOWS):
    fws = []
    trs = []
    scales = []
    for wavelength_0, scale in members:
        fw, tr = weights_member[float(wavelength_0.to_value(u.AA))]
        fws.append(fw[0].ndarray)
        trs.append(tr[0].ndarray)
        scales.append(scale)

    # each member's share of the group's detected flux (uniform scene),
    # used to convex-combine the backprojections
    flux = np.array(
        [
            scale * sum(float(fw[c, j][2].sum()) for c in range(num_channel) for j in range(num_cell))
            for fw, scale in zip(fws, scales)
        ]
    )
    alpha = flux / flux.sum()
    if len(members) > 1:
        log(f"window {name}: backprojection flux split {np.array2string(alpha, precision=3)}")

    for c in range(num_channel):
        for j in range(num_cell):
            k = w * num_cell + j
            if len(members) == 1:
                arr_fw[c, k] = fws[0][c, j]
                arr_tr[c, k] = trs[0][c, j]
            else:
                arr_fw[c, k] = tuple(
                    np.concatenate([f[c, j][i] for f in fws])
                    if i < 2
                    else np.concatenate(
                        [
                            (s * f[c, j][2]).astype(np.float32)
                            for f, s in zip(fws, scales)
                        ]
                    )
                    for i in range(3)
                )
                arr_tr[c, k] = tuple(
                    np.concatenate([t[c, j][i] for t in trs])
                    if i < 2
                    else np.concatenate(
                        [
                            (a * t[c, j][2]).astype(np.float32)
                            for t, a in zip(trs, alpha)
                        ]
                    )
                    for i in range(3)
                )

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
log("assembled")

# ---- synthetic contiguous scene wavelength vertices for ctis bookkeeping ----
# cell widths are each window's primary-member physical widths; only the
# widths matter (voxel volumes), the absolute values are bookkeeping
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

shape_scene = {
    ax: n for ax, n in shape_in.items() if ax != mart_instrument.axis_channel
}
uniform = na.ScalarArray(
    np.ones(tuple(shape_scene.values())), axes=tuple(shape_scene)
)

# ---- Gaussian seed (amplitudes from Table 1) ----
velocity_center = (
    velocity[dict(wavelength=slice(None, -1))]
    + velocity[dict(wavelength=slice(1, None))]
) / 2
velocity_center_kms = velocity_center.to(u.km / u.s)

profile = np.empty(num_wavelength)
for w, (name, _, width, amplitude) in enumerate(WINDOWS):
    g = amplitude * np.exp(-np.square(velocity_center / width) / 2)
    profile[w * num_cell : (w + 1) * num_cell] = np.maximum(np.asarray(g.ndarray), 0.01)
profile = na.ScalarArray(profile, axes=("wavelength",))
seed = uniform * profile

# ---- geometry for maps and the Event E cutout ----
x_vertices = coordinates_scene.position.x.ndarray.to_value(u.arcsec)
y_vertices = coordinates_scene.position.y.ndarray.to_value(u.arcsec)
x_center_grid = (x_vertices[:-1] + x_vertices[1:]) / 2
y_center_grid = (y_vertices[:-1] + y_vertices[1:]) / 2

X_EVENT, Y_EVENT, HALF_EVENT = 47.4, -89.0, 40.0
sx = slice(
    int(np.searchsorted(x_center_grid, X_EVENT - HALF_EVENT)),
    int(np.searchsorted(x_center_grid, X_EVENT + HALF_EVENT)) + 1,
)
sy = slice(
    int(np.searchsorted(y_center_grid, Y_EVENT - HALF_EVENT)),
    int(np.searchsorted(y_center_grid, Y_EVENT + HALF_EVENT)) + 1,
)
np.savez(
    OUT / "grid.npz",
    x_center=x_center_grid,
    y_center=y_center_grid,
    velocity_center=velocity_center_kms.ndarray.to_value(u.km / u.s),
    event_slice_x=[sx.start, sx.stop],
    event_slice_y=[sy.start, sy.stop],
    lines=np.array([w[0] for w in WINDOWS]),
    ratio_mgx=RATIO_MGX,
)


def as_xy(a: na.AbstractScalar) -> np.ndarray:
    src = (a.axes.index("field_x"), a.axes.index("field_y"))
    return np.moveaxis(np.asarray(a.ndarray), src, (0, 1))


mart = ctis.inverters.MartInverter(
    instrument=mart_instrument,
    gamma=1,
    threshold_convergence=1e-2,
    num_iteration=100,
)

t = 15
t0 = time.time()
frame = l1[dict(time=t)]
electrons = np.maximum(frame.outputs, 0)
mean_channel = electrons.mean(("detector_x", "detector_y"))
mean_reference = mean_channel[dict(channel=0)]
factor_norm = mean_reference / mean_channel
electrons = electrons * factor_norm
images = na.FunctionArray(
    inputs=mart_instrument.coordinates_sensor,
    outputs=electrons,
)

unit_scene = mart_instrument.backproject(images.outputs).outputs.unit

# ---- sanity: backproject(forward(uniform)) should be O(1) and flat ----
image_u = mart_instrument.image(uniform * unit_scene, noise=False)
b = mart_instrument.backproject(image_u.outputs)
b_val = b.outputs / b.outputs.mean()
for w, (name, _, _, _) in enumerate(WINDOWS):
    jw = dict(wavelength=slice(w * num_cell, (w + 1) * num_cell))
    sub = b_val[jw]
    log(
        f"sanity {name}: backproject(forward(1)) rel. mean "
        f"{float(sub.mean().ndarray):.3f} std {float(sub.std().ndarray):.3f}"
    )

guess_base = seed * unit_scene
image_guess = mart_instrument.image(guess_base, noise=False)
scale = images.outputs.sum() / image_guess.outputs.sum()
guess = guess_base * scale

inversion = mart(images, guess=guess)
solution = inversion.solutions[{mart.axis_iteration: ~0}].outputs

chi2 = inversion.mean_chi_squared
chi2_final = chi2[{mart.axis_iteration: ~0}].ndarray
niter = inversion.num_iteration

maps_I = []
maps_v = []
for w in range(num_window):
    jw = dict(wavelength=slice(w * num_cell, (w + 1) * num_cell))
    radiance = solution[jw]
    intensity = radiance.sum("wavelength")
    vel = (radiance * velocity_center_kms).sum("wavelength") / intensity
    maps_I.append(as_xy(intensity).astype(np.float32))
    maps_v.append(as_xy(vel.to(u.km / u.s)).astype(np.float32))

jw_OV = dict(wavelength=slice(5 * num_cell, 6 * num_cell))
cutout = as_xy(solution[jw_OV])[sx, sy, ...].astype(np.float32)

np.savez(
    OUT / f"frame_{t:02d}.npz",
    intensity=np.stack(maps_I),
    velocity=np.stack(maps_v),
    cutout_OV=cutout,
    chi2_final=np.asarray(chi2_final),
    chi2_history=np.asarray(chi2.ndarray),
    num_iteration=niter,
    factor_norm=np.asarray(factor_norm.ndarray),
    total_signal=float(electrons.sum().ndarray.to_value(u.electron)),
)
log(
    f"frame {t:02d}: {niter:3d} iters, chi2 {np.array2string(np.asarray(chi2_final), precision=2)}, "
    f"{time.time() - t0:5.0f} s"
)
log("DONE")

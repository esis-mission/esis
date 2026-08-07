# ESIS Level-4 MART inversion — reproduction guide

This folder contains everything needed to reproduce the tied-window
MART inversions of the ESIS 2019 flight data whose FITS products and
movies were distributed for review (2026-08). It accompanies the
`feature/mart-level1` branch of https://github.com/esis-mission/esis
(open PR #70).

## What was distributed

- **Per-line FITS files** (`esis_level_4_*.fits`), written by
  `export_fits.py` from the production run
  `gpu_tied_p0.75_nv24_fs1.03.npz`. Each file holds one spectral
  window as a `(time, velocity, y, x)` cube with a full WCS
  (helioprojective arcsec, Doppler km/s, seconds from a reference
  epoch), coregistered onto a common sky frame with the measured scene
  drift removed. A `PHOTONS` extension gives the detected-photon
  equivalent, so S/N ≈ sqrt(PHOTONS). The files are self-describing —
  plain astropy/sunpy is enough to read them.
- **Movies** (intensity, Doppler, event close-ups with AIA context),
  rendered from the same cube by `render_movies.py`.

To read the FITS back into the full Python object (coordinates,
diagnostics, animation methods):

```python
import esis
level_4 = esis.data.Level_4.from_fits("path/to/fits_dir/")
```

`from_fits` accepts the directory of a split product or a single-line
file. This needs the `feature/mart-level1` branch (see below); reading
the files with astropy alone needs nothing.

## The model, in one paragraph

Level-1 electrons from all four channels are inverted simultaneously by
MART into five spectral windows — He I 584, O III 600, O IV 608+610,
Mg X 610+625, O V 630 — on a common (velocity × sky) grid. The O IV
doublet members are tied at a fixed photon ratio of 0.538 and the Mg X
doublet at 0.52, which is what de-blends the overlapping O IV / Mg X
610 Å region. The scene is constrained to the field-stop octagon
(vertex-on-axis, radius inflated 3% to cover the pointing excursion —
`--fs-scale 1.03`, settled by `compare_fs_scan.py`); pixels in the
frame-transfer shadow and a 4-pixel detector border are masked; a
per-frame per-channel pedestal is subtracted and the channels are
normalized to channel 0. The noise model is the exact per-wavelength
one from the optika sensor chain (Fano + gain + read noise). MART runs
per frame from a Gaussian-profile seed, χ² convergence at Δχ² < 1e-2.
EA (effective-area) gains are deliberately omitted.

## Environment

Python ≥ 3.12. Everything except esis itself comes from PyPI:

```bash
python -m venv esis-env && source esis-env/bin/activate
pip install "numpy==2.4.6" "named-arrays==2.4.0" "regridding==3.2.1" \
    "optika==2.1.0" "ctis==0.3.0" "msfc-ccd==1.0.1" \
    "solar-dynamics-observatory==1.0.1" torch
git clone -b feature/mart-level1 https://github.com/esis-mission/esis.git
pip install -e ./esis
```

Notes on versions (verified 2026-08-07):

- **optika 2.1.0** and **ctis 0.3.0** on PyPI are identical to their
  repos' `main` tips — no branch checkouts needed. (An older README in
  this folder pins optika to `feature/interpolated-system` and
  regridding to a SHA; both landed in the releases above. Ignore it.)
- **regridding ≥ 3.2.0** contains the transpose-weights fix (PR #47);
  its effect on the MART results was verified to be negligible, but use
  3.2.1 anyway.
- The line tying is **not** an optika/ctis feature — it is implemented
  in this folder's scripts by concatenating and ratio-scaling the
  per-member weights.
- `torch` needs CUDA for GPU runs; the reference driver also runs on
  CPU (slowly).

Set the esis cache location **before the first import** (weights and
intermediate products land here; see sizes below):

```bash
export ESIS_CACHE_DIR=/path/with/lots/of/space
```

First use downloads the flight FITS via the `esis.flights.f1.data`
loaders and AIA context via JSOC; allow network and some patience.

## Reproducing a run

The scripts must be run from this folder (they import each other as
plain modules).

**1. Build the weights cache.** This is the expensive step: the
linearized instrument and the regridding weights for the seven member
wavelengths. At the production grid (0.75″ pitch, 24 velocity bins)
it is ~1.3 TB of joblib cache and ~106 s/member/frame-chunk on CPU;
`build_weights_parallel.py` builds members in parallel. At a coarse
grid (e.g. `--pitch 1.5 --num-velocity 14`) it is far smaller and is a
sensible first target.

```bash
python build_weights_parallel.py --pitch 1.5 --num-velocity 14
```

**2. Invert.** The portable reference driver (adapted from the
validated `local_mart.py`; streams weights onto one device, never
holding them in host memory):

```bash
python reproduce_level4.py --pitch 1.5 --num-velocity 14          # first light
python reproduce_level4.py --pitch 0.75 --num-velocity 24         # production grid
```

Useful flags: `--device cpu`, `--frames 15` (single frame),
`--coadd` (register + sum the whole flight into one deep frame — good
for faint-line intensity, wrong for velocities, not exportable),
`--out DIR`. The production grid fits on one ~80 GB GPU; halve
`--chunk` if memory is tight.

**3. Export to FITS** (requires an all-frames run):

```bash
python export_fits.py --npz ~/esis_local_runs/reproduce_p0.75_nv24.npz \
    --out ./fits --pitch 0.75 --num-velocity 24
```

**4. Movies** (optional): `render_movies.py` renders drift-corrected
intensity/Doppler/event GIFs from the npz.

## Provenance of the distributed results

The production cube was made on MSU's tempest cluster, one H100,
~10 min for all 30 frames:

```bash
python -u gpu_mart_tied.py --pitch 0.75 --num-velocity 24 \
    --fs-scale 1.03 --index-int32 --single-device
```

`gpu_mart_tied.py` is kept verbatim as the executed production
artifact — it contains tempest-specific paths (its output directory,
and a reference npz from a superseded untied run used only to orient
the field-stop octagon; the settled orientation is hard-coded in
`local_mart.py`/`reproduce_level4.py`, which need no such file). The
esis package state was `feature/mart-level1` @ 2926070.

## Script map

| Script | Role |
|---|---|
| `reproduce_level4.py` | **Portable reference driver** (start here) |
| `tied_config.py` | Window table, ties/ratios, velocity + scene grids |
| `stream_assembly.py` | Streamed tied-weights assembly onto the device |
| `gpu_mart.py` | Noise-probe factors (`_probe_factors`); early untied driver |
| `local_mart.py` | Validated single-GPU driver `reproduce_level4.py` is adapted from |
| `probe_invariance.py` | Coarse-grid noise-probe build + invariance check |
| `build_weights_parallel.py` | Parallel weights-cache build (the expensive step) |
| `gpu_mart_tied.py` | Production driver as run on tempest (verbatim, cluster paths) |
| `export_fits.py` | npz → self-describing Level-4 FITS (+ PHOTONS extension) |
| `render_movies.py` | npz → drift-corrected GIFs |
| `compare_*.py`, `doppler_limits.py`, `plot_pure_maps.py` | Diagnostics used during review |
| `submit_level_4.py`, `build_tied_weights.py`, `gpu_mart_full.py`, `torch_weights.py` | Superseded CPU/two-GPU pipeline, kept for the record |
| `prototypes/` | Frame-15 prototypes the production driver was ported from |

Questions: Jacob Parker (jacobdparker@gmail.com).

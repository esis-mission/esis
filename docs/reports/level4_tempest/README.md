# Parallel Level-4 inversion on tempest

Scripts for computing the time-dependent Level-4 product on MSU's tempest
cluster, one SLURM job per frame.

## One-time setup

```bash
ssh tempest-login.msu.montana.edu
mkdir -p ~/repos ~/venvs && cd ~/repos
git clone -b feature/mart-level1 https://github.com/esis-mission/esis.git
git clone -b feature/interpolated-system https://github.com/sun-data/optika.git
git clone https://github.com/sun-data/ctis.git   # then check out the pinned SHA
python3 -m venv ~/venvs/esis && source ~/venvs/esis/bin/activate
pip install "numpy==2.4.6" "named-arrays==2.3.0" \
    "git+https://github.com/sun-data/regridding@f749f193e"
pip install -e ~/repos/optika
pip install --no-deps -e ~/repos/ctis
pip install -e ~/repos/esis
```

The joblib cache lives on the group allocation (10 TB):
every job exports `ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache`.

## Running

```bash
cd ~/repos/esis/docs/reports/level4_tempest
source ~/venvs/esis/bin/activate

# shared-weights parallel inversion: build -> 30-frame array -> gather
python submit_level_4.py production

# the sequential warm-start chain, for comparison (shares the weights)
python submit_level_4.py compare-chain

# per-frame weights for a subset of frames (each builds ~hundreds of GB
# of weights cache; mind the group quota)
python submit_level_4.py compare-weights --frames 0,7,15,22,29
```

Job DAG for `production`: the build job (fairshare, exclusive node) computes
the Level-1 product, the linearization, and the shared regridding weights,
and inverts the reference frame; the array job (unsafe, preemptable +
requeued, one 1.2 TB job per frame) inverts every frame from the Gaussian
seed; the gather job stacks the cached frames into the full
`esis.data.Level_4` product.

## Comparing

```bash
python compare.py chain                      # parallel seed vs warm chain
python compare.py weights --frames 0,15,29   # shared vs per-frame weights
```

Both print per-frame χ², the cross-correlation shift between the O V
intensity maps, and the mean fractional difference.

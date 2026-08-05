#!/bin/bash
#SBATCH --job-name=l4-movie-both
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --mem=700G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/gpu/movie-both-%j.log

# Both arms of the coalignment over the whole flight, so the comparison can
# be shown as a function of time rather than at three frames.
#
# Sequential in one job on purpose: the runner names its output from the
# configuration tag alone, so two concurrent jobs at the same pitch would
# overwrite each other's cube. Each arm's output is renamed as soon as it
# lands.
#
# The weights of both arms are already in their caches, so neither is
# rebuilt; this is inversion time only.

set -e

source ~/venvs/esis/bin/activate

GPU=~/git_repos/esis-coalign/gpu
OUT=/home/group/charleskankelborg/level4_gpu_tied
TAG=p0.75_nv24_fs1.03

COMMON="--pitch 0.75 --num-velocity 24 --frames all --fs-scale 1.03"
COMMON="$COMMON --index-int32 --single-device --save-residual"

set -x

# --- baseline -----------------------------------------------------------
export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache
cd ~/repos/esis/docs/reports/level4_tempest
python -u gpu_mart_tied.py $COMMON
mv "$OUT/gpu_tied_$TAG.npz" "$GPU/movie_baseline_$TAG.npz"
mv "$OUT/gpu_tied_${TAG}_residual.npz" "$GPU/movie_baseline_${TAG}_residual.npz"

# --- corrected ----------------------------------------------------------
export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache-coalign
cd "$GPU"
python -u run_corrected.py $COMMON
mv "$OUT/gpu_tied_$TAG.npz" "$GPU/movie_corrected_$TAG.npz"
mv "$OUT/gpu_tied_${TAG}_residual.npz" "$GPU/movie_corrected_${TAG}_residual.npz"

#!/bin/bash
#SBATCH --job-name=l4-accept-corr
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --mem=900G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/gpu/accept-corrected-%j.log

# Corrected arm of the coalignment acceptance test. Its own cache, because
# the correction is injected downstream of the joblib key: sharing the
# baseline cache would silently reuse the baseline weights and make the two
# arms identical. The weights are therefore built from cold, which is the
# bulk of this job's wall clock.

source ~/venvs/esis/bin/activate
export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache-coalign

mkdir -p "$ESIS_CACHE_DIR"

cd ~/git_repos/esis-coalign/gpu
set -x
python -u run_corrected.py \
    --pitch 0.75 \
    --num-velocity 24 \
    --frames 14,15,16 \
    --fs-scale 1.03 \
    --index-int32 \
    --single-device

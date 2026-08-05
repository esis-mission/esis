#!/bin/bash
#SBATCH --job-name=l4-accept-base
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --mem=400G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/gpu/accept-baseline-%j.log

# Baseline arm of the coalignment acceptance test, at production scale on
# one GPU. Uses the weights already in the group cache, so nothing is
# rebuilt: only the corrected arm needs a new build.

source ~/venvs/esis/bin/activate
export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache
cd ~/repos/esis/docs/reports/level4_tempest
set -x
python -u gpu_mart_tied.py \
    --pitch 0.75 \
    --num-velocity 24 \
    --frames 14,15,16 \
    --fs-scale 1.03 \
    --index-int32 \
    --single-device

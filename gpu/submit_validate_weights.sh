#!/bin/bash
#SBATCH --job-name=gpuw-validate
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/gpu/validate_weights_%j.out

# Validate the GPU weights build against the CPU build at the production
# scene pitch, on the current stack. The prototype was written against
# optika 2.0 and named-arrays 2.3; both have since been released forward
# (optika 2.1.0 changed the semantics of `image`), so agreement has to be
# re-established before the build is trusted for a production run.

source ~/venvs/esis/bin/activate
export MPLBACKEND=Agg
export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache

cd ~/git_repos/esis-coalign/gpu
python -u torch_weights.py --pitch 0.75 --elements 4

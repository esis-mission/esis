#!/bin/bash
#SBATCH --job-name=mart-accept
#SBATCH --partition=unsafe
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/martfine_%A_%a.out

source ~/venvs/esis/bin/activate
export MPLBACKEND=Agg
export ESIS_CORRECTED=$SLURM_ARRAY_TASK_ID
export ESIS_PITCH_SCENE=1.0
export ESIS_PITCH_VELOCITY=17.5
export ESIS_NUM_ITERATION=100

# the two configurations must not share a joblib cache: the correction is
# injected downstream of the cache key, so a shared cache would let the
# second run reuse the first run's weights and return an identical answer
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache-fine-baseline
else
    export ESIS_CACHE_DIR=/home/group/charleskankelborg/esis-cache-fine-corrected
fi
mkdir -p "$ESIS_CACHE_DIR"

cd ~/git_repos/esis-coalign
python -u mart_acceptance.py

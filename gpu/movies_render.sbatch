#!/bin/bash
#SBATCH --job-name=l4-movie-render
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=unsafe
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/gpu/movie-render-%j.log

# Render the comparison movies from the two saved cubes. No weights and no
# GPU: this only needs the scene grid and the coordinates, so it runs on a
# plain CPU node while the card goes back to the queue.

set -e

source ~/venvs/esis/bin/activate

GPU=~/git_repos/esis-coalign/gpu
TAG=p0.75_nv24_fs1.03

cd "$GPU"
set -x
python -u compare_movies.py \
    --baseline "movie_baseline_$TAG.npz" \
    --corrected "movie_corrected_$TAG.npz" \
    --residual-baseline "movie_baseline_${TAG}_residual.npz" \
    --residual-corrected "movie_corrected_${TAG}_residual.npz" \
    --out ~/git_repos/esis-coalign/docs/reports/channel_coalignment/movies

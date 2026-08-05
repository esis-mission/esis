#!/bin/bash
#SBATCH --job-name=l4-compare
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=unsafe
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/gpu/compare-%j.log

source ~/venvs/esis/bin/activate
cd ~/git_repos/esis-coalign/gpu
python -u compare_acceptance.py

#!/bin/bash
#SBATCH --job-name=coalign-vig
#SBATCH --partition=unsafe
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/home/t26q518/git_repos/esis-coalign/vignetting_%j.out

source ~/venvs/esis/bin/activate
export MPLBACKEND=Agg
cd ~/git_repos/esis-coalign
python -u diagnose_vignetting.py

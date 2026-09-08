#!/bin/bash
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
# Reproduce the committed distortion fit on the cluster.
#
#   sbatch --job-name=dist-channel --array=0-3 reproduce.sbatch channel <dir>
#   sbatch --job-name=dist-combine reproduce.sbatch combine <dir>
#   sbatch --job-name=dist-pointing --array=0-29 reproduce.sbatch pointing <dir>
#   sbatch --job-name=dist-gather reproduce.sbatch gather <dir>
#
# The environment must provide `esis` with the linearized fit, `named_arrays`
# and `optika` with device support, `regridding` >= 3.4 with `torch`, and a
# numba that can see the CUDA driver (see the notes in the report).
COMMAND=$1
DIRECTORY=$2
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export NUMBA_CUDA_LOG_LEVEL=CRITICAL
export ESIS_DEVICE=${ESIS_DEVICE:-cuda}
export ESIS_WORKERS=${ESIS_WORKERS:-6}
source ${ESIS_VENV:-$HOME/venvs/linfit}/bin/activate
export CUDA_HOME=${CUDA_HOME:-$HOME/linfit/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
mkdir -p "$DIRECTORY"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
case "$COMMAND" in
  channel|pointing) python reproduce.py "$COMMAND" "${SLURM_ARRAY_TASK_ID:-0}" "$DIRECTORY" ;;
  combine|gather)   python reproduce.py "$COMMAND" "$DIRECTORY" ;;
  *) echo "unknown command $COMMAND"; exit 1 ;;
esac

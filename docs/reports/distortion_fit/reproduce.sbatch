#!/bin/bash
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
# Reproduce the committed distortion fit on the cluster, in this order:
#
#   sbatch --job-name=dist-channel --array=0-3 reproduce.sbatch channel <dir>
#   sbatch --job-name=dist-edges reproduce.sbatch edges <dir>
#   sbatch --job-name=dist-combine reproduce.sbatch combine <dir>
#   sbatch --job-name=dist-defocus reproduce.sbatch defocus <dir>
#   sbatch --job-name=dist-pointing --array=0-29 reproduce.sbatch pointing <dir>
#   sbatch --job-name=dist-gather reproduce.sbatch gather <dir>
#   sbatch --job-name=dist-accept reproduce.sbatch accept <dir>
#   sbatch --job-name=dist-blink reproduce.sbatch blink <dir>
#   sbatch --job-name=dist-coalign reproduce.sbatch coalign <dir>  # frames from ESIS_FRAMES
#
# and, on a finished reference:
#
#   sbatch --job-name=dist-polish reproduce.sbatch polish <dir>    # from start_reference.ecsv
#   sbatch --job-name=dist-align reproduce.sbatch align <dir>      # alignment only, in place
#   sbatch --job-name=dist-coregister reproduce.sbatch coregister <dir>
#
# The environment is the released libraries with `torch` and a numba that can
# see the CUDA driver (see the report).  Loading the Level-1 frames peaks
# above 100 GB of memory.
COMMAND=$1
DIRECTORY=$2
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export NUMBA_CUDA_LOG_LEVEL=CRITICAL
# "cpu" runs on the host; an unset variable means cuda, an empty one is kept
export ESIS_DEVICE=${ESIS_DEVICE-cuda}
export ESIS_WORKERS=${ESIS_WORKERS:-6}
source ${ESIS_VENV:-$HOME/venvs/linfit}/bin/activate
export CUDA_HOME=${CUDA_HOME:-$HOME/linfit/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
mkdir -p "$DIRECTORY"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
case "$COMMAND" in
  channel|pointing) python reproduce.py "$COMMAND" "${SLURM_ARRAY_TASK_ID:-0}" "$DIRECTORY" ;;
  edges|combine|polish|align|defocus|gather|coregister|accept) python reproduce.py "$COMMAND" "$DIRECTORY" ;;
  blink)            python blink.py render "$DIRECTORY" "$ESIS_DEVICE" ;;
  coalign)          python coalign.py render "$DIRECTORY" ${ESIS_FRAMES:-15} ;;
  *) echo "unknown command $COMMAND"; exit 1 ;;
esac

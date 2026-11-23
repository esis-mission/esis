#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
# One stage of the distortion fit as a Slurm job, an example of running the
# fit on a cluster; chain.sh submits every stage.  The account, partition and
# GPUs are the cluster's, given to sbatch (chain.sh takes them from
# ESIS_SBATCH and ESIS_SBATCH_GPU).  A stage on its own, from this directory:
#
#   sbatch --array=0-3 stage.sbatch channel <dir>
#   sbatch stage.sbatch edges <dir>
#   sbatch --array=0-29 stage.sbatch pointing <dir>
#   sbatch stage.sbatch record <out> <pass>...
#
# Loading the Level-1 frames peaks above 100 GB of memory.  Environment:
# ESIS_VENV, the virtual environment, which must have this checkout installed
# (the job stops if esis imports from elsewhere); CUDA_HOME, if the CUDA
# toolkit numba needs is not on the default paths; and the options of
# reproduce.py (ESIS_DEVICE, ESIS_WORKERS, ESIS_SEED, ESIS_NUM_SCENE).
set -euo pipefail
COMMAND=$1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export NUMBA_CUDA_LOG_LEVEL=CRITICAL
# a private numba cache: ndfilters caches functions whose signature holds a
# dispatcher, and saving to a shared index that already holds another job's
# dispatchers raises ReferenceError
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$NUMBA_CACHE_DIR"
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
# shellcheck disable=SC1091
source "$ESIS_VENV/bin/activate"
if [ -n "${CUDA_HOME:-}" ]; then
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH:-}
fi
# sbatch runs a copy of this script, so find the checkout from where it was
# submitted, which chain.sh makes this directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
ROOT=$(realpath "$(git rev-parse --show-toplevel)")
FOUND=$(python -c "import pathlib, esis; print(pathlib.Path(esis.__file__).resolve().parents[1])")
if [ "$FOUND" != "$ROOT" ]; then
  echo "esis imports from $FOUND, not from this checkout $ROOT;" \
    "set ESIS_VENV to a venv with this checkout installed (pip install -e)" >&2
  exit 2
fi
echo "esis $(git rev-parse HEAD)$(git diff --quiet HEAD -- "$ROOT" || echo ' (modified)') from $ROOT"
case "$COMMAND" in
  channel|pointing) python ../reproduce.py "$COMMAND" "${SLURM_ARRAY_TASK_ID:-0}" "$2" ;;
  next)             bash chain.sh "$2" next "$3" ;;
  *)                python ../reproduce.py "$@" ;;
esac

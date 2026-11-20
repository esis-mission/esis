#!/bin/bash
#SBATCH --account=group-charleskankelborg
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
# One stage of the distortion fit on the cluster.  chain.sh runs the whole
# procedure, iterating the field stop, as one command:
#
#   bash chain.sh <dir>
#
# The stages it runs, each of which can also be submitted on its own:
#
#   sbatch --job-name=dist-channel --array=0-3 reproduce.sbatch channel <dir>
#   sbatch --job-name=dist-edges reproduce.sbatch edges <dir>
#   sbatch --job-name=dist-combine reproduce.sbatch combine <dir>
#   sbatch --job-name=dist-defocus reproduce.sbatch defocus <dir>
#   sbatch --job-name=dist-pointing --array=0-29 reproduce.sbatch pointing <dir>
#   sbatch --job-name=dist-gather reproduce.sbatch gather <dir>
#   sbatch --job-name=dist-fieldstop reproduce.sbatch fieldstop <dir>  # the shape of the field stop
#   sbatch --job-name=dist-converged reproduce.sbatch converged <dir>  # did it move the stop
#   sbatch --job-name=dist-accept reproduce.sbatch accept <dir>
#   sbatch --job-name=dist-coregister reproduce.sbatch coregister <dir>
#   sbatch --job-name=dist-record reproduce.sbatch record <out> <pass>... [--straight <dir>]
#   sbatch --job-name=dist-next reproduce.sbatch next <dir> <pass>     # chain.sh's decision
#
# and, on a finished reference:
#
#   sbatch --job-name=dist-blink reproduce.sbatch blink <dir>
#   sbatch --job-name=dist-coalign reproduce.sbatch coalign <dir>  # frames from ESIS_FRAMES
#   sbatch --job-name=dist-polish reproduce.sbatch polish <dir>    # from start_reference.ecsv
#   sbatch --job-name=dist-align reproduce.sbatch align <dir>      # alignment only, in place
#
# The environment is the released libraries with `torch` and a numba that can
# see the CUDA driver (see the report).  Loading the Level-1 frames peaks
# above 100 GB of memory.
COMMAND=$1
DIRECTORY=$2
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export NUMBA_CUDA_LOG_LEVEL=CRITICAL
# a private numba cache: ndfilters caches functions whose signature holds a
# dispatcher, and saving to a shared index that already holds another job's
# dispatchers raises ReferenceError
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$NUMBA_CACHE_DIR"
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
# "cpu" runs on the host; an unset variable means cuda, an empty one is kept
export ESIS_DEVICE=${ESIS_DEVICE-cuda}
export ESIS_WORKERS=${ESIS_WORKERS:-6}
source ${ESIS_VENV:-$HOME/venvs/linfit}/bin/activate
export CUDA_HOME=${CUDA_HOME:-$HOME/linfit/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
mkdir -p "$DIRECTORY"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# the fit must run the code of this checkout: refuse a venv whose esis is
# installed from somewhere else, and log the commit
ROOT=$(realpath "$(git rev-parse --show-toplevel)")
FOUND=$(python -c "import pathlib, esis; print(pathlib.Path(esis.__file__).resolve().parents[1])")
if [ "$FOUND" != "$ROOT" ]; then
  echo "esis imports from $FOUND, not from this checkout $ROOT;" \
    "set ESIS_VENV to a venv with this checkout installed (pip install -e)" >&2
  exit 2
fi
echo "esis $(git rev-parse HEAD)$(git diff --quiet HEAD -- "$ROOT" || echo ' (modified)') from $ROOT"
case "$COMMAND" in
  channel|pointing) python reproduce.py "$COMMAND" "${SLURM_ARRAY_TASK_ID:-0}" "$DIRECTORY" ;;
  edges|combine|polish|align|defocus|gather|fieldstop|coregister|accept|converged) python reproduce.py "$COMMAND" "$DIRECTORY" ;;
  record)           python reproduce.py record "${@:2}" ;;
  next)             bash chain.sh "$DIRECTORY" next "$3" ;;
  blink)            python blink.py render "$DIRECTORY" "$ESIS_DEVICE" ;;
  coalign)          python coalign.py render "$DIRECTORY" ${ESIS_FRAMES:-15} ;;
  *) echo "unknown command $COMMAND"; exit 1 ;;
esac

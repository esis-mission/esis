#!/bin/bash
# An example of running the distortion fit of ESIS-I on a Slurm cluster, the
# stages of reproduce.py as jobs, with the channels and the frames side by
# side.  On one machine, `python ../reproduce.py run <dir>` does the same in
# one process.  From this directory:
#
#   ESIS_VENV=<venv> ESIS_SBATCH="--account=<a> --partition=<p>" \
#     ESIS_SBATCH_GPU="--gres=gpu:1" ESIS_DEVICE=cuda bash chain.sh <dir>
#
# Each pass is submitted as a chain of dependent jobs; when a pass ends, a
# job decides whether the field stop has converged (reproduce.py converged)
# and submits the next pass, or scores and records the last one into
# <dir>/committed.  Every job's log is beside its output, <dir>/jobs.txt
# lists the jobs, and <dir>/provenance.txt records the commit and the
# installed packages.
#
# Environment: ESIS_VENV (required), the virtual environment, which must have
# this checkout installed; ESIS_SBATCH, options for every job, such as the
# account and partition; ESIS_SBATCH_GPU, options for the jobs that use the
# device, such as the GPU request; ESIS_ACCOUNTS, a comma-separated list of
# accounts to deal the device jobs among, when one account's GPU limit is
# smaller than the work; ESIS_MAX_PASSES (default 4); and the options of
# reproduce.py (ESIS_DEVICE, ESIS_WORKERS, ESIS_SEED, ESIS_NUM_SCENE).
#
# Internal forms, used by the jobs themselves:
#
#   bash chain.sh <dir> pass <k>     submit pass k
#   bash chain.sh <dir> next <k>     after pass k: the next pass, or the end
#   bash chain.sh <dir> final <k>    score and record pass k
set -euo pipefail
shopt -s inherit_errexit
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$(realpath -m "$1")
STEP=${2:-start}
K=${3:-0}
cd "$HERE"
: "${ESIS_VENV:?set ESIS_VENV to a virtual environment with this checkout installed}"
export ESIS_VENV
export ESIS_SEED=${ESIS_SEED:-0}
export ESIS_NUM_SCENE=${ESIS_NUM_SCENE:-401}
export ESIS_DEVICE=${ESIS_DEVICE:-cpu}
export ESIS_WORKERS=${ESIS_WORKERS:-8}
MAX=${ESIS_MAX_PASSES:-4}
read -r -a COMMON <<< "${ESIS_SBATCH:-}"
read -r -a DEVICE <<< "${ESIS_SBATCH_GPU:-}"
IFS=, read -r -a ACCOUNTS <<< "${ESIS_ACCOUNTS:-}"
mkdir -p "$D"

if [ "$STEP" = start ]; then
  # fail now, not in the first job, when the venv would run other code
  # shellcheck disable=SC1091
  source "$ESIS_VENV/bin/activate"
  ROOT=$(realpath "$(git rev-parse --show-toplevel)")
  FOUND=$(python -c "import pathlib, esis; print(pathlib.Path(esis.__file__).resolve().parents[1])")
  if [ "$FOUND" != "$ROOT" ]; then
    echo "esis imports from $FOUND, not from this checkout $ROOT;" \
      "set ESIS_VENV to a venv with this checkout installed (pip install -e)" >&2
    exit 2
  fi
  git diff --quiet HEAD -- "$ROOT" || echo "warning: $ROOT has uncommitted changes" >&2
  {
    git rev-parse HEAD
    git diff --stat HEAD -- "$ROOT"
    env | grep '^ESIS_' | sort
    pip freeze
  } > "$D/provenance.txt"
fi

# submit one stage after the jobs in DEP and print its job id:
#   run <name> <time> <directory> <sbatch options and arguments...>
run() {
  local name=$1 time=$2 dir=$3
  shift 3
  mkdir -p "$dir"
  sbatch --parsable --job-name="$(basename "$D")-$name" --time="$time" \
    "${DEP[@]}" "${COMMON[@]}" --output="$dir/%x_%A_%a.out" --export=ALL "$@"
}

# submit one stage that uses the device, as run does, and print its job ids
# joined by colons, ready for afterok.  With several accounts in
# ESIS_ACCOUNTS, single jobs take them in turn, and an array --array=a-b is
# dealt among them as one array per account over interleaved indices (a-b:n,
# a+1-b:n, ...), each task keeping its own index.
device() {
  local name=$1 time=$2 dir=$3 n=${#ACCOUNTS[@]} turn first="" last="" j ids=""
  local arg args=()
  shift 3
  if [ "$n" = 0 ]; then
    run "$name" "$time" "$dir" "${DEVICE[@]}" "$@"
    return
  fi
  for arg in "$@"; do
    if [[ $arg =~ ^--array=([0-9]+)-([0-9]+)$ ]]; then
      first=${BASH_REMATCH[1]}
      last=${BASH_REMATCH[2]}
    else
      args+=("$arg")
    fi
  done
  if [ -z "$first" ]; then
    turn=$(($(cat "$D/.turn" 2>/dev/null || echo 0) % n))
    echo $(((turn + 1) % n)) > "$D/.turn"
    run "$name" "$time" "$dir" "${DEVICE[@]}" --account="${ACCOUNTS[$turn]}" "$@"
    return
  fi
  for ((j = 0; j < n && first + j <= last; j++)); do
    ids="$ids:$(run "$name" "$time" "$dir" "${DEVICE[@]}" \
      --account="${ACCOUNTS[$j]}" --array="$((first + j))-$last:$n" "${args[@]}")"
  done
  echo "${ids#:}"
}

submit_pass() {
  local k=$1
  local P="$D/pass_$k"
  local H="" E C F T G S N
  DEP=()
  mkdir -p "$P"
  if [ "$k" = 0 ]; then
    H=$(device channel 12:00:00 "$P" --array=0-3 stage.sbatch channel "$P")
    DEP=(--dependency="afterok:$H")
  else
    cp "$D/pass_$((k - 1))/field_stop.ecsv" "$P/field_stop_input.ecsv"
    cp "$D"/pass_0/channel_?.ecsv "$P/"
  fi
  E=$(device edges 04:00:00 "$P" stage.sbatch edges "$P")
  DEP=(--dependency="afterok:$E")
  C=$(device combine 12:00:00 "$P" stage.sbatch combine "$P")
  DEP=(--dependency="afterok:$C")
  F=$(device defocus 04:00:00 "$P" stage.sbatch defocus "$P")
  DEP=(--dependency="afterok:$F")
  T=$(device pointing 08:00:00 "$P" --array=0-29 stage.sbatch pointing "$P")
  DEP=(--dependency="afterok:$T")
  G=$(run gather 00:30:00 "$P" --cpus-per-task=2 --mem=8G stage.sbatch gather "$P")
  DEP=(--dependency="afterok:$G")
  S=$(device fieldstop 04:00:00 "$P" stage.sbatch fieldstop "$P")
  DEP=(--dependency="afterok:$S")
  N=$(run next 00:30:00 "$P" --cpus-per-task=2 --mem=8G stage.sbatch next "$D" "$k")
  echo "pass $k: $H $E $C $F $T $G $S next $N" | tee -a "$D/jobs.txt"
}

submit_final() {
  local k=$1 j A W
  local P="$D/pass_$k"
  local passes=()
  DEP=()
  A=$(device accept 06:00:00 "$P" stage.sbatch accept "$P")
  for ((j = 0; j <= k; j++)); do passes+=("$D/pass_$j"); done
  DEP=(--dependency="afterok:$A")
  W=$(run record 02:00:00 "$D" --mem=64G stage.sbatch record "$D/committed" "${passes[@]}")
  echo "final on pass $k: accept $A record $W" | tee -a "$D/jobs.txt"
}

case "$STEP" in
  start) submit_pass 0 ;;
  pass) submit_pass "$K" ;;
  final) submit_final "$K" ;;
  next)
    # shellcheck disable=SC1091
    source "$ESIS_VENV/bin/activate"
    status=0
    python ../reproduce.py converged "$D/pass_$K" || status=$?
    if [ "$status" = 0 ] && [ "$K" != 0 ]; then
      submit_final "$K"
    elif [ "$status" = 0 ] || [ "$status" = 3 ]; then
      # the fit with the regular octagon never ends the iteration
      if [ $((K + 1)) -ge "$MAX" ]; then
        echo "the field stop did not converge in $MAX passes" >&2
        exit 1
      fi
      submit_pass $((K + 1))
    else
      exit "$status"
    fi
    ;;
  *)
    echo "unknown step $STEP" >&2
    exit 1
    ;;
esac

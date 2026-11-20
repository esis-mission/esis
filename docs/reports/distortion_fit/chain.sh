#!/bin/bash
# Run the distortion fit of ESIS-I end to end on the cluster, as one command:
#
#   bash chain.sh <dir>
#
# The field stop is iterated with the fit.  Pass 0 fits every channel from
# scratch with the regular octagon of the design and measures the field stop
# through that fit; each later pass refits, from the window edges on, with
# the stop the pass before it measured, reusing pass 0's absolute fits, and
# measures it again.  The iteration ends at the first pass whose measurement
# does not move the stop (reproduce.py converged), after at most
# ESIS_MAX_PASSES (default 4) passes.  That pass is the fit: it is scored,
# co-registered, refit with the drift read from the straight edges to compare
# against, and its tables are written into <dir>/committed by
# reproduce.py record, ready to copy into
# esis/flights/f1/optics/_instruments/_data.
#
#   <dir>/pass_<k>/     one pass; field_stop_input.ecsv is the stop it was
#                       fit with (absent in pass 0), field_stop.ecsv the one
#                       it measured
#   <dir>/straight/     the last pass with the straight-edge drift
#   <dir>/committed/    the tables to commit
#   <dir>/jobs.txt      every job submitted
#
# When a pass ends, a job submits the next pass or the final stages, so the
# whole procedure needs only this command.  Every job's log is beside its
# output.  <dir>/provenance.txt records the commit and the installed
# packages.  Environment: ESIS_VENV (the virtual environment, which must have
# this checkout installed; the run stops if esis imports from elsewhere),
# ESIS_PARTITION (default gpufairshare), ESIS_ACCOUNT (the accounts for the
# GPU stages, separated by commas, default the one in reproduce.sbatch; the
# work is spread among them), ESIS_GPU (a GPU type, e.g. a40), ESIS_EXCLUDE (nodes to
# avoid, default the ones whose GPUs are too old for the torch build),
# ESIS_CPU_PARTITION (for the decisions and the record, default unsafe), and
# the ESIS_* settings of reproduce.py.
#
# Internal forms, used by the jobs themselves:
#
#   bash chain.sh <dir> pass <k>     submit pass k
#   bash chain.sh <dir> next <k>     after pass k: the next pass, or the end
#   bash chain.sh <dir> final <k>    the final stages on pass k
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$(realpath -m "$1")
STEP=${2:-start}
K=${3:-0}
cd "$HERE"
export ESIS_SEED=${ESIS_SEED:-0}
export ESIS_NUM_SCENE=${ESIS_NUM_SCENE:-401}
export ESIS_DEVICE=${ESIS_DEVICE-cuda}
export ESIS_WORKERS=${ESIS_WORKERS:-8}
export ESIS_PRIMARY=${ESIS_PRIMARY:-0}
export ESIS_DRIFT_DEGREE=${ESIS_DRIFT_DEGREE:-3}
export ESIS_DEFOCUS_DEGREE=${ESIS_DEFOCUS_DEGREE:-2}
MAX=${ESIS_MAX_PASSES:-4}
GPU="--partition=${ESIS_PARTITION:-gpufairshare} --exclude=${ESIS_EXCLUDE:-gpu[001-015,021-025,053,059-061]}"
if [ -n "${ESIS_GPU:-}" ]; then GPU="$GPU --gres=gpu:$ESIS_GPU:1"; fi
IFS=, read -r -a ACCOUNTS <<< "${ESIS_ACCOUNT:-}"
CPU="--partition=${ESIS_CPU_PARTITION:-unsafe} --gres=none"
mkdir -p "$D"
if [ "$STEP" = start ]; then
  # fail now, not in the first job, when the venv would run other code
  # shellcheck disable=SC1091
  source "${ESIS_VENV:-$HOME/venvs/linfit}/bin/activate"
  ROOT=$(realpath "$(git rev-parse --show-toplevel)")
  FOUND=$(python -c "import pathlib, esis; print(pathlib.Path(esis.__file__).resolve().parents[1])")
  if [ "$FOUND" != "$ROOT" ]; then
    echo "esis imports from $FOUND, not from this checkout $ROOT;" \
      "set ESIS_VENV to a venv with this checkout installed (pip install -e)" >&2
    exit 2
  fi
  git diff --quiet HEAD -- "$ROOT" || echo "warning: $ROOT has uncommitted changes" >&2
  { git rev-parse HEAD; git diff --stat HEAD -- "$ROOT"; pip freeze; } > "$D/provenance.txt"
fi

# submit one stage, after the jobs in $DEP, and print its job id:
#   run <name> <time> <directory> <sbatch options and arguments...>
run() {
  local name=$1 time=$2 dir=$3
  shift 3
  mkdir -p "$dir"
  # shellcheck disable=SC2086
  sbatch --parsable --job-name="$(basename "$D")-$name" --time="$time" $DEP \
    --output="$dir/%x_%A_%a.out" --export=ALL "$@"
}

# submit one stage on the GPUs, as run does, and print its job ids joined by
# colons, ready for afterok.  With several accounts in ESIS_ACCOUNT, whose GPU
# limits add up, single jobs take them in turn, and an array --array=a-b is
# dealt among them as one array per account over interleaved indices (a-b:n,
# a+1-b:n, ...), each task keeping its own index.
gpu() {
  local name=$1 time=$2 dir=$3 n=${#ACCOUNTS[@]} turn first last j ids=""
  local arg args=()
  shift 3
  if [ "$n" = 0 ]; then
    # shellcheck disable=SC2086
    run "$name" "$time" "$dir" $GPU "$@"
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
  if [ -z "${first:-}" ]; then
    turn=$(cat "$D/.turn" 2>/dev/null || echo 0)
    echo $(((turn + 1) % n)) > "$D/.turn"
    # shellcheck disable=SC2086
    run "$name" "$time" "$dir" $GPU --account="${ACCOUNTS[$turn]}" "$@"
    return
  fi
  for ((j = 0; j < n && first + j <= last; j++)); do
    # shellcheck disable=SC2086
    ids="$ids:$(run "$name" "$time" "$dir" $GPU --account="${ACCOUNTS[$j]}" \
      --array="$((first + j))-$last:$n" "${args[@]}")"
  done
  echo "${ids#:}"
}

submit_pass() {
  local k=$1
  local P="$D/pass_$k"
  local H="" E C F T G S N
  DEP=""
  mkdir -p "$P"
  if [ "$k" = 0 ]; then
    export ESIS_FIELD_STOP=regular
    # shellcheck disable=SC2086
    H=$(gpu channel 12:00:00 "$P" --array=0-3 reproduce.sbatch channel "$P")
    DEP="--dependency=afterok:$H"
  else
    cp "$D/pass_$((k - 1))/field_stop.ecsv" "$P/field_stop_input.ecsv"
    cp "$D"/pass_0/channel_?.ecsv "$P/"
    export ESIS_FIELD_STOP="$P/field_stop_input.ecsv"
  fi
  # shellcheck disable=SC2086
  {
    E=$(gpu edges 04:00:00 "$P" reproduce.sbatch edges "$P")
    DEP="--dependency=afterok:$E"
    C=$(gpu combine 12:00:00 "$P" reproduce.sbatch combine "$P")
    DEP="--dependency=afterok:$C"
    F=$(gpu defocus 04:00:00 "$P" reproduce.sbatch defocus "$P")
    DEP="--dependency=afterok:$F"
    T=$(gpu pointing 08:00:00 "$P" --array=0-29 reproduce.sbatch pointing "$P")
    DEP="--dependency=afterok:$T"
    G=$(gpu gather 00:30:00 "$P" reproduce.sbatch gather "$P")
    DEP="--dependency=afterok:$G"
    S=$(gpu fieldstop 04:00:00 "$P" reproduce.sbatch fieldstop "$P")
    DEP="--dependency=afterok:$S"
    N=$(run next 00:30:00 "$P" $CPU --cpus-per-task=2 --mem=8G reproduce.sbatch next "$D" "$k")
  }
  echo "pass $k (field stop $ESIS_FIELD_STOP): $H $E $C $F $T $G $S next $N" | tee -a "$D/jobs.txt"
}

submit_final() {
  local k=$1
  local P="$D/pass_$k" Q="$D/straight"
  local A R AC QE QF QT QG QA W passes
  export ESIS_FIELD_STOP="$P/field_stop_input.ecsv"
  DEP=""
  # shellcheck disable=SC2086
  {
    A=$(gpu accept 06:00:00 "$P" reproduce.sbatch accept "$P")
    R=$(gpu coregister 04:00:00 "$P" reproduce.sbatch coregister "$P")
    DEP="--dependency=afterok:$R"
    AC=$(ESIS_COREGISTERED=1 gpu accept-co 06:00:00 "$P" reproduce.sbatch accept "$P")
    # the same fit with the drift read from the straight edges, to compare
    mkdir -p "$Q"
    cp "$P"/channel_?.ecsv "$P/distortion_reference.ecsv" "$P/field_stop_input.ecsv" "$Q/"
    export ESIS_FIELD_STOP="$Q/field_stop_input.ecsv"
    DEP=""
    QE=$(ESIS_DRIFT=edges gpu edges 04:00:00 "$Q" reproduce.sbatch edges "$Q")
    DEP="--dependency=afterok:$QE"
    QF=$(gpu defocus 04:00:00 "$Q" reproduce.sbatch defocus "$Q")
    DEP="--dependency=afterok:$QF"
    QT=$(gpu pointing 08:00:00 "$Q" --array=0-29 reproduce.sbatch pointing "$Q")
    DEP="--dependency=afterok:$QT"
    QG=$(gpu gather 00:30:00 "$Q" reproduce.sbatch gather "$Q")
    DEP="--dependency=afterok:$QG"
    QA=$(gpu accept 06:00:00 "$Q" reproduce.sbatch accept "$Q")
    export ESIS_FIELD_STOP="$P/field_stop_input.ecsv"
    passes=$(for j in $(seq 0 "$k"); do printf '%s ' "$D/pass_$j"; done)
    DEP="--dependency=afterok:$A:$AC:$QA"
    W=$(run record 02:00:00 "$D" $CPU --cpus-per-task=16 --mem=64G reproduce.sbatch record "$D/committed" $passes --straight "$Q")
  }
  echo "final on pass $k: accept $A coregister $R accept-co $AC straight $QE $QF $QT $QG $QA record $W" | tee -a "$D/jobs.txt"
}

case "$STEP" in
  start) submit_pass 0 ;;
  pass) submit_pass "$K" ;;
  final) submit_final "$K" ;;
  next)
    source "${ESIS_VENV:-$HOME/venvs/linfit}/bin/activate"
    set +e
    python reproduce.py converged "$D/pass_$K"
    status=$?
    set -e
    if [ "$status" = 0 ]; then
      if [ "$K" = 0 ]; then
        echo "the fit with the regular octagon cannot end the iteration"
        exit 1
      fi
      submit_final "$K"
    elif [ "$status" = 3 ]; then
      if [ $((K + 1)) -ge "$MAX" ]; then
        echo "the field stop did not converge in $MAX passes"
        exit 1
      fi
      submit_pass $((K + 1))
    else
      exit "$status"
    fi
    ;;
  *)
    echo "unknown step $STEP"
    exit 1
    ;;
esac

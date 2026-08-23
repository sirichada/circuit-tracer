#!/usr/bin/env bash
# Run tracing scripts sequentially, saving a timestamped log per stage.
# Usage:
#   bash run_tracing_all.sh            # runs 270m, 1b, 4b in order
#   bash run_tracing_all.sh 4b         # runs only 4b
#   bash run_tracing_all.sh 1b 4b      # runs 1b then 4b
# For a long unattended run, launch inside tmux, or prefix with nohup:
#   nohup bash run_tracing_all.sh > run_all.log 2>&1 &

set -uo pipefail  # no -e: one stage failing shouldn't stop the rest

cd "$(dirname "$0")"

LOGDIR="./logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

declare -A SCRIPTS=(
    [270m]="tracing-270m.py"
    [1b]="tracing-1b.py"
    [4b]="tracing-4b.py"
)
DEFAULT_ORDER=(270m 1b 4b)

run_stage() {
    local name="$1"
    local script="$2"
    local log="$LOGDIR/trace-${name}_${TS}.log"
    echo "=== [$name] starting $(date) — log: $log ==="
    python "$script" 2>&1 | tee "$log"
    local status="${PIPESTATUS[0]}"
    echo "=== [$name] finished $(date), exit code $status ==="
    return "$status"
}

stages=("$@")
if [ "${#stages[@]}" -eq 0 ]; then
    stages=("${DEFAULT_ORDER[@]}")
fi

overall_status=0
for name in "${stages[@]}"; do
    script="${SCRIPTS[$name]:-}"
    if [ -z "$script" ]; then
        echo "Unknown stage '$name' — expected one of: ${!SCRIPTS[*]}" >&2
        overall_status=1
        continue
    fi
    if ! run_stage "$name" "$script"; then
        overall_status=1
    fi
done

echo "=== All stages complete $(date), overall exit code $overall_status ==="
exit "$overall_status"

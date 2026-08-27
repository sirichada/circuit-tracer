#!/usr/bin/env bash
# Run threshold_sensitivity.py sequentially per size, saving a timestamped log per stage.
# Usage:
#   bash run_threshold_all.sh            # runs 270m, 1b, 4b in order
#   bash run_threshold_all.sh 4b         # runs only 4b
#   bash run_threshold_all.sh 1b 4b      # runs 1b then 4b

set -uo pipefail  # no -e: one stage failing shouldn't stop the rest

cd "$(dirname "$0")"

LOGDIR="./logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

DEFAULT_ORDER=(270m 1b 4b)

run_stage() {
    local name="$1"
    local log="$LOGDIR/threshold-${name}_${TS}.log"
    echo "=== [$name] starting $(date) — log: $log ==="
    python threshold_sensitivity.py --size "$name" 2>&1 | tee "$log"
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
    if [[ ! " ${DEFAULT_ORDER[*]} " =~ " ${name} " ]]; then
        echo "Unknown stage '$name' — expected one of: ${DEFAULT_ORDER[*]}" >&2
        overall_status=1
        continue
    fi
    if ! run_stage "$name"; then
        overall_status=1
    fi
done

echo "=== All stages complete $(date), overall exit code $overall_status ==="
exit "$overall_status"

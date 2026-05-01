#!/usr/bin/env bash
# Log-growth watchdog (Phase 7 Step 5C).
#
# Runs as a systemd timer every 60 minutes. For each log file under
# /var/log/dungeon-master, compare the current size against the
# size recorded by the previous run. If any file has grown more
# than ${GROWTH_THRESHOLD_BYTES} (default 100 MB), alert.
#
# Use case: detect runaway logs from a stuck retry loop or a
# repeating-token failure mode that's getting through the runaway
# detector. Operator gets paged to look at what's spamming the
# log before it fills the disk.
#
# State file: /var/lib/dungeon-master/watchdog-state/log-growth
# Each line: "<filename> <size-bytes-at-last-tick>"

set -uo pipefail

LOG_DIR="${LOG_DIR:-/var/log/dungeon-master}"
STATE_DIR="${STATE_DIR:-/var/lib/dungeon-master/watchdog-state}"
STATE_FILE="${STATE_FILE:-${STATE_DIR}/log-growth}"
ALERT_HOOK="${ALERT_HOOK:-/opt/dungeon-master/deploy/alerts/notify.sh}"
# 100 MB default — picked from the brief. Tunable via env if
# real-traffic measurement says it needs to move.
GROWTH_THRESHOLD_BYTES="${GROWTH_THRESHOLD_BYTES:-104857600}"

mkdir -p "${STATE_DIR}"

if [[ ! -d "${LOG_DIR}" ]]; then
    # No log directory yet (fresh deploy). Nothing to measure.
    exit 0
fi

# Read previous sizes into an associative array.
declare -A prev_size=()
if [[ -f "${STATE_FILE}" ]]; then
    while read -r line; do
        [[ -z "${line}" ]] && continue
        # Allow filenames with spaces by splitting on the LAST space.
        size="${line##* }"
        name="${line% *}"
        prev_size["${name}"]="${size}"
    done < "${STATE_FILE}"
fi

# Walk current state. ``find -printf`` keeps the format machine-
# parseable even for unusual filenames.
declare -A current_size=()
while IFS=$'\t' read -r name size; do
    current_size["${name}"]="${size}"
done < <(find "${LOG_DIR}" -maxdepth 2 -type f -name '*.log' -printf '%f\t%s\n')

# Compare and alert on growth. Note: we compare against last-tick
# size, not against zero — so logrotate kicking in (file shrinks)
# resets the baseline naturally on the next tick.
for name in "${!current_size[@]}"; do
    cur="${current_size[${name}]}"
    prev="${prev_size[${name}]:-0}"
    delta=$(( cur - prev ))
    if (( delta >= GROWTH_THRESHOLD_BYTES )); then
        delta_mb=$(( delta / 1048576 ))
        if [[ -x "${ALERT_HOOK}" ]]; then
            "${ALERT_HOOK}" warn "log-growth" "${LOG_DIR}/${name} grew ${delta_mb} MB in the last hour" || true
        fi
    fi
done

# Persist current sizes for the next tick. Atomic write.
{
    for name in "${!current_size[@]}"; do
        printf '%s %s\n' "${name}" "${current_size[${name}]}"
    done
} > "${STATE_FILE}.tmp"
mv "${STATE_FILE}.tmp" "${STATE_FILE}"
exit 0

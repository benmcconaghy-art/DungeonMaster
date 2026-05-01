#!/usr/bin/env bash
# Disk-space watchdog (Phase 7 Step 5C).
#
# Runs as a systemd timer every 15 minutes. Reports the used %
# of the filesystem hosting /var/lib/dungeon-master. Alerts on
# state transitions:
#
#   < 80% used: ok
#   80% - 89%:  warn  (alert "approaching capacity")
#   >= 90%:     critical (alert "imminent fill")
#
# State file: /var/lib/dungeon-master/watchdog-state/disk-space
# Single line: previous state ("ok"|"warn"|"critical").

set -uo pipefail

DATA_DIR="${DATA_DIR:-/var/lib/dungeon-master}"
STATE_DIR="${STATE_DIR:-/var/lib/dungeon-master/watchdog-state}"
STATE_FILE="${STATE_FILE:-${STATE_DIR}/disk-space}"
ALERT_HOOK="${ALERT_HOOK:-/opt/dungeon-master/deploy/alerts/notify.sh}"
WARN_THRESHOLD="${WARN_THRESHOLD:-80}"
CRITICAL_THRESHOLD="${CRITICAL_THRESHOLD:-90}"

mkdir -p "${STATE_DIR}"

# df -P: POSIX output, single line per filesystem. The 5th column
# is "Capacity" (the % used). Strip the trailing % sign.
used_pct="$(df -P "${DATA_DIR}" | awk 'NR==2 {gsub("%","",$5); print $5}')"

if [[ -z "${used_pct}" || ! "${used_pct}" =~ ^[0-9]+$ ]]; then
    # df parse failed for some reason; alert at warn so an operator
    # notices but don't update the state file (let next tick retry).
    if [[ -x "${ALERT_HOOK}" ]]; then
        "${ALERT_HOOK}" warn "disk-space" "could not parse df output for ${DATA_DIR}" || true
    fi
    exit 0
fi

if (( used_pct >= CRITICAL_THRESHOLD )); then
    new_state="critical"
elif (( used_pct >= WARN_THRESHOLD )); then
    new_state="warn"
else
    new_state="ok"
fi

prev_state="ok"
[[ -f "${STATE_FILE}" ]] && prev_state="$(cat "${STATE_FILE}")"

# Only alert on transitions, not on every tick. An always-degraded
# disk would spam the alert log with one line per 15 minutes
# otherwise.
if [[ "${new_state}" != "${prev_state}" ]]; then
    case "${new_state}" in
        critical)
            "${ALERT_HOOK}" critical disk-space "${DATA_DIR} at ${used_pct}% (>= ${CRITICAL_THRESHOLD}%); free space immediately" 2>/dev/null || true
            ;;
        warn)
            "${ALERT_HOOK}" warn disk-space "${DATA_DIR} at ${used_pct}% (>= ${WARN_THRESHOLD}%)" 2>/dev/null || true
            ;;
        ok)
            "${ALERT_HOOK}" warn disk-space "${DATA_DIR} recovered to ${used_pct}% (was ${prev_state})" 2>/dev/null || true
            ;;
    esac
fi

echo "${new_state}" > "${STATE_FILE}.tmp"
mv "${STATE_FILE}.tmp" "${STATE_FILE}"
exit 0

#!/usr/bin/env bash
# FLUX deep-health watchdog (Phase 7 Step 5C).
#
# Runs as a systemd timer every 5 minutes. Hits FLUX's /generate
# with a tiny (256x256, 1 step) probe — same shape as the in-
# process worker watchdog, but driven by systemd so it surfaces
# even when the image worker itself is down.
#
# Why two FLUX watchdogs? The in-process one keeps the
# ``image:status`` Valkey key fresh while the worker is alive;
# this one alerts when the SYSTEM is degraded (worker down,
# Valkey down, FLUX down) — the conditions under which the
# in-process watchdog stops emitting.
#
# State machine:
#   - On success: write "ok" to state file. If previous state was
#     "degraded", transition back and alert at "warn" (recovered).
#   - On failure: append to a failure log. If failures span
#     ${DEGRADED_THRESHOLD_S} or more (default 120s), set state to
#     "degraded" and alert at "critical".
#
# State file: /var/lib/dungeon-master/watchdog-state/flux-health
# (single line: "<status>:<since-iso>")

set -uo pipefail

FLUX_BASE_URL="${FLUX_BASE_URL:-http://svrai01.mcconaghygroup.internal:11437}"
STATE_DIR="${STATE_DIR:-/var/lib/dungeon-master/watchdog-state}"
STATE_FILE="${STATE_FILE:-${STATE_DIR}/flux-health}"
ALERT_HOOK="${ALERT_HOOK:-/opt/dungeon-master/deploy/alerts/notify.sh}"
# Sustained-failure window before flipping to degraded. Matches the
# in-process worker's DEGRADED_THRESHOLD_S so the two watchdogs agree
# on what counts as a real outage vs a transient blip.
DEGRADED_THRESHOLD_S="${DEGRADED_THRESHOLD_S:-120}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-30}"

mkdir -p "${STATE_DIR}"

now_epoch() { date -u +%s; }
now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

read_state() {
    if [[ -f "${STATE_FILE}" ]]; then
        cat "${STATE_FILE}"
    else
        echo "ok:never"
    fi
}

write_state() {
    printf '%s:%s\n' "$1" "$2" > "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "${STATE_FILE}"
}

alert() {
    local severity="$1"; local message="$2"
    if [[ -x "${ALERT_HOOK}" ]]; then
        "${ALERT_HOOK}" "${severity}" "flux-health" "${message}" || true
    fi
}

# ---------- probe -------------------------------------------------------------

probe_payload='{"prompt":"watchdog probe","width":256,"height":256,"num_inference_steps":1}'

# Use --max-time to bound wall clock. -f makes curl exit non-zero on
# HTTP 4xx/5xx. We don't read the response body — the exit code is
# the signal.
probe_ok=0
if curl -fsS -X POST \
        --max-time "${PROBE_TIMEOUT_S}" \
        -H "Content-Type: application/json" \
        -d "${probe_payload}" \
        -o /dev/null \
        "${FLUX_BASE_URL}/generate"; then
    probe_ok=1
fi

# ---------- state transitions -------------------------------------------------

state="$(read_state)"
prev_status="${state%%:*}"
prev_since="${state#*:}"
now_iso_str="$(now_iso)"

if (( probe_ok == 1 )); then
    if [[ "${prev_status}" == "degraded" ]]; then
        alert warn "FLUX recovered after $(date -u -d "${prev_since}" +%s 2>/dev/null || echo unknown) baseline"
        write_state ok "${now_iso_str}"
    else
        write_state ok "${prev_since}"
    fi
    exit 0
fi

# Probe failed. Decide whether to degrade.
case "${prev_status}" in
    ok)
        # First failure: record the timestamp but don't alert yet —
        # let DEGRADED_THRESHOLD_S elapse so transient blips don't page.
        write_state failing "${now_iso_str}"
        ;;
    failing)
        # Sustained: check elapsed time since first failure.
        first_failure_epoch=$(date -u -d "${prev_since}" +%s 2>/dev/null || echo "$(now_epoch)")
        elapsed=$(( $(now_epoch) - first_failure_epoch ))
        if (( elapsed >= DEGRADED_THRESHOLD_S )); then
            alert critical "FLUX probe has been failing for ${elapsed}s (>= ${DEGRADED_THRESHOLD_S}s threshold)"
            write_state degraded "${prev_since}"
        else
            write_state failing "${prev_since}"
        fi
        ;;
    degraded)
        # Already degraded; don't re-alert on every tick.
        write_state degraded "${prev_since}"
        ;;
    *)
        # Unknown state — reset to failing and start the clock.
        write_state failing "${now_iso_str}"
        ;;
esac

# Watchdogs always exit 0 — non-zero would cause systemd to mark the
# service Failed and stop calling it. The state file + alert hook are
# the channels that matter.
exit 0

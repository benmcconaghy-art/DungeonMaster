#!/usr/bin/env bash
# Dungeon Master alerting hook.
#
# Phase 7 hardening (Step 4C). Watchdogs (FLUX-health, disk-space,
# log-growth) call this script when state transitions to "degraded"
# or "critical". Default behaviour: write to syslog + a dedicated
# log file. Operators extend later for Slack / PagerDuty / email.
#
# Contract:
#   notify.sh <severity> <component> <message>
#
#   severity   "warn" | "critical"
#   component  short name of the watchdog: "flux", "disk", "log-growth", ...
#   message    free-form description (anything after $2 is concatenated)
#
# Exit code: ALWAYS 0. The watchdogs must not be perturbed by an
# alert-delivery failure. If syslog is broken or the log dir isn't
# writable, that's an independent problem; we don't compound it by
# making the watchdog crash too.
#
# This script is intentionally short and dependency-free. The whole
# point is that an operator can read it in under a minute, see what
# it does, and bolt their own delivery channel onto the bottom block.

set -u  # not -e — we want best-effort execution, not failure propagation
# shellcheck disable=SC2034  # PIPESTATUS is set by `set -o pipefail`
set -o pipefail

# ---------- arg parsing -------------------------------------------------------

if [[ $# -lt 3 ]]; then
    # Don't return non-zero — the watchdog calling us mis-built the
    # call, which is a code bug we want logged but not propagated.
    logger -t dungeon-master-alert -p user.error \
        "notify.sh called with $# args, expected at least 3 (severity, component, message)"
    exit 0
fi

severity="$1"
component="$2"
shift 2
# Concatenate the remaining args into a single message string,
# preserving spaces.
message="$*"

# Normalise + validate severity. Anything other than the two known
# values gets silently coerced to ``warn`` so an unknown severity
# doesn't drop the alert entirely.
case "${severity}" in
    warn|critical) : ;;
    *)
        logger -t dungeon-master-alert -p user.error \
            "notify.sh got unknown severity ${severity}; coercing to warn"
        severity="warn"
        ;;
esac

# Map severity → syslog priority. ``critical`` → user.crit (high
# urgency); ``warn`` → user.warning. Both land under the same syslog
# tag so an operator's ``journalctl -t dungeon-master-alert`` shows
# the full alert history.
case "${severity}" in
    critical) priority="user.crit"   ;;
    warn)     priority="user.warning" ;;
esac

ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
formatted="[${severity}] [${component}] ${message}"

# ---------- 1. syslog ---------------------------------------------------------

# logger is part of util-linux; present on AlmaLinux out of the box.
logger -t dungeon-master-alert -p "${priority}" -- "${formatted}" || true

# ---------- 2. dedicated log file ---------------------------------------------

# /var/log/dungeon-master is created (and SELinux-labelled) by
# bootstrap.sh + deploy/selinux/install.sh. If something has gone
# sideways and the directory isn't writable, fall back to /tmp so
# the alert at least lands somewhere greppable.
log_dir="/var/log/dungeon-master"
if [[ ! -w "${log_dir}" ]]; then
    log_dir="/tmp"
fi
log_file="${log_dir}/alerts.log"

printf '%s %s\n' "${ts}" "${formatted}" >>"${log_file}" 2>/dev/null || true

# ---------- 3. operator extension point ---------------------------------------

# Add Slack / email / PagerDuty / etc. delivery here when an operator
# wants real-time notification. Keep the additions short and resilient
# (set timeouts; ``|| true`` on every external call) so a network
# blip doesn't wedge the alert path.
#
# Example sketch:
#
#   if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
#       curl --max-time 5 -sS -X POST -H 'Content-Type: application/json' \
#            -d "{\"text\":\"${formatted}\"}" \
#            "${SLACK_WEBHOOK_URL}" >/dev/null 2>&1 || true
#   fi

exit 0

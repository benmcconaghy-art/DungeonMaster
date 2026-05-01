#!/usr/bin/env bash
# Nightly SQLite backup with integrity check + retention rotation.
#
# Phase 7 hardening (Step 5A). Cron runs this at 02:00 every day as
# the ``dungeonmaster`` user. The .backup pattern is per spec §13.
#
# What this script does, in order:
#   1. Take a coordinated SQLite snapshot via the ``.backup`` SQL command
#      (NOT a raw filesystem copy — that races with WAL writes and can
#      produce a corrupt copy).
#   2. Run ``PRAGMA integrity_check`` against the snapshot. If the
#      result is anything other than "ok" lines, alert and KEEP the
#      previous good backup intact.
#   3. Rotate retention: keep the last 8 daily snapshots, the most
#      recent 12 Sunday snapshots ("weekly"), and the most recent 6
#      first-of-month snapshots ("monthly"). The naming scheme makes
#      this purely a filename-pattern decision.
#
# Backups are local-only — no S3, no rsync to a remote, no cloud target
# (the brief is explicit). An operator who wants off-box copies can
# rsync /var/backups/dm to wherever they want; that's their concern.
#
# Exits non-zero on backup or integrity-check failure so a cron-
# monitoring tool flags it. Rotation failures log + alert but don't
# fail the run — better a slightly-too-old backup than a missed run.

set -euo pipefail

# ---------- config -----------------------------------------------------------

DB_PATH="${DB_PATH:-/var/lib/dungeon-master/dm.db}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/dm}"
ALERT_HOOK="${ALERT_HOOK:-/opt/dungeon-master/deploy/alerts/notify.sh}"
KEEP_DAILY="${KEEP_DAILY:-8}"
KEEP_WEEKLY="${KEEP_WEEKLY:-12}"
KEEP_MONTHLY="${KEEP_MONTHLY:-6}"

# ---------- helpers ----------------------------------------------------------

log() { printf '[dm-backup] %s\n' "$*"; }

alert() {
    local severity="$1"
    local message="$2"
    if [[ -x "${ALERT_HOOK}" ]]; then
        "${ALERT_HOOK}" "${severity}" "backup" "${message}" || true
    else
        # Fall back to logger so the alert is at least in syslog.
        logger -t dungeon-master-alert -p user.warning -- "[backup] ${message}" || true
    fi
}

# ---------- preflight --------------------------------------------------------

if [[ ! -r "${DB_PATH}" ]]; then
    alert critical "DB file not readable: ${DB_PATH}"
    log "FATAL: ${DB_PATH} not readable"
    exit 1
fi

mkdir -p "${BACKUP_DIR}"

today="$(date -u +%Y-%m-%d)"
target="${BACKUP_DIR}/dm-${today}.db"
tmp_target="${target}.tmp"

# ---------- 1. take the snapshot ---------------------------------------------

log "snapshotting ${DB_PATH} -> ${target}"
# .backup is the only safe live-DB copy method. Run it under sqlite3's
# CLI so it commits a coordinated WAL checkpoint before producing the
# output file. Output goes to a .tmp first; if anything fails the
# previous good snapshot stays intact.
if ! sqlite3 "${DB_PATH}" ".backup '${tmp_target}'"; then
    alert critical "sqlite3 .backup failed for ${DB_PATH}"
    rm -f "${tmp_target}"
    exit 2
fi

# ---------- 2. integrity-check the copy --------------------------------------

log "integrity-checking ${tmp_target}"
# Capture the result so we can both decide and log on failure. The
# expected output for a clean DB is the single line "ok"; anything
# else (corruption, missing rows, etc.) is a fail.
if ! integrity_result="$(sqlite3 "${tmp_target}" "PRAGMA integrity_check")"; then
    alert critical "integrity check command failed against ${tmp_target}"
    rm -f "${tmp_target}"
    exit 3
fi
if [[ "${integrity_result}" != "ok" ]]; then
    alert critical "integrity check FAILED for ${tmp_target}: ${integrity_result}"
    log "integrity check failed; preserving previous good snapshot, dropping ${tmp_target}"
    rm -f "${tmp_target}"
    exit 4
fi

# Atomic rename only after integrity check passes. The previous good
# snapshot at ${target} (from a prior run on the same date) is
# overwritten only here, AFTER the new copy is verified.
mv "${tmp_target}" "${target}"
log "snapshot complete: ${target}"

# ---------- 3. retention rotation --------------------------------------------

# Daily: keep the most recent KEEP_DAILY by filename mtime; drop older.
# Weekly: a snapshot is considered "weekly" if its date is a Sunday.
# Monthly: a snapshot is "monthly" if its date is the first of the month.
#
# All three are computed from the filename (dm-YYYY-MM-DD.db) so the
# logic is purely calendar-based; mtime drift doesn't affect it.

list_backups() {
    # Newest first. Stable: lexical sort on YYYY-MM-DD is chronological.
    find "${BACKUP_DIR}" -maxdepth 1 -name 'dm-*.db' -type f -printf '%f\n' \
        | sort -r
}

is_sunday() {
    # $1 is YYYY-MM-DD; date +%u gives 1-7 (Mon-Sun).
    [[ "$(date -u -d "$1" +%u)" == "7" ]]
}

is_first_of_month() {
    [[ "$1" == *-01 ]]
}

# Bucket counts only increment when we add a *fresh* file. Without
# this dedup, a date that satisfies multiple buckets (e.g. a 1st-of-
# month that's also today's daily) burns a slot in each bucket — and
# the oldest entry of the lower-priority bucket gets evicted to make
# room for an entry that was already going to be kept anyway.
declare -A keep_map=()

add_kept() {
    if [[ -z "${keep_map[$1]:-}" ]]; then
        keep_map["$1"]=1
        return 0  # added fresh
    fi
    return 1  # already kept by an earlier bucket
}

# Always keep the N most recent dailies.
daily_kept=0
while read -r fname; do
    if (( daily_kept < KEEP_DAILY )) && add_kept "${fname}"; then
        daily_kept=$(( daily_kept + 1 ))
    fi
done < <(list_backups)

# Weekly: walk newest → oldest, accept up to KEEP_WEEKLY Sunday files
# that aren't already kept by the daily bucket.
weekly_kept=0
while read -r fname; do
    [[ "${fname}" =~ dm-([0-9]{4}-[0-9]{2}-[0-9]{2})\.db ]] || continue
    d="${BASH_REMATCH[1]}"
    if is_sunday "${d}" && (( weekly_kept < KEEP_WEEKLY )) && add_kept "${fname}"; then
        weekly_kept=$(( weekly_kept + 1 ))
    fi
done < <(list_backups)

# Monthly: walk newest → oldest, accept up to KEEP_MONTHLY first-of-
# month files that aren't already kept by daily or weekly buckets.
monthly_kept=0
while read -r fname; do
    [[ "${fname}" =~ dm-([0-9]{4}-[0-9]{2}-[0-9]{2})\.db ]] || continue
    d="${BASH_REMATCH[1]}"
    if is_first_of_month "${d}" && (( monthly_kept < KEEP_MONTHLY )) && add_kept "${fname}"; then
        monthly_kept=$(( monthly_kept + 1 ))
    fi
done < <(list_backups)

dropped=0
while read -r fname; do
    if [[ -z "${keep_map[${fname}]:-}" ]]; then
        log "rotating out: ${fname}"
        if ! rm -f "${BACKUP_DIR}/${fname}"; then
            alert warn "rotation rm failed for ${fname}"
        else
            dropped=$(( dropped + 1 ))
        fi
    fi
done < <(list_backups)

log "retention: kept ${#keep_map[@]}, dropped ${dropped}"
log "done."
exit 0

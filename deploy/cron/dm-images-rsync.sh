#!/usr/bin/env bash
# Nightly rsync of the generated-image directory.
#
# Phase 7 hardening (Step 5A). Runs at 03:00 every day as the
# ``dungeonmaster`` user, an hour after the SQLite backup so the
# disk and CPU burst from the .backup is finished.
#
# Why rsync rather than a snapshot?
#   - The image directory is purely append-only files; nothing in the
#     app rewrites a generated PNG once it's written. ``rsync -a``
#     with --delta is the idiomatic delta-friendly copy and survives
#     interruptions (resumable on next run).
#   - Removing files: spec §13 doesn't have a delete path for old
#     images, but if one is added later, ``--delete`` can be flipped
#     on. Today we DON'T delete from the backup — orphaned PNGs in
#     the backup directory are cheap.
#
# Local-only by default. An operator who wants off-box copies can
# add a second rsync stanza below.

set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:-/var/lib/dungeon-master/images/}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/dm/images/}"
ALERT_HOOK="${ALERT_HOOK:-/opt/dungeon-master/deploy/alerts/notify.sh}"

log() { printf '[dm-images-rsync] %s\n' "$*"; }

if [[ ! -d "${IMAGE_DIR}" ]]; then
    log "image dir ${IMAGE_DIR} does not exist; nothing to back up"
    exit 0
fi

mkdir -p "${BACKUP_DIR}"

log "rsyncing ${IMAGE_DIR} -> ${BACKUP_DIR}"
# -a: archive (perms, timestamps, etc.). --partial: keep partial
# transfers between runs so a 2GB image isn't re-fetched if cron
# missed a tick. Quiet by default; --stats added for audit log.
if ! rsync -a --partial --stats "${IMAGE_DIR}" "${BACKUP_DIR}"; then
    if [[ -x "${ALERT_HOOK}" ]]; then
        "${ALERT_HOOK}" warn images-rsync "rsync ${IMAGE_DIR} -> ${BACKUP_DIR} failed" || true
    fi
    log "rsync failed"
    exit 1
fi

log "done."

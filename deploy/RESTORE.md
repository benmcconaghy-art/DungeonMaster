# Dungeon Master — restore procedure

Phase 7 hardening (Step 5B). One-page playbook for restoring the
SQLite database from a nightly backup. Print this and tape it
above the deploy box if you're feeling thorough.

## Backup layout

```
/var/backups/dm/
  dm-YYYY-MM-DD.db        # nightly snapshots (8 daily, 12 Sunday, 6 1st-of-month)
  images/                 # rsynced PNGs (delta, append-only)
```

`dm-backup.sh` runs at 02:00 UTC; `dm-images-rsync.sh` at 03:00 UTC.
Backups are local-only by default. If the host disk is the failure
domain, restore from off-box copies first (operator's responsibility).

## Restore steps

1. **Stop the services** so nothing writes during restore:

   ```bash
   sudo systemctl stop dungeon-master.service dungeon-master-imageworker.service
   ```

2. **Pick the backup** to restore. List candidates:

   ```bash
   ls -lh /var/backups/dm/dm-*.db
   ```

   Most recent first; the file's mtime matches the snapshot date.

3. **Verify the backup** before clobbering the live DB:

   ```bash
   sqlite3 /var/backups/dm/dm-2026-05-01.db "PRAGMA integrity_check"
   # Expected output: ok
   ```

   If anything other than `ok`, pick an older backup and try again.

4. **Move the live DB out of the way** (don't delete — keep it for
   forensics if the corruption isn't yet understood):

   ```bash
   sudo -u dungeonmaster mv /var/lib/dungeon-master/dm.db \
                            /var/lib/dungeon-master/dm.db.broken-$(date -u +%FT%H%M%SZ)
   sudo -u dungeonmaster rm -f /var/lib/dungeon-master/dm.db-wal \
                               /var/lib/dungeon-master/dm.db-shm
   ```

   The `-wal` / `-shm` sidecars belong to the broken DB and must
   be cleared so SQLite doesn't try to reattach them to the
   restored file.

5. **Copy the backup into place**:

   ```bash
   sudo -u dungeonmaster cp /var/backups/dm/dm-2026-05-01.db \
                            /var/lib/dungeon-master/dm.db
   ```

6. **Re-verify** in situ — paranoia is free:

   ```bash
   sudo -u dungeonmaster sqlite3 /var/lib/dungeon-master/dm.db "PRAGMA integrity_check"
   ```

7. **Restart the services**:

   ```bash
   sudo systemctl start dungeon-master.service dungeon-master-imageworker.service
   ```

8. **Smoke-test**:

   ```bash
   curl -k https://dm.mcconaghygroup.internal/health
   # Expected: {"status":"ok","db":"ok"}
   ```

   Then log in via browser, open a campaign, confirm characters /
   sessions / messages render.

## Image directory

Generated PNGs live under `/var/lib/dungeon-master/images/<uuid>.png`
and are rsynced to `/var/backups/dm/images/`. To restore image
files, mirror in the opposite direction:

```bash
sudo -u dungeonmaster rsync -a /var/backups/dm/images/ \
                              /var/lib/dungeon-master/images/
```

PNGs are immutable once written; missing files surface as broken
image cards in the UI but never break the DB.

## What to log when this happens

Open a journal entry under `Follow-ups` in `AGENTS.md` describing:
the backup file used, the integrity-check result on the live DB
that prompted restore, the time gap (how much player work, if any,
was lost), and any AVC denials or systemd unit failures noticed
during the restore. Restore is rare; capturing the context the
*next* operator needs is most of the value of having gone through it.

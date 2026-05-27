# Phase 7 deploy-readiness checklist

Phase 7 (hardening) is **complete from a development standpoint**:
all the pieces are landed, unit-tested where unit-testable, and
syntactically / smoke-tested where shell-scripted. But several
pieces only meaningfully verify against a real production deploy —
SELinux enforcing mode, watchdog timer firings, backup integrity
on a production-shaped database, restore-procedure usability,
cron actually firing on schedule.

This document is the operator's runbook for the **first time
production is stood up**. Run through the checklist; if any item
fails, fix it then. The Phase 7 work expects this checklist to
pass; if it doesn't, the Phase 7 surface needs a follow-up rather
than the deploy.

---

## What's already verified (do not re-verify on the deploy box)

These are locked by the test suite and CI; passing them again
locally adds no signal:

- 485 unit + integration tests pass
- Rate limits trigger on the auth-shaped endpoints with the
  right Retry-After + human message (`tests/test_ratelimit.py`)
- Invite-revocation lifecycle: mint, list, revoke, redeem, the
  legacy 7-day grace path, single-use semantics
  (`tests/test_invite_revocation.py`)
- Structured-logging contracts: JSON shape, request_id
  propagation through `asyncio.create_task` and
  `asyncio.to_thread`, X-Request-ID inbound + outbound,
  access-log per-request record, extras lifted to top level
  (`tests/test_logging_config.py`)
- /metrics exposition is parseable by the Prometheus parser, http
  counter increments per request, llm counter increments per
  call, path-label cardinality respects the route template
  (`tests/test_metrics.py`)

These are smoke-tested locally:

- Alerting hook (`deploy/alerts/notify.sh`) writes to syslog +
  the local fallback log file when invoked manually
- Backup retention rotation (`deploy/cron/dm-backup.sh`)
  correctly preserves 8 daily + 12 Sunday + 6 first-of-month
  with bucket-budget dedup, against a 6-month synthetic fixture
  (181 entries → 24 kept, oldest first-of-month preserved)
- Backup integrity-failure path: bad PRAGMA result drops the
  .tmp, preserves the previous good snapshot, exits 4
- Disk-space + log-growth watchdogs run end-to-end against
  stubbed state files and exit 0

---

## What requires the deploy box (run this list when standing up production)

### 1. SELinux enforcing mode

The dev box is in `Enforcing` mode but the unit suite uses
in-memory SQLite, doesn't bind real port 8001, doesn't write to
`/var/lib/dungeon-master/`. The policy module loads cleanly via
`semodule -i` (verified locally with `checkmodule -m -M` +
`semodule_package`) but the policy hasn't been exercised against
the running service.

```bash
# On the deploy box, after bootstrap.sh has run:
sudo getenforce            # → Enforcing
sudo semodule -l | grep dungeon-master   # → dungeon-master 1.0.0
sudo semanage port -l | grep -E '\bhttp_port_t\b.*\b8001\b'
sudo getsebool httpd_can_network_connect  # → on
sudo ls -lZ /var/lib/dungeon-master/dm.db   # → var_lib_t

# Start services and watch for AVC denials:
sudo systemctl start dungeon-master.service dungeon-master-imageworker.service
sudo journalctl -u dungeon-master.service --since "5 minutes ago" | grep -i avc
sudo ausearch -m AVC -ts recent

# Smoke-play one turn end-to-end (login, create campaign,
# roll character, submit pc_action). If any step blows up
# with a permission error and an AVC line appears, that's a
# rule the policy module needs.
```

If AVC denials appear: `audit2allow -a -m dungeon-master-extra`,
fold the rules into `deploy/selinux/dungeon-master.te`, re-run
`deploy/selinux/install.sh`. Don't mark the deploy ready until
the AVC log is clean for the duration of a representative play
session.

### 2. Watchdog timer drills

Install verifies the units exist and the timers are enabled. It
does NOT verify alerts actually fire. Drill each one:

```bash
# FLUX-health: temporarily block the FLUX service to force probe
# failures. Either firewall it or stop it on YOUR_AI_SERVER.
# Wait DEGRADED_THRESHOLD_S (120s default) plus one tick (5min)
# and confirm an alert lands.
sudo journalctl -t dungeon-master-alert --since "10 minutes ago"
# Or: tail -f /var/log/dungeon-master/alerts.log
# Then unblock and confirm the recovery alert fires within 5min.

# Disk-space: drop a big sparse file to push usage over 80%.
sudo -u dungeonmaster dd if=/dev/zero of=/var/lib/dungeon-master/.fill bs=1M count=<N>
# Wait one tick (15min) — confirm a "warn" alert.
# Bump count to push over 90% — confirm a "critical" alert.
sudo rm /var/lib/dungeon-master/.fill
# Confirm a "recovered" alert next tick.

# Log-growth: append 100MB+ to a log file in one hour.
sudo -u dungeonmaster bash -c 'yes | head -c 105m >> /var/log/dungeon-master/app.log'
# Wait one hour (or trigger manually:
sudo systemctl start dm-watchdog-log-growth.service
# ).  Confirm alert.
```

Each drill should produce a line in `/var/log/dungeon-master/alerts.log`
and a syslog entry under tag `dungeon-master-alert`. Both. If
either is missing, the alert hook isn't reaching that channel.

### 3. Backup integrity check on real DB

The unit tests stub `sqlite3` because the dev box doesn't have
the CLI installed. Production has it (bootstrap.sh installs the
`sqlite` package). Run the backup against the live DB to confirm:

```bash
sudo -u dungeonmaster /opt/dungeon-master/deploy/cron/dm-backup.sh
ls -lh /var/backups/dm/dm-$(date -u +%Y-%m-%d).db
sudo -u dungeonmaster sqlite3 /var/backups/dm/dm-$(date -u +%Y-%m-%d).db "PRAGMA integrity_check"
# → ok
```

Open the backup with sqlite3 and confirm it has actual schema +
rows (not just "fake-db" text from the smoke test):

```bash
sudo -u dungeonmaster sqlite3 /var/backups/dm/dm-$(date -u +%Y-%m-%d).db \
    "SELECT name FROM sqlite_master WHERE type='table'" | head
```

Should list `users`, `campaigns`, `campaign_invites`, etc.

### 4. Restore procedure timing — cold operator drill

`deploy/RESTORE.md` says "one-page playbook." The verification is
**an operator who has never run a restore follows the procedure
cold and gets to a working system in under 30 minutes**. If they
have to consult the codebase or ask a question, the playbook is
incomplete.

Drill (in a maintenance window, with players warned):

```bash
# 1. Take a fresh backup so you have a known-good restore point.
sudo -u dungeonmaster /opt/dungeon-master/deploy/cron/dm-backup.sh

# 2. Hand the operator the URL of RESTORE.md and a printout
#    if they want one. Start a stopwatch.

# 3. Operator follows the procedure: stop services, mv broken
#    DB aside, rm -wal -shm, cp backup into place, integrity
#    check, restart, smoke-test.

# 4. Stop the stopwatch when /health returns ok and the
#    operator confirms they can log in via browser.
```

If the timing is over 30 minutes OR the operator had to ask a
clarifying question, file a doc-improvement Follow-up against
`deploy/RESTORE.md`.

### 5. Cron schedule verification

Bootstrap installs the cron entries. Verify they're picked up by
the cron daemon and that they fire on schedule:

```bash
sudo systemctl status crond           # → active (running)
sudo cat /etc/cron.d/dungeon-master   # → entries present
sudo journalctl -u crond --since "2 days ago" | grep dungeonmaster
# Should show two CMD lines per day: 02:00 (backup) and 03:00 (rsync).
```

If no cron logs after two days have passed: the cron daemon may
not be reading `/etc/cron.d/` (SELinux confusion, file mode, file
encoding). Investigate before assuming the backups exist.

### 6. /metrics endpoint internal-only

The endpoint is on the FastAPI app and the nginx config is
gated to localhost-only. Verify both:

```bash
# From the box itself (should succeed):
curl -k -s https://dm.example.internal/metrics | head
curl -s http://127.0.0.1:8001/metrics | head

# From another LAN host (should be 403):
curl -k https://dm.example.internal/metrics
# → "403 Forbidden" from nginx
```

If the LAN host gets through, the nginx `location = /metrics`
block isn't taking effect — check nginx -t and reload.

---

## What to do when this checklist completes

- Each item passing → tick it on a paper printout, file the
  printout with the deploy.
- Any item failing → file a Follow-up in `AGENTS.md` under the
  appropriate trigger ("when posture changes" / "when this bites"
  / a bare-date trigger if it's blocking imminent player traffic).
- After all items pass: this document can be marked "verified
  YYYY-MM-DD on <host>". Phase 7 is then deploy-verified, not
  just dev-complete.

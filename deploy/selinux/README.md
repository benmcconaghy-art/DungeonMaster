# SELinux policy module — Dungeon Master

Phase 7 hardening (spec §13). The deployment runs on AlmaLinux 10
with SELinux in enforcing mode. The pieces in this directory exist
so the deploy posture is reviewable and version-controlled rather
than living only as ad-hoc `semanage` commands an operator
remembers to type.

## Files

- **`dungeon-master.te`** — Type Enforcement source. Currently
  carries no allow rules: the service runs as
  `unconfined_service_t` and the stock targeted policy already
  permits the operations the app needs. The file exists as a hook
  for future tightening (custom domain transitions,
  audit2allow-derived allow rules, dontaudit suppressions).
- **`dungeon-master.fc`** — File-context regex. Labels
  `/var/lib/dungeon-master`, `/var/log/dungeon-master`, and
  `/etc/dungeon-master` with their canonical types
  (`var_lib_t`, `var_log_t`, `etc_t`).
- **`install.sh`** — Operational installer. Compiles the .te +
  .fc into a .pp module, loads it via `semodule -i`, runs the
  `semanage port` / `semanage fcontext` / `restorecon` /
  `setsebool` commands the deployment needs. Idempotent. Run as
  root.

## When to run

Automatically: `deploy/bootstrap.sh` calls `install.sh` near the
end of the deploy. Re-running bootstrap.sh re-runs this; it's
safe.

Manually: any time the .te / .fc files change. Re-running
`semodule -i` upgrades the loaded module in place.

## Verifying the install

```bash
# Module is loaded
semodule -l | grep dungeon-master

# Port 8001 is labelled http_port_t
semanage port -l | grep -E '\bhttp_port_t\b.*\b8001\b'

# Files have the expected types
ls -lZ /var/lib/dungeon-master/dm.db        # → var_lib_t
ls -lZ /var/log/dungeon-master/             # → var_log_t

# Boolean is on (so nginx can talk to localhost:8001)
getsebool httpd_can_network_connect          # → on

# No AVC denials in the audit log since the service started
journalctl -u dungeon-master.service --since "1 hour ago" \
    | grep -i avc || echo "no AVCs"
```

## When AVC denials show up

Run the service under enforcing mode (the default after
bootstrap.sh). If something fails with a "Permission denied" that
looks SELinux-shaped, check:

```bash
ausearch -m AVC -ts recent
```

Then derive the allow rules:

```bash
ausearch -m AVC -ts recent | audit2allow -a -m dungeon-master-extra
```

This produces a `.te` snippet — copy the new `allow ...;` lines
into `dungeon-master.te` (preserving the existing `module` /
`require` blocks), then re-run `install.sh` to rebuild and load
the updated module.

Don't paste raw audit2allow output verbatim — read the rules,
decide whether each one is something the service genuinely needs
(allow) or something we'd rather suppress without granting
(`dontaudit`).

## When to introduce a confined domain

Today the service runs as `unconfined_service_t`. If the deploy
posture ever tightens (e.g. a public-internet exposure, multi-
tenant deployment, regulatory pressure), the next step is a real
confined domain — `dungeon_master_t` with a `type_transition`
from `init_t`, allow rules narrow to what the service actually
needs, and `restrict` rules keeping it out of paths it doesn't.
That's a meaningful policy-engineering task, not a one-line
change. The .te file in this directory is structured so the
addition lands in this file rather than as a separate module.

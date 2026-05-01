#!/usr/bin/env bash
# Install the Dungeon Master SELinux policy module + apply the
# operational SELinux state the deployment needs.
#
# Phase 7 hardening (spec §13). Run as root after the systemd unit
# install and before starting the service. Idempotent — safe to re-run.
#
# Pieces this script handles:
#
#   1. Compile dungeon-master.te + .fc into a .pp module.
#   2. Load the module via semodule -i.
#   3. Label TCP port 8001 as http_port_t so nginx (httpd_t) can
#      proxy_pass to the gunicorn worker on a non-default port.
#   4. Add file-context entries for the deployment paths and run
#      restorecon over them so existing files pick up the right type.
#   5. Set httpd_can_network_connect = 1 so nginx can speak to
#      127.0.0.1:8001 over a localhost socket.
#
# The .te/.fc pair carries the file-context declarations; the
# port labeling and boolean are operator-time semanage/setsebool
# operations and live here rather than in the policy module per the
# rationale in dungeon-master.te.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "deploy/selinux/install.sh must be run as root" >&2
    exit 1
fi

if ! command -v semodule >/dev/null 2>&1; then
    echo "SELinux tooling (semodule) not found; skipping." >&2
    echo "If this host is supposed to enforce SELinux, install:" >&2
    echo "    dnf install -y policycoreutils policycoreutils-python-utils" >&2
    exit 2
fi

if ! command -v checkmodule >/dev/null 2>&1; then
    echo "checkmodule not found; install:" >&2
    echo "    dnf install -y checkpolicy" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d -t dm-selinux.XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

cd "${WORK_DIR}"

log() { printf '\n[selinux] %s\n' "$*"; }

# ---------- 1+2: compile and load the module ----------------------------------

log "Compiling dungeon-master.te"
cp "${SCRIPT_DIR}/dungeon-master.te" "${SCRIPT_DIR}/dungeon-master.fc" .
checkmodule -m -M -o dungeon-master.mod dungeon-master.te
semodule_package -o dungeon-master.pp \
    -m dungeon-master.mod \
    -f dungeon-master.fc

log "Loading the policy module via semodule"
semodule -i dungeon-master.pp

# ---------- 3: port labeling --------------------------------------------------

# semanage port -a fails with a non-zero exit if the port is already
# labelled; the silently-idempotent path is to check first. The nginx
# config in deploy/nginx.conf proxies to 127.0.0.1:8001 so this label
# is what lets httpd_t reach the gunicorn worker.
log "Ensuring TCP port 8001 is labelled http_port_t"
if ! semanage port -l | awk '$1 == "http_port_t" && $2 == "tcp" { print $0 }' \
        | grep -q -E '\b8001\b'; then
    semanage port -a -t http_port_t -p tcp 8001
else
    log "  port 8001 already labelled, skipping"
fi

# ---------- 4: file context entries + restorecon ------------------------------

# semanage fcontext is also non-idempotent on -a; use -a -m if exists,
# otherwise -a. The standard idiom: try -a, if it fails because the
# entry exists, fall through to -m to update.
fcontext_apply() {
    local pathspec="$1"
    local type="$2"
    if semanage fcontext -l | grep -F -q "${pathspec}"; then
        log "  fcontext for ${pathspec} already present, refreshing"
        semanage fcontext -m -t "${type}" "${pathspec}" || true
    else
        semanage fcontext -a -t "${type}" "${pathspec}"
    fi
}

log "Applying file-context entries for /var/lib, /var/log, /etc"
fcontext_apply '/var/lib/dungeon-master(/.*)?' 'var_lib_t'
fcontext_apply '/var/log/dungeon-master(/.*)?' 'var_log_t'
fcontext_apply '/etc/dungeon-master(/.*)?'     'etc_t'

log "Running restorecon over the deployment paths"
# -F forces relabel even if the type matches (safety against stale
# user/role components from a previous install). -v makes the change
# log visible in install output for an operator review trail.
restorecon -RFv /var/lib/dungeon-master /var/log/dungeon-master /etc/dungeon-master 2>&1 \
    | sed 's/^/[selinux]   /' || true

# ---------- 5: nginx → localhost boolean --------------------------------------

log "Setting httpd_can_network_connect = 1 (-P persists across reboots)"
setsebool -P httpd_can_network_connect 1

log "SELinux policy install complete."
echo
echo "Verify with:"
echo "  semodule -l | grep dungeon-master"
echo "  semanage port -l | grep 8001"
echo "  matchpathcon /var/lib/dungeon-master/dm.db"
echo "  getsebool httpd_can_network_connect"
echo
echo "If AVC denials appear later under enforcing mode:"
echo "  ausearch -m AVC -ts recent | audit2allow -a"
echo "  # add the resulting allow rules to dungeon-master.te and re-run this script"

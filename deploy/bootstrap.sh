#!/usr/bin/env bash
# Bootstrap a fresh AlmaLinux 10.1 host into a runnable Dungeon Master
# deployment. Idempotent — safe to re-run after a partial failure or to
# pick up package / cert / config drift.
#
# Covers spec §13: Packages, Database directory, App user and directories,
# TLS (self-signed cert), firewalld, and the Application install up
# through `alembic upgrade head`. Stops short of starting the service
# so an operator can review the env file before the first boot.
#
# Run as root.
#
# Usage:
#   sudo ./deploy/bootstrap.sh [--server-name=dm.mcconaghygroup.internal] \
#                              [--server-ip=<host-ip>] \
#                              [--repo=<git-url>] \
#                              [--ref=<branch-or-tag>]

set -euo pipefail

# ---------- defaults / args ----------------------------------------------------

SERVER_NAME="dm.mcconaghygroup.internal"
SERVER_IP=""
REPO_URL=""
REPO_REF="main"
APP_USER="dungeonmaster"
APP_DIR="/opt/dungeon-master"
DATA_DIR="/var/lib/dungeon-master"
LOG_DIR="/var/log/dungeon-master"
ETC_DIR="/etc/dungeon-master"
TLS_CERT="/etc/pki/tls/certs/dm.crt"
TLS_KEY="/etc/pki/tls/private/dm.key"

for arg in "$@"; do
    case "$arg" in
        --server-name=*) SERVER_NAME="${arg#*=}" ;;
        --server-ip=*)   SERVER_IP="${arg#*=}" ;;
        --repo=*)        REPO_URL="${arg#*=}" ;;
        --ref=*)         REPO_REF="${arg#*=}" ;;
        -h|--help)
            grep -E '^#' "$0" | sed -e 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "bootstrap.sh must be run as root" >&2
    exit 1
fi

log() { printf '\n[bootstrap] %s\n' "$*"; }

# ---------- §13 Packages -------------------------------------------------------

log "Installing system packages"
# Valkey replaces Redis on AlmaLinux 10 — Redis was dropped from the base
# repos after the 2024 SSPL relicensing. Valkey is the Linux Foundation
# fork and is wire-protocol-compatible: redis-py and redis://... URLs work
# unchanged. Service name and config path differ (valkey.service,
# /etc/valkey/valkey.conf) but the runtime contract is identical.
dnf install -y \
    python3.12 python3.12-devel \
    valkey nginx \
    sqlite \
    gcc git \
    openssl \
    firewalld \
    curl tar

systemctl enable --now valkey
systemctl enable --now firewalld

# uv is not in the AppStream; install from the official script if absent.
# We install system-wide under /usr/local/bin so the systemd unit and the
# operator's interactive shells both find it.
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    UV_INSTALL_DIR=/usr/local/bin curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh
fi

# ---------- §13 App user and directories --------------------------------------

if ! id "${APP_USER}" >/dev/null 2>&1; then
    log "Creating ${APP_USER} user"
    useradd -r -m -d "${DATA_DIR}" -s /sbin/nologin "${APP_USER}"
fi

log "Ensuring data, log, image, and config directories exist"
install -d -m 0750 -o "${APP_USER}" -g "${APP_USER}" "${DATA_DIR}"
install -d -m 0750 -o "${APP_USER}" -g "${APP_USER}" "${DATA_DIR}/images"
install -d -m 0750 -o "${APP_USER}" -g "${APP_USER}" "${LOG_DIR}"
install -d -m 0755 -o root          -g root          "${ETC_DIR}"

# ---------- §13 TLS — self-signed cert ----------------------------------------

if [[ ! -f "${TLS_CERT}" || ! -f "${TLS_KEY}" ]]; then
    log "Generating self-signed TLS certificate (5-year validity)"
    san_dns="DNS:${SERVER_NAME},DNS:${SERVER_NAME%%.*}"
    san="${san_dns}"
    if [[ -n "${SERVER_IP}" ]]; then
        san="${san_dns},IP:${SERVER_IP}"
    fi
    install -d -m 0755 /etc/pki/tls/certs /etc/pki/tls/private
    openssl req -x509 -nodes -newkey rsa:4096 \
        -keyout "${TLS_KEY}" \
        -out    "${TLS_CERT}" \
        -days 1825 \
        -subj "/CN=${SERVER_NAME}" \
        -addext "subjectAltName=${san}"
    chmod 600 "${TLS_KEY}"
    chown root:nginx "${TLS_KEY}"
else
    log "TLS certificate already present at ${TLS_CERT}, leaving in place"
fi

# ---------- §13 firewalld -----------------------------------------------------

log "Configuring firewalld (https + http for redirect)"
firewall-cmd --permanent --add-service=https >/dev/null
firewall-cmd --permanent --add-service=http  >/dev/null
firewall-cmd --reload >/dev/null

# ---------- §13 Application install -------------------------------------------

if [[ ! -d "${APP_DIR}/.git" ]]; then
    if [[ -z "${REPO_URL}" ]]; then
        cat <<EOF >&2
First-time install needs a repo URL. Re-run with:
  sudo $0 --repo=<git-url> [--ref=${REPO_REF}]
or, if the source is already on disk, drop it at ${APP_DIR} (chowned to ${APP_USER})
and re-run this script — the clone step will be skipped.
EOF
        exit 3
    fi
    log "Cloning ${REPO_URL} (${REPO_REF}) into ${APP_DIR}"
    git clone --branch "${REPO_REF}" "${REPO_URL}" "${APP_DIR}"
else
    log "Repo already present at ${APP_DIR}, fetching latest ${REPO_REF}"
    git -C "${APP_DIR}" fetch --quiet origin "${REPO_REF}"
    git -C "${APP_DIR}" checkout --quiet "${REPO_REF}"
    git -C "${APP_DIR}" reset --hard --quiet "origin/${REPO_REF}"
fi

chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

# Drop a baseline env file the systemd unit references. Operators edit
# this in place — never overwrite if it already exists.
if [[ ! -f "${ETC_DIR}/env" ]]; then
    log "Writing default ${ETC_DIR}/env (review before starting the service)"
    session_secret="$(openssl rand -hex 32)"
    install -m 0640 -o root -g "${APP_USER}" /dev/stdin "${ETC_DIR}/env" <<EOF
# /etc/dungeon-master/env — runtime configuration. Loaded by the systemd
# unit and read by app/config.py via pydantic-settings. Anything not set
# here falls back to the defaults in app/config.py (which match the
# production layout described in spec §13).

DB_PATH=${DATA_DIR}/dm.db
IMAGE_STORAGE_PATH=${DATA_DIR}/images
VLLM_BASE_URL=http://svrai01.mcconaghygroup.internal:8000
FLUX_BASE_URL=http://svrai01.mcconaghygroup.internal:11437
REDIS_URL=redis://127.0.0.1:6379/0
SESSION_SECRET=${session_secret}
EOF
else
    log "${ETC_DIR}/env already present, leaving operator copy in place"
fi

log "Installing Python dependencies via uv"
sudo -u "${APP_USER}" --preserve-env=PATH bash -lc \
    "cd ${APP_DIR} && /usr/local/bin/uv sync --frozen 2>/dev/null || /usr/local/bin/uv sync"

log "Running database migrations"
# Pull DB_PATH from the env file so a one-shot bootstrap matches the
# location systemd will use when the service starts.
db_path="$(grep -E '^DB_PATH=' "${ETC_DIR}/env" | cut -d= -f2-)"
sudo -u "${APP_USER}" bash -lc \
    "cd ${APP_DIR} && DB_PATH='${db_path}' /usr/local/bin/uv run alembic upgrade head"

# ---------- nginx + systemd unit symlinks --------------------------------------

log "Installing systemd units and nginx config"
install -m 0644 "${APP_DIR}/deploy/dungeon-master.service"             /etc/systemd/system/dungeon-master.service
install -m 0644 "${APP_DIR}/deploy/dungeon-master-imageworker.service" /etc/systemd/system/dungeon-master-imageworker.service
install -d -m 0755 /etc/nginx/conf.d
install -m 0644 "${APP_DIR}/deploy/nginx.conf"                         /etc/nginx/conf.d/dungeon-master.conf

systemctl daemon-reload
nginx -t

log "Bootstrap complete."
cat <<EOF

Next steps (manual, intentionally not auto-started so you can review):

  1. Inspect ${ETC_DIR}/env and adjust if needed.
  2. Start the services:
       systemctl enable --now nginx
       systemctl enable --now dungeon-master.service
       # Phase 5+ only:
       # systemctl enable --now dungeon-master-imageworker.service
  3. Verify:
       curl -k https://${SERVER_NAME}/health
       # Expect: {"status":"ok","db":"ok"}

Re-running this script is safe: existing users, certs, config files, and
clones are detected and left in place; only fresh-install steps execute.
EOF

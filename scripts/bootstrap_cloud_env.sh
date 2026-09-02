#!/usr/bin/env bash
set -euo pipefail

readonly UV_VERSION="0.11.29"
readonly UV_BIN="/usr/local/bin/uv"
readonly PYTHON_VERSION="3.10.12"
readonly VENV_DIR="/opt/dlm-wonn-venv"
readonly UV_PYTHON_DIR="/opt/dlm-wonn-python"
readonly UV_CACHE_DIR="/var/cache/dlm-wonn-uv"
readonly MANIFEST_PATH="/opt/dlm-wonn-environment-manifest.json"

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
requirements_lock="$repo_root/requirements-lock.txt"

log() {
    printf '[bootstrap] %s\n' "$*"
}

die() {
    printf '[bootstrap] ERROR: %s\n' "$*" >&2
    exit 1
}

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    die "run this script as root (for example: sudo bash $0)"
fi

[ -r /etc/os-release ] || die "/etc/os-release is missing"
# shellcheck disable=SC1091
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "Ubuntu is required; detected ${PRETTY_NAME:-unknown}"

kernel="$(uname -s)"
[ "$kernel" = "Linux" ] || die "Linux is required; detected $kernel"
machine="$(uname -m)"
[ "$machine" = "x86_64" ] || die "linux/amd64 is required; detected linux/$machine"
[ -f "$requirements_lock" ] || die "missing dependency lock: $requirements_lock"

available_kb="$(df -Pk / | awk 'NR == 2 {print $4}')"
[ -n "$available_kb" ] || die "could not determine free space on the system disk"
log "system disk free space: $((available_kb / 1024 / 1024)) GiB"
if [ "$available_kb" -lt $((15 * 1024 * 1024)) ]; then
    log "WARNING: less than the recommended 15 GiB is free on the system disk"
fi

export DEBIAN_FRONTEND=noninteractive
log "installing Ubuntu system dependencies"
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    ninja-build \
    pkg-config \
    python3-venv \
    rsync \
    time \
    tmux \
    util-linux
git lfs install --system

log "checking access to the uv installer and Python package index"
curl --proto '=https' --tlsv1.2 -LsSf --retry 3 \
    "https://astral.sh/uv/$UV_VERSION/install.sh" -o /dev/null
curl --proto '=https' --tlsv1.2 -LsSf --retry 3 \
    "https://pypi.org/simple/packaging/" -o /dev/null

installed_uv_version=""
if [ -x "$UV_BIN" ]; then
    installed_uv_version="$($UV_BIN --version | awk '{print $2}')"
fi
if [ "$installed_uv_version" != "$UV_VERSION" ]; then
    log "installing uv $UV_VERSION"
    curl --proto '=https' --tlsv1.2 -LsSf \
        "https://astral.sh/uv/$UV_VERSION/install.sh" \
        | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
[ -x "$UV_BIN" ] || die "uv was not installed at $UV_BIN"
[ "$($UV_BIN --version | awk '{print $2}')" = "$UV_VERSION" ] \
    || die "failed to install uv $UV_VERSION"

mkdir -p "$UV_PYTHON_DIR" "$UV_CACHE_DIR"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR"
export UV_CACHE_DIR

log "installing uv-managed Python $PYTHON_VERSION"
"$UV_BIN" python install --managed-python --no-bin "$PYTHON_VERSION"
python_source="$("$UV_BIN" python find --managed-python "$PYTHON_VERSION")"

if [ -x "$VENV_DIR/bin/python" ]; then
    existing_version="$($VENV_DIR/bin/python -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
else
    existing_version=""
fi
if [ "$existing_version" != "$PYTHON_VERSION" ]; then
    log "creating Python $PYTHON_VERSION environment at $VENV_DIR"
    "$UV_BIN" venv --clear --python "$python_source" "$VENV_DIR"
else
    log "reusing Python $PYTHON_VERSION environment at $VENV_DIR"
fi

python_include="$($VENV_DIR/bin/python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
[ -f "$python_include/Python.h" ] \
    || die "uv-managed Python is missing $python_include/Python.h"
command -v c++ >/dev/null 2>&1 || die "C++ compiler was not installed"
command -v ninja >/dev/null 2>&1 || die "ninja was not installed"

log "synchronizing exact Python dependencies from requirements-lock.txt"
"$UV_BIN" pip sync \
    --python "$VENV_DIR/bin/python" \
    --strict \
    "$requirements_lock"

log "checking installed dependency metadata"
"$UV_BIN" pip check --python "$VENV_DIR/bin/python"

log "writing environment manifest to $MANIFEST_PATH"
"$VENV_DIR/bin/python" - \
    "$MANIFEST_PATH" "$requirements_lock" "$UV_VERSION" <<'PY'
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

manifest_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
uv_version = sys.argv[3]
os_release = {}
for raw_line in Path("/etc/os-release").read_text().splitlines():
    if "=" in raw_line:
        key, value = raw_line.split("=", 1)
        os_release[key] = value.strip().strip('"')

payload = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "os": os_release.get("PRETTY_NAME", os_release.get("ID", "unknown")),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "uv": uv_version,
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "requirements_lock": str(lock_path),
    "requirements_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
}
temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary_path, manifest_path)
PY

log "cleaning installation caches"
"$UV_BIN" cache clean --cache-dir "$UV_CACHE_DIR"
apt-get clean
find /var/lib/apt/lists -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

log "environment ready at $VENV_DIR"
log "manifest written to $MANIFEST_PATH"
log "run: bash $repo_root/scripts/verify_cloud_env.sh"

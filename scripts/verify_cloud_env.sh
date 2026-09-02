#!/usr/bin/env bash
set -euo pipefail

readonly UV_BIN="/usr/local/bin/uv"
readonly EXPECTED_CUDA_RUNTIME="13.0"

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="${DLM_WONN_PYTHON:-/opt/dlm-wonn-venv/bin/python}"
expected_gpu_count="${DLM_WONN_EXPECTED_GPU_COUNT:-4}"
summary_path="${DLM_WONN_VERIFY_SUMMARY:-/tmp/dlm-wonn-cloud-environment-verification.json}"
requirements_lock="$repo_root/requirements-lock.txt"

log() {
    printf '[verify] %s\n' "$*"
}

die() {
    printf '[verify] ERROR: %s\n' "$*" >&2
    exit 1
}

[ -x "$python_bin" ] || die "missing environment: $python_bin"
[ -f "$requirements_lock" ] || die "missing dependency lock: $requirements_lock"
[[ "$expected_gpu_count" =~ ^[1-9][0-9]*$ ]] \
    || die "DLM_WONN_EXPECTED_GPU_COUNT must be a positive integer"
[ -x "$UV_BIN" ] || die "uv is missing at $UV_BIN"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not on PATH"
command -v c++ >/dev/null 2>&1 || die "C++ compiler is not on PATH"
command -v ninja >/dev/null 2>&1 || die "ninja is not on PATH"
command -v git >/dev/null 2>&1 || die "git is not on PATH"

git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "$repo_root is not a Git checkout"
if [ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]; then
    die "project checkout must be clean before cloud verification"
fi

python_include="$($python_bin -c 'import sysconfig; print(sysconfig.get_path("include"))')"
[ -f "$python_include/Python.h" ] || die "missing Python header: $python_include/Python.h"
export CPATH="$python_include${CPATH:+:$CPATH}"

log "checking Python 3.10.12"
"$python_bin" - <<'PY'
import sys

actual = sys.version_info[:3]
if actual != (3, 10, 12):
    raise SystemExit(f"expected Python 3.10.12, found {sys.version.split()[0]}")
print(f"[verify] Python: {sys.version.split()[0]}")
print(f"[verify] Python executable: {sys.executable}")
PY

log "checking exact dependency versions"
UV_NO_CACHE=1 "$UV_BIN" pip check --python "$python_bin"
"$python_bin" - "$requirements_lock" <<'PY'
import sys
from importlib import metadata
from pathlib import Path

from packaging.utils import canonicalize_name

lock_path = Path(sys.argv[1])
expected = {}
for line_number, raw_line in enumerate(lock_path.read_text().splitlines(), 1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.count("==") != 1:
        raise SystemExit(f"unsupported lock entry at {lock_path}:{line_number}: {line}")
    name, version = line.split("==", 1)
    expected[canonicalize_name(name)] = version

installed = {
    canonicalize_name(dist.metadata["Name"]): dist.version
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
missing = sorted(set(expected) - set(installed))
unexpected = sorted(set(installed) - set(expected))
mismatched = sorted(
    (name, expected[name], installed[name])
    for name in set(expected) & set(installed)
    if installed[name] != expected[name]
)
if missing or unexpected or mismatched:
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unexpected:
        details.append("unexpected: " + ", ".join(unexpected))
    if mismatched:
        details.append(
            "version mismatch: "
            + ", ".join(f"{name} expected={want} actual={got}" for name, want, got in mismatched)
        )
    raise SystemExit("dependency lock mismatch\n  " + "\n  ".join(details))
print(f"[verify] dependencies: exact match ({len(expected)} packages)")
PY

log "checking NVIDIA driver inventory"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

log "checking PyTorch, CUDA, $expected_gpu_count GPU(s), BF16, and torch.compile"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" - "$expected_gpu_count" "$summary_path" "$EXPECTED_CUDA_RUNTIME" <<'PY'
import ctypes
import ctypes.util
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_cuda_version(value):
    major, minor = (int(part) for part in value.split(".")[:2])
    return major * 1000 + minor * 10


expected_gpu_count = int(sys.argv[1])
summary_path = Path(sys.argv[2])
expected_cuda_runtime = sys.argv[3]
if torch.version.cuda is None:
    raise SystemExit("installed PyTorch has no CUDA runtime")
if torch.version.cuda != expected_cuda_runtime:
    raise SystemExit(
        f"expected PyTorch CUDA runtime {expected_cuda_runtime}, found {torch.version.cuda}"
    )

cuda_library = ctypes.util.find_library("cuda")
if not cuda_library:
    raise SystemExit("NVIDIA driver library libcuda.so was not found")
driver = ctypes.CDLL(cuda_library)
if driver.cuInit(0) != 0:
    raise SystemExit("NVIDIA driver initialization failed")
driver_version = ctypes.c_int()
if driver.cuDriverGetVersion(ctypes.byref(driver_version)) != 0:
    raise SystemExit("could not query the NVIDIA driver CUDA API version")
required_driver_api = parse_cuda_version(torch.version.cuda)
if driver_version.value < required_driver_api:
    raise SystemExit(
        f"NVIDIA driver CUDA API {driver_version.value} is older than "
        f"the PyTorch CUDA {torch.version.cuda} runtime requirement {required_driver_api}"
    )

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
gpu_count = torch.cuda.device_count()
if gpu_count != expected_gpu_count:
    raise SystemExit(f"expected {expected_gpu_count} visible GPUs, found {gpu_count}")

print(f"[verify] PyTorch: {torch.__version__}")
print(f"[verify] CUDA runtime: {torch.version.cuda}")
print(f"[verify] NVIDIA driver CUDA API: {driver_version.value}")
gpu_names = []
for device_index in range(gpu_count):
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device_index):
        fp32_left = torch.randn((64, 64), device=device, dtype=torch.float32)
        fp32_right = torch.randn((64, 64), device=device, dtype=torch.float32)
        fp32_result = fp32_left @ fp32_right
        torch.cuda.synchronize(device)
        if not torch.isfinite(fp32_result).all().item():
            raise SystemExit(f"GPU {device_index} FP32 matrix multiplication failed")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise SystemExit(f"GPU {device_index} does not support native BF16")
        left = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
        right = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
        result = left @ right
        torch.cuda.synchronize(device)
    if result.dtype != torch.bfloat16 or not torch.isfinite(result.float()).all().item():
        raise SystemExit(f"GPU {device_index} BF16 matrix multiplication failed")
    props = torch.cuda.get_device_properties(device_index)
    gpu_names.append(props.name)
    print(f"[verify] GPU {device_index}: {props.name}; FP32 and BF16 compute OK")


def eager_function(value):
    return torch.sin(value) * 2.0 + 1.0


compile_input = torch.randn(1024, device="cuda:0")
compiled_function = torch.compile(eager_function, backend="inductor", fullgraph=True)
compiled_result = compiled_function(compile_input)
torch.cuda.synchronize(0)
if not torch.allclose(compiled_result, eager_function(compile_input), rtol=1e-5, atol=1e-6):
    raise SystemExit("torch.compile result differs from eager execution")
print("[verify] torch.compile: Inductor CUDA execution OK")

for module_name in (
    "configs.config",
    "modules.model",
    "modules.model_factory",
    "modules.wonn_model",
    "train_step",
):
    importlib.import_module(module_name)
print("[verify] project imports: OK")

summary = {
    "status": "passed",
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "nvidia_driver_cuda_api": driver_version.value,
    "gpu_count": gpu_count,
    "gpu_names": gpu_names,
    "cuda_fp32": "passed",
    "cuda_bf16": "passed",
    "torch_compile": "passed",
    "project_imports": "passed",
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(temporary_path, summary_path)
print("[verify] summary: " + json.dumps(summary, sort_keys=True))
print(f"[verify] summary file: {summary_path}")
PY

log "all cloud environment checks passed"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="${DLM_WONN_PYTHON:-$repo_root/.venv/bin/python}"
config_path="src/configs/training_configs/train_de-en-WONN-L12K768T3-phase5-90k.yml"
run_dir="${DLM_WONN_L12K768_RUN_DIR:-$repo_root/outputs/phase5/wonn_l12k768t3_seed42_b12_0_90k}"
elf_run_dir="${DLM_WONN_ELF_BASELINE:-$repo_root/outputs/phase5/elf_b_seed42_b12_0_90k}"
analysis_dir="$run_dir/analysis/final"
status_path="$run_dir/pipeline_status.json"
label="WONN-L12K768T3"
current_stage="preflight"
pipeline_done=0

cd "$repo_root"

if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    echo "refusing to start a formal run from a dirty checkout" >&2
    git status --short >&2
    exit 2
fi
if [ ! -x "$python_bin" ]; then
    echo "missing Python environment: $python_bin" >&2
    exit 3
fi
if [ ! -s "$config_path" ]; then
    echo "missing training config: $config_path" >&2
    exit 4
fi

mkdir -p "$run_dir"
exec 9>"$run_dir/pipeline.lock"
if ! flock -n 9; then
    echo "another pipeline holds $run_dir/pipeline.lock" >&2
    exit 5
fi

experiment_commit="$(git rev-parse HEAD)"
runtime_hashes="$run_dir/runtime_source_hashes.sha256"
find src -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$runtime_hashes"
sha256sum \
    "$config_path" \
    scripts/analyze_wonn_l12k768_90k.py \
    scripts/run_wonn_l12k768_90k_pipeline.sh \
    >> "$runtime_hashes"
git status --short --untracked-files=all > "$run_dir/source_worktree_status.txt"
printf '%s\n' "$experiment_commit" > "$run_dir/source_commit"
printf '%s\n' "$config_path" > "$run_dir/source_config"

write_status() {
    local value="$1"
    local exit_code="${2:-}"
    command=(
        "$python_bin" scripts/phase5_pipeline_checks.py status
        "$status_path" "$value" "$experiment_commit" "$experiment_commit"
        "$current_stage" "$$"
    )
    if [ -n "$exit_code" ]; then
        command+=(--exit-code "$exit_code")
    fi
    "${command[@]}"
}

on_exit() {
    local exit_code="$?"
    if [ "$pipeline_done" -ne 1 ]; then
        write_status failed "$exit_code" || true
    fi
}
trap on_exit EXIT
write_status running

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-2}"

python_version="$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_include="${DLM_WONN_PYTHON_INCLUDE:-$($python_bin -c 'import sysconfig; print(sysconfig.get_path("include"))')}"
if [ ! -f "$python_include/Python.h" ]; then
    user_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
    python_header=""
    if [ -d "$user_home/miniconda3/envs" ]; then
        python_header="$(find "$user_home/miniconda3/envs" -path "*/include/python$python_version/Python.h" -print -quit)"
    fi
    if [ -n "$python_header" ]; then
        python_include="$(dirname "$python_header")"
    fi
fi
if [ ! -f "$python_include/Python.h" ]; then
    echo "missing Python.h for torch.compile" >&2
    exit 6
fi
export CPATH="$python_include${CPATH:+:$CPATH}"

for step in $(seq 10000 10000 90000); do
    "$python_bin" - "$elf_run_dir" "$step" <<'PY'
import json
from pathlib import Path
import sys
run_dir = Path(sys.argv[1])
step = int(sys.argv[2])
marker = json.loads((run_dir / "evaluations" / f"checkpoint_{step}" / "evaluation_complete.json").read_text())
if marker.get("status") != "complete" or marker.get("step") != step or marker.get("num_samples", 0) < 1000:
    raise SystemExit(f"ELF baseline checkpoint {step} lacks 1000 completed samples")
PY
done

if [ "${DLM_WONN_PREFLIGHT_ONLY:-0}" = "1" ]; then
    current_stage="preflight validated"
    write_status running
    pipeline_done=1
    echo "WONN L12/K768 90K preflight validated"
    exit 0
fi

completion="$run_dir/training_complete.json"
completed_step=0
if [ -s "$completion" ]; then
    completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step", 0))' "$completion")"
fi
if [ "$completed_step" -lt 90000 ]; then
    current_stage="$label training to 90000"
    write_status running
    resume_args=()
    if find "$run_dir" -maxdepth 1 -type f -name 'checkpoint_*' -print -quit | grep -q .; then
        resume_args=(--config_override "resume=$run_dir")
    fi
    /usr/bin/time -v -o "$run_dir/process_resources_training.txt" \
        "$python_bin" src/train.py \
        --config "$config_path" \
        --config_override "output_dir=$run_dir" \
        "${resume_args[@]}" >> "$run_dir/pipeline.log" 2>&1
fi

if [ ! -s "$completion" ]; then
    echo "training exited without $completion" >&2
    exit 7
fi
completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step"))' "$completion")"
if [ "$completed_step" -ne 90000 ]; then
    echo "training completed at $completed_step, expected 90000" >&2
    exit 8
fi

for step in $(seq 10000 10000 90000); do
    checkpoint="$run_dir/checkpoint_$step"
    eval_dir="$run_dir/evaluations/checkpoint_$step"
    temporary_dir="${eval_dir}.in_progress"
    if ! "$python_bin" scripts/phase5_pipeline_checks.py checkpoint "$checkpoint"; then
        echo "missing checkpoint: $checkpoint" >&2
        exit 9
    fi
    if "$python_bin" scripts/phase5_pipeline_checks.py evaluation \
        "$eval_dir" "$label" "$step" 1000; then
        echo "$label checkpoint $step evaluation already complete"
        continue
    fi
    timestamp="$(date +%Y%m%dT%H%M%S)"
    if [ -e "$eval_dir" ]; then
        mv "$eval_dir" "${eval_dir}.archived_${timestamp}"
    fi
    if [ -e "$temporary_dir" ]; then
        mv "$temporary_dir" "${temporary_dir}.archived_${timestamp}"
    fi
    mkdir -p "$temporary_dir"
    current_stage="$label evaluation at $step"
    write_status running
    "$python_bin" src/eval.py \
        --config "$config_path" \
        --config_override "output_dir=$temporary_dir" \
        --config_override "num_samples=1000" \
        --config_override "batch_size=16" \
        --checkpoint_path "$checkpoint" \
        --seed 42 >> "$run_dir/pipeline.log" 2>&1
    if ! "$python_bin" scripts/phase5_pipeline_checks.py finalize-evaluation \
        "$temporary_dir" "$label" "$step" 1000; then
        echo "incomplete evaluation artifacts for checkpoint $step" >&2
        exit 10
    fi
    mv "$temporary_dir" "$eval_dir"
done

current_stage="final BLEU/chrF comparison and charts"
write_status running
"$python_bin" scripts/analyze_wonn_l12k768_90k.py \
    --wonn-run-dir "$run_dir" \
    --elf-run-dir "$elf_run_dir" \
    --output-dir "$analysis_dir" >> "$run_dir/pipeline.log" 2>&1

current_stage="complete"
write_status complete 0
"$python_bin" - "$run_dir/pipeline_complete.json" "$experiment_commit" "$analysis_dir/comparison.json" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
payload = {
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_commit": sys.argv[2],
    "wonn_terminal_step": 90000,
    "analysis": sys.argv[3],
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
pipeline_done=1

if command -v notify-send >/dev/null 2>&1; then
    notify-send "DLM-WONN Phase 5" "WONN L12/K768 90K training and curve report completed" || true
fi

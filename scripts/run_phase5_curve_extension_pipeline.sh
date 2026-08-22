#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
common_git_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(dirname "$common_git_dir")"
python_bin="${DLM_WONN_PYTHON:-$main_checkout/.venv/bin/python}"
run_root="outputs/phase5/curve_to_plateau_v1"
elf_base="outputs/phase5/formal50k/elf_b_seed42_b12"
wonn_base="outputs/phase5/redesign_v2/formal50k/wonn_l6t3_seed42_b12"
elf_config="src/configs/training_configs/train_de-en-ELF-B-phase5-90k.yml"
wonn_config="src/configs/training_configs/train_de-en-WONN-L6T3-phase5-130k.yml"
elf_dir="$run_root/elf_b_seed42_b12"
wonn_dir="$run_root/wonn_l6t3_seed42_b12"
status_path="$run_root/pipeline_status.json"
current_stage="preflight"
pipeline_done=0

cd "$repo_root"

if [ "$(git branch --show-current)" != "phase5-wmt14" ]; then
    echo "refusing to run outside the phase5-wmt14 branch" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    echo "refusing to start a formal run from a dirty worktree" >&2
    git status --short >&2
    exit 3
fi
if [ ! -x "$python_bin" ]; then
    echo "missing Python environment: $python_bin" >&2
    exit 4
fi

mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
if ! flock -n 9; then
    echo "another curve-extension pipeline holds $run_root/pipeline.lock" >&2
    exit 5
fi

experiment_commit="$(git rev-parse HEAD)"
runner_commit="$experiment_commit"

write_status() {
    local value="$1"
    local exit_code="${2:-}"
    command=(
        "$python_bin" scripts/phase5_pipeline_checks.py status
        "$status_path" "$value" "$experiment_commit" "$runner_commit"
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

for required in \
    "$elf_config" \
    "$wonn_config" \
    "outputs/phase5/formal50k/pipeline_complete.json" \
    "outputs/phase5/redesign_v2/formal50k/pipeline_complete.json" \
    "$elf_base/training_complete.json" \
    "$elf_base/checkpoint_50000" \
    "$elf_base/evaluations/checkpoint_50000/evaluation_complete.json" \
    "$wonn_base/training_complete.json" \
    "$wonn_base/checkpoint_50000" \
    "$wonn_base/evaluations/checkpoint_50000/evaluation_complete.json"; do
    if [ ! -s "$required" ]; then
        echo "missing required parent artifact: $required" >&2
        exit 7
    fi
done

"$python_bin" - "$elf_base/training_complete.json" "$wonn_base/training_complete.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    payload = json.load(open(path, encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("completed_optimizer_step") != 50000:
        raise SystemExit(f"parent run is not complete at 50000: {path}")
PY
"$python_bin" -c 'from pathlib import Path; from scripts.summarize_phase5_run import _audit_checkpoint; _audit_checkpoint(Path("outputs/phase5/formal50k/elf_b_seed42_b12/checkpoint_50000"), 50000); _audit_checkpoint(Path("outputs/phase5/redesign_v2/formal50k/wonn_l6t3_seed42_b12/checkpoint_50000"), 50000)' \
    2>&1 | tee "$run_root/preflight_checkpoint_audit.log"

manifest_path="$run_root/experiment_manifest.json"
if [ ! -s "$manifest_path" ]; then
    "$python_bin" scripts/capture_phase5_manifest.py \
        --repo-root "$repo_root" \
        --output "$manifest_path" \
        --config "$elf_config" \
        --config "$wonn_config"
fi

"$python_bin" - "$run_root/continuation_lineage.json" "$experiment_commit" "$elf_base" "$wonn_base" <<'PY'
import json
from pathlib import Path
import sys
output, commit, elf, wonn = sys.argv[1:]
payload = {
    "status": "ready",
    "orchestrator_commit": commit,
    "parents": {
        "Transformer ELF-B": {
            "run_dir": elf,
            "checkpoint": f"{elf}/checkpoint_50000",
            "source_commit": Path(elf, "source_commit").read_text(encoding="utf-8").strip(),
        },
        "WONN-L6T3": {
            "run_dir": wonn,
            "checkpoint": f"{wonn}/checkpoint_50000",
            "source_commit": Path(wonn, "source_commit").read_text(encoding="utf-8").strip(),
        },
    },
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

if [ "${DLM_WONN_PREFLIGHT_ONLY:-0}" = "1" ]; then
    current_stage="preflight validated"
    write_status running
    pipeline_done=1
    echo "curve-extension preflight validated"
    exit 0
fi

run_training() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local base_checkpoint="$4"
    local target_step="$5"
    local completion="$output_dir/training_complete.json"
    local log_path="$output_dir/pipeline.log"
    local resource_path="$output_dir/process_resources_to_${target_step}.txt"
    local resume_path="$base_checkpoint"
    mkdir -p "$output_dir"
    if [ -s "$completion" ]; then
        completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step", 0))' "$completion")"
        if [ "$completed_step" -ge "$target_step" ]; then
            echo "$label already reached step $completed_step; skipping target $target_step"
            return
        fi
    fi
    if find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_*' -print -quit | grep -q .; then
        resume_path="$output_dir"
    fi
    printf '%s\n' "$experiment_commit" > "$output_dir/source_commit"
    printf '%s\n' "$config_path" > "$output_dir/source_config"
    sha256sum "$config_path" > "$output_dir/source_config.sha256"
    echo "starting or resuming $label to optimizer step $target_step from $resume_path"
    /usr/bin/time -v -o "$resource_path" \
        "$python_bin" src/train.py \
        --config "$config_path" \
        --config_override "stop_optimizer_steps=$target_step" \
        --config_override "resume=$resume_path" >> "$log_path" 2>&1
    if [ ! -s "$completion" ]; then
        echo "$label exited without training_complete.json" >&2
        exit 8
    fi
    completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step"))' "$completion")"
    if [ "$completed_step" -ne "$target_step" ]; then
        echo "$label completed step $completed_step, expected $target_step" >&2
        exit 9
    fi
}

evaluate_checkpoint() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local step="$4"
    local expected_samples="$5"
    local checkpoint="$output_dir/checkpoint_$step"
    local eval_dir="$output_dir/evaluations/checkpoint_$step"
    local temporary_dir="${eval_dir}.in_progress"
    local log_path="$output_dir/pipeline.log"
    if ! "$python_bin" scripts/phase5_pipeline_checks.py checkpoint "$checkpoint"; then
        echo "missing $label checkpoint: $checkpoint" >&2
        exit 10
    fi
    if "$python_bin" scripts/phase5_pipeline_checks.py evaluation \
        "$eval_dir" "$label" "$step" "$expected_samples"; then
        echo "$label checkpoint $step evaluation with $expected_samples samples already complete"
        return
    fi
    timestamp="$(date +%Y%m%dT%H%M%S)"
    if [ -e "$eval_dir" ]; then
        mv "$eval_dir" "${eval_dir}.archived_${timestamp}"
    fi
    if [ -e "$temporary_dir" ]; then
        mv "$temporary_dir" "${temporary_dir}.archived_${timestamp}"
    fi
    mkdir -p "$temporary_dir"
    echo "evaluating $label checkpoint $step on $expected_samples validation examples"
    "$python_bin" src/eval.py \
        --config "$config_path" \
        --config_override "output_dir=$temporary_dir" \
        --config_override "num_samples=$expected_samples" \
        --config_override "batch_size=16" \
        --checkpoint_path "$checkpoint" \
        --seed 42 >> "$log_path" 2>&1
    if ! "$python_bin" scripts/phase5_pipeline_checks.py finalize-evaluation \
        "$temporary_dir" "$label" "$step" "$expected_samples"; then
        echo "$label checkpoint $step evaluation artifacts are incomplete" >&2
        exit 11
    fi
    mv "$temporary_dir" "$eval_dir"
}

run_source_diagnostic() {
    local step="$1"
    local output_dir="$wonn_dir/diagnostics/checkpoint_$step"
    local report="$output_dir/diagnostics.json"
    if "$python_bin" scripts/phase5_pipeline_checks.py diagnostic "$report" "WONN-L6T3"; then
        echo "WONN-L6T3 source diagnostic at $step already complete"
        return
    fi
    echo "running WONN-L6T3 correct/shuffled/zero source diagnostic at $step"
    "$python_bin" scripts/diagnose_phase5_checkpoint.py \
        --root "$run_root" \
        --output-dir "$output_dir" \
        --num-samples 256 \
        --batch-size 16 \
        --seed 42 \
        --model-spec "WONN-L6T3|$wonn_config|wonn_l6t3_seed42_b12/checkpoint_$step" \
        --t-values 0.5 \
        --rollout-starts 0.0 \
        --overwrite >> "$wonn_dir/pipeline.log" 2>&1
    if ! "$python_bin" scripts/phase5_pipeline_checks.py diagnostic "$report" "WONN-L6T3"; then
        echo "WONN-L6T3 source diagnostic at $step is incomplete" >&2
        exit 12
    fi
}

monitor_checkpoint() {
    local label="$1"
    local base_dir="$2"
    local output_dir="$3"
    local step="$4"
    local monitor_dir="$output_dir/monitoring/checkpoint_$step"
    local report="$monitor_dir/convergence.json"
    mkdir -p "$monitor_dir"
    "$python_bin" scripts/analyze_phase5_curve_extension.py monitor \
        --label "$label" \
        --base-run-dir "$base_dir" \
        --extension-run-dir "$output_dir" \
        --step "$step" \
        --history-dir "$output_dir/monitoring" \
        --output "$report" \
        --bootstrap-resamples 200
    "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$report"
}

write_terminal_status() {
    local output_dir="$1"
    local label="$2"
    local step="$3"
    local reason="$4"
    "$python_bin" - "$output_dir/terminal_status.json" "$label" "$step" "$reason" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
payload = {"status": "complete", "label": sys.argv[2], "terminal_step": int(sys.argv[3]), "reason": sys.argv[4]}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

train_to_plateau() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local base_dir="$4"
    local cap="$5"
    local base_checkpoint="$base_dir/checkpoint_50000"
    local terminal_step=0
    local terminal_reason="budget_cap"
    local decision="continue"
    for step in $(seq 60000 10000 "$cap"); do
        current_stage="$label training to $step"
        write_status running
        run_training "$label" "$config_path" "$output_dir" "$base_checkpoint" "$step"
        current_stage="$label evaluation at $step"
        write_status running
        evaluate_checkpoint "$label" "$config_path" "$output_dir" "$step" 1000
        if [ "$label" = "WONN-L6T3" ] && [ $((step % 20000)) -eq 0 ]; then
            current_stage="$label source diagnostic at $step"
            write_status running
            run_source_diagnostic "$step"
        fi
        current_stage="$label convergence monitor at $step"
        write_status running
        decision="$(monitor_checkpoint "$label" "$base_dir" "$output_dir" "$step" | tail -n 1)"
        if [ "$decision" = "abort_abnormal" ]; then
            echo "$label abnormal-output monitor failed at step $step" >&2
            exit 30
        fi
        if [ "$decision" = "stop_converged" ]; then
            terminal_step="$step"
            terminal_reason="two_consecutive_plateau_intervals"
            break
        fi
        terminal_step="$step"
    done
    if [ "$terminal_step" -eq 0 ]; then
        echo "$label did not reach its first extension checkpoint" >&2
        exit 31
    fi
    current_stage="$label terminal evaluation at $terminal_step"
    write_status running
    evaluate_checkpoint "$label" "$config_path" "$output_dir" "$terminal_step" 3000
    if [ "$label" = "WONN-L6T3" ]; then
        current_stage="$label terminal source diagnostic at $terminal_step"
        write_status running
        run_source_diagnostic "$terminal_step"
    fi
    write_terminal_status "$output_dir" "$label" "$terminal_step" "$terminal_reason"
}

train_to_plateau "Transformer ELF-B" "$elf_config" "$elf_dir" "$elf_base" 90000
train_to_plateau "WONN-L6T3" "$wonn_config" "$wonn_dir" "$wonn_base" 130000

elf_step="$($python_bin -c 'import json; print(json.load(open("outputs/phase5/curve_to_plateau_v1/elf_b_seed42_b12/terminal_status.json"))["terminal_step"])')"
wonn_step="$($python_bin -c 'import json; print(json.load(open("outputs/phase5/curve_to_plateau_v1/wonn_l6t3_seed42_b12/terminal_status.json"))["terminal_step"])')"
current_stage="final comparison report"
write_status running
"$python_bin" scripts/analyze_phase5_curve_extension.py report \
    --root "$run_root" \
    --elf-base "$elf_base" \
    --wonn-base "$wonn_base" \
    --elf-step "$elf_step" \
    --wonn-step "$wonn_step" \
    --output-dir "$run_root/analysis/final" \
    --bootstrap-resamples 1000

current_stage="complete"
write_status complete 0
completed_at="$(date --iso-8601=seconds)"
printf '{\n  "status": "complete",\n  "completed_at": "%s",\n  "experiment_commit": "%s",\n  "elf_terminal_step": %s,\n  "wonn_terminal_step": %s,\n  "analysis": "%s"\n}\n' \
    "$completed_at" "$experiment_commit" "$elf_step" "$wonn_step" \
    "$run_root/analysis/final/comparison.json" > "$run_root/pipeline_complete.json"
pipeline_done=1

if command -v notify-send >/dev/null 2>&1; then
    notify-send "DLM-WONN Phase 5" "WMT14 ELF 90K / WONN 130K curve experiment completed" || true
fi

#!/usr/bin/env python
"""Monitor convergence and summarize the Phase 5 WMT14 curve extension."""

import argparse
import json
import math
from pathlib import Path
import statistics

from analyze_phase5_50k import (
    _load_evaluation,
    _paired_bootstrap,
    _read_json,
    _read_jsonl,
    _write_csv,
    _write_json,
)


POLICY = {
    "evaluation_interval_steps": 10000,
    "monitor_samples": 1000,
    "terminal_samples": 3000,
    "bleu_ci95_upper_improvement_lt": 0.2,
    "chrf2_ci95_upper_improvement_lt": 0.5,
    "rolling_loss_relative_improvement_lt": 0.01,
    "rolling_loss_window_steps": 5000,
    "required_consecutive_plateau_intervals": 2,
    "maximum_empty_rate_pct": 10.0,
    "minimum_unique_rate_pct": 90.0,
    "length_ratio_range": [0.5, 1.5],
}


def _load_evaluation_from_marker(run_dir: Path, step: int) -> dict:
    marker_path = run_dir / "evaluations" / f"checkpoint_{step}" / "evaluation_complete.json"
    marker = _read_json(marker_path)
    if (
        marker.get("status") != "complete"
        or marker.get("step") != step
        or not isinstance(marker.get("num_samples"), int)
    ):
        raise ValueError(f"{marker_path} is not a complete step-{step} evaluation")
    return _load_evaluation(run_dir, step, marker["num_samples"])


def _aligned_pair(earlier: dict, later: dict, limit: int | None = None) -> tuple[dict, dict]:
    count = min(len(earlier["rows"]), len(later["rows"]))
    if limit is not None:
        count = min(count, limit)
    if count <= 0:
        raise ValueError("paired evaluations have no aligned examples")

    def truncate(evaluation: dict) -> dict:
        rows = evaluation["rows"][:count]
        ids = [row.get("id") for row in rows]
        if ids != list(range(count)):
            raise ValueError("paired evaluation IDs are not consecutive from zero")
        return {
            "sources": [row["source"] for row in rows],
            "references": [row["reference"] for row in rows],
            "hypotheses": [row["generated"] for row in rows],
        }

    return truncate(earlier), truncate(later)


def _merged_training_rows(base_run_dir: Path, extension_run_dir: Path) -> list[dict]:
    by_step = {}
    for path in (
        base_run_dir / "train_metrics.jsonl",
        extension_run_dir / "train_metrics.jsonl",
    ):
        for row in _read_jsonl(path):
            step = row.get("optimizer_step")
            if isinstance(step, int) and not isinstance(step, bool):
                by_step[step] = row
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise ValueError("training metrics are empty")
    required = ("loss", "l2_loss", "ce_loss", "lr", "samples_per_second")
    if any(
        any(
            not isinstance(row.get(field), (int, float))
            or isinstance(row.get(field), bool)
            or not math.isfinite(row[field])
            for field in required
        )
        for row in rows
    ):
        raise ValueError("training metrics contain missing or non-finite values")
    return rows


def _loss_windows(rows: list[dict], step: int) -> dict:
    width = POLICY["rolling_loss_window_steps"]
    previous = [row for row in rows if step - 2 * width < row["optimizer_step"] <= step - width]
    current = [row for row in rows if step - width < row["optimizer_step"] <= step]
    if not previous or not current:
        raise ValueError(f"step {step} lacks two complete rolling loss windows")
    previous_mean = statistics.mean(row["loss"] for row in previous)
    current_mean = statistics.mean(row["loss"] for row in current)
    relative_improvement = (previous_mean - current_mean) / max(abs(previous_mean), 1e-12)
    return {
        "previous": {
            "exclusive_start": step - 2 * width,
            "inclusive_end": step - width,
            "records": len(previous),
            "mean_loss": previous_mean,
        },
        "current": {
            "exclusive_start": step - width,
            "inclusive_end": step,
            "records": len(current),
            "mean_loss": current_mean,
        },
        "relative_improvement": relative_improvement,
    }


def _previous_interval_results(history_dir: Path, step: int) -> list[dict]:
    results = []
    for path in sorted(history_dir.glob("checkpoint_*/convergence.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete" and payload.get("step", step) < step:
            results.append(payload)
    return sorted(results, key=lambda payload: payload["step"])


def evaluate_convergence(
    label: str,
    base_run_dir: Path,
    extension_run_dir: Path,
    step: int,
    history_dir: Path,
    bootstrap_resamples: int,
) -> dict:
    previous_step = step - POLICY["evaluation_interval_steps"]
    previous_dir = base_run_dir if previous_step <= 50000 else extension_run_dir
    previous_eval = _load_evaluation_from_marker(previous_dir, previous_step)
    current_eval = _load_evaluation_from_marker(extension_run_dir, step)
    if len(previous_eval["rows"]) < POLICY["monitor_samples"]:
        raise ValueError(f"step {previous_step} has fewer than 1000 monitor samples")
    if len(current_eval["rows"]) != POLICY["monitor_samples"]:
        raise ValueError(f"step {step} must have exactly 1000 monitor samples")
    previous_pair, current_pair = _aligned_pair(
        previous_eval, current_eval, POLICY["monitor_samples"],
    )
    bootstrap = _paired_bootstrap(
        previous_pair, current_pair, bootstrap_resamples, seed=20260823 + step,
    )
    bleu = bootstrap["bleu_delta_wonn_minus_elf"]
    chrf2 = bootstrap["chrf2_delta_wonn_minus_elf"]
    rows = _merged_training_rows(base_run_dir, extension_run_dir)
    loss = _loss_windows(rows, step)
    current_metrics = current_eval["metrics"]
    abnormal_checks = {
        "empty_rate_exceeds_limit": (
            current_metrics["empty_rate_pct"] > POLICY["maximum_empty_rate_pct"]
        ),
        "unique_rate_below_limit": (
            current_metrics["unique_rate_pct"] < POLICY["minimum_unique_rate_pct"]
        ),
        "length_ratio_outside_range": not (
            POLICY["length_ratio_range"][0]
            <= current_metrics["length_ratio"]
            <= POLICY["length_ratio_range"][1]
        ),
    }
    quality_plateau = (
        bleu["ci95_high"] < POLICY["bleu_ci95_upper_improvement_lt"]
        and chrf2["ci95_high"] < POLICY["chrf2_ci95_upper_improvement_lt"]
    )
    loss_plateau = (
        loss["relative_improvement"]
        < POLICY["rolling_loss_relative_improvement_lt"]
    )
    interval_plateau = quality_plateau and loss_plateau
    prior = _previous_interval_results(history_dir, step)
    plateau_flags = [item.get("interval_plateau", False) for item in prior]
    plateau_flags.append(interval_plateau)
    consecutive = 0
    for flag in reversed(plateau_flags):
        if not flag:
            break
        consecutive += 1
    if any(abnormal_checks.values()):
        decision = "abort_abnormal"
    elif consecutive >= POLICY["required_consecutive_plateau_intervals"]:
        decision = "stop_converged"
    else:
        decision = "continue"
    return {
        "status": "complete",
        "label": label,
        "step": step,
        "previous_step": previous_step,
        "decision": decision,
        "policy": POLICY,
        "paired_monitor_samples": len(current_pair["references"]),
        "current_metrics": current_metrics,
        "bootstrap_current_minus_previous": bootstrap,
        "loss_windows": loss,
        "quality_plateau": quality_plateau,
        "loss_plateau": loss_plateau,
        "interval_plateau": interval_plateau,
        "consecutive_plateau_intervals": consecutive,
        "abnormal_checks": abnormal_checks,
    }


def _evaluation_steps(base_run_dir: Path, extension_run_dir: Path, terminal_step: int) -> list[int]:
    steps = []
    for run_dir in (base_run_dir, extension_run_dir):
        for marker in run_dir.glob("evaluations/checkpoint_*/evaluation_complete.json"):
            try:
                payload = _read_json(marker)
            except (OSError, json.JSONDecodeError):
                continue
            step = payload.get("step")
            if payload.get("status") == "complete" and isinstance(step, int) and step <= terminal_step:
                steps.append(step)
    return sorted(set(steps))


def _diagnostic_rows(extension_run_dir: Path, terminal_step: int) -> list[dict]:
    rows = []
    for path in sorted(extension_run_dir.glob("diagnostics/checkpoint_*/diagnostics.json")):
        payload = _read_json(path)
        if payload.get("status") != "complete":
            continue
        step = int(path.parent.name.split("_")[-1])
        if step > terminal_step:
            continue
        conditioning = payload["models"]["WONN-L6T3"]["conditioning"]
        correct = conditioning["correct"]["chrf2"]
        shuffled = conditioning["shuffled"]["chrf2"]
        zero = conditioning["zero"]["chrf2"]
        rows.append({
            "step": step,
            "correct_chrf2": correct,
            "shuffled_chrf2": shuffled,
            "zero_chrf2": zero,
            "correct_minus_shuffled_chrf2": correct - shuffled,
            "correct_minus_zero_chrf2": correct - zero,
        })
    return rows


def _training_summary(base_run_dir: Path, extension_run_dir: Path, terminal_step: int) -> dict:
    base = _read_json(base_run_dir / "training_complete.json")
    extension = _read_json(extension_run_dir / "training_complete.json")
    if extension.get("completed_optimizer_step") != terminal_step:
        raise ValueError(f"{extension_run_dir} did not complete terminal step {terminal_step}")
    rows = [
        row for row in _merged_training_rows(base_run_dir, extension_run_dir)
        if row["optimizer_step"] <= terminal_step
    ]
    if not rows or rows[-1]["optimizer_step"] != terminal_step:
        raise ValueError(f"training metrics do not end at terminal step {terminal_step}")
    post_warmup = [row for row in rows if row["optimizer_step"] >= 5000]
    return {
        "terminal_step": terminal_step,
        "parameters": extension.get("model_parameters"),
        "total_training_seconds": (
            base["elapsed_training_seconds"] + extension["elapsed_training_seconds"]
        ),
        "extension_training_seconds": extension["elapsed_training_seconds"],
        "median_samples_per_second": statistics.median(
            row["samples_per_second"] for row in post_warmup
        ),
        "peak_allocated_cuda_mib": max(
            value for value in (
                base.get("peak_allocated_cuda_mib"),
                extension.get("peak_allocated_cuda_mib"),
            ) if value is not None
        ),
        "final_loss": rows[-1]["loss"],
        "final_l2_loss": rows[-1]["l2_loss"],
        "final_ce_loss": rows[-1]["ce_loss"],
    }


def build_report(
    root: Path,
    elf_base: Path,
    wonn_base: Path,
    elf_step: int,
    wonn_step: int,
    output_dir: Path,
    bootstrap_resamples: int,
) -> dict:
    specs = {
        "Transformer ELF-B": {
            "base": elf_base,
            "extension": root / "elf_b_seed42_b12",
            "terminal_step": elf_step,
        },
        "WONN-L6T3": {
            "base": wonn_base,
            "extension": root / "wonn_l6t3_seed42_b12",
            "terminal_step": wonn_step,
        },
    }
    evaluations = {}
    evaluation_rows = []
    best = {}
    for label, spec in specs.items():
        steps = _evaluation_steps(spec["base"], spec["extension"], spec["terminal_step"])
        evaluations[label] = {}
        for step in steps:
            run_dir = spec["base"] if step <= 50000 else spec["extension"]
            evaluation = _load_evaluation_from_marker(run_dir, step)
            evaluations[label][step] = evaluation
            evaluation_rows.append({
                "model": label,
                "step": step,
                "num_samples": len(evaluation["rows"]),
                **evaluation["metrics"],
                **{field: evaluation["recorded"].get(field) for field in (
                    "generation_seconds", "decode_seconds", "samples_per_second",
                    "peak_allocated_cuda_mib", "timing_scope",
                )},
            })
        if len(evaluations[label][spec["terminal_step"]]["rows"]) != POLICY["terminal_samples"]:
            raise ValueError(
                f"{label} terminal step {spec['terminal_step']} must have exactly "
                f"{POLICY['terminal_samples']} samples"
            )
        best_step = max(steps, key=lambda value: evaluations[label][value]["metrics"]["bleu"])
        best[label] = {
            "step": best_step,
            "bleu": evaluations[label][best_step]["metrics"]["bleu"],
            "chrf2": evaluations[label][best_step]["metrics"]["chrf2"],
            "checkpoint": str((spec["extension"] if best_step > 50000 else spec["base"]) / f"checkpoint_{best_step}"),
        }

    common_steps = sorted(set(evaluations["Transformer ELF-B"]) & set(evaluations["WONN-L6T3"]))
    common_step = common_steps[-1]
    common_pair = _aligned_pair(
        evaluations["Transformer ELF-B"][common_step],
        evaluations["WONN-L6T3"][common_step],
        POLICY["monitor_samples"],
    )
    terminal_pair = _aligned_pair(
        evaluations["Transformer ELF-B"][elf_step],
        evaluations["WONN-L6T3"][wonn_step],
        POLICY["terminal_samples"],
    )
    same_step_bootstrap = _paired_bootstrap(
        *common_pair, resamples=bootstrap_resamples, seed=20260824,
    )
    terminal_bootstrap = _paired_bootstrap(
        *terminal_pair, resamples=bootstrap_resamples, seed=20260825,
    )
    training = {
        label: _training_summary(
            spec["base"], spec["extension"], spec["terminal_step"],
        )
        for label, spec in specs.items()
    }
    terminal_status = {
        label: _read_json(spec["extension"] / "terminal_status.json")
        for label, spec in specs.items()
    }
    source_diagnostics = _diagnostic_rows(specs["WONN-L6T3"]["extension"], wonn_step)
    payload = {
        "status": "complete",
        "policy": POLICY,
        "terminal_status": terminal_status,
        "training": training,
        "evaluations": evaluation_rows,
        "common_step": common_step,
        "same_step_bootstrap_wonn_minus_elf": same_step_bootstrap,
        "terminal_endpoint_bootstrap_wonn_minus_elf": terminal_bootstrap,
        "best_checkpoints": best,
        "wonn_source_diagnostics": source_diagnostics,
        "interpretation_contract": {
            "same_step_comparison": "fair optimizer-step comparison through the last common checkpoint",
            "terminal_endpoint_comparison": "curve-controlled endpoints with different maximum budgets; not a same-budget claim",
            "monitoring_split": "WMT14 validation; repeated monitoring can bias checkpoint selection",
        },
    }
    _write_json(output_dir / "comparison.json", payload)
    _write_json(output_dir / "best_checkpoints.json", best)
    evaluation_fields = [
        "model", "step", "num_samples", "bleu", "chrf2", "ter",
        "empty_rate_pct", "unique_rate_pct", "mean_generated_words",
        "mean_reference_words", "length_ratio", "generation_seconds",
        "decode_seconds", "samples_per_second", "peak_allocated_cuda_mib",
        "timing_scope",
    ]
    _write_csv(output_dir / "evaluation_summary.csv", evaluation_rows, evaluation_fields)
    if source_diagnostics:
        _write_csv(
            output_dir / "wonn_source_diagnostics.csv",
            source_diagnostics,
            list(source_diagnostics[0]),
        )

    def metric(label: str, step: int, name: str) -> float:
        return evaluations[label][step]["metrics"][name]

    elf_terminal = metric("Transformer ELF-B", elf_step, "bleu")
    wonn_terminal = metric("WONN-L6T3", wonn_step, "bleu")
    markdown = [
        "# WMT14 curve-to-plateau comparison\n",
        "\n## Headline\n",
        f"- Fair same-step checkpoint: **{common_step:,} steps**. WONN minus ELF BLEU "
        f"{metric('WONN-L6T3', common_step, 'bleu') - metric('Transformer ELF-B', common_step, 'bleu'):+.4f}; "
        f"chrF++ {metric('WONN-L6T3', common_step, 'chrf2') - metric('Transformer ELF-B', common_step, 'chrf2'):+.4f}.\n",
        f"- Curve-controlled endpoints: ELF **{elf_step:,}** steps (BLEU {elf_terminal:.4f}), "
        f"WONN **{wonn_step:,}** steps (BLEU {wonn_terminal:.4f}). This endpoint comparison is not a same-budget claim.\n",
        f"- Best observed checkpoints: ELF {best['Transformer ELF-B']['step']:,}, "
        f"WONN {best['WONN-L6T3']['step']:,}.\n",
        "\n## Learning curve\n",
        "| Step | ELF BLEU | WONN BLEU | ELF chrF++ | WONN chrF++ |\n",
        "|---:|---:|---:|---:|---:|\n",
    ]
    for step in common_steps:
        markdown.append(
            f"| {step:,} | {metric('Transformer ELF-B', step, 'bleu'):.4f} | "
            f"{metric('WONN-L6T3', step, 'bleu'):.4f} | "
            f"{metric('Transformer ELF-B', step, 'chrf2'):.4f} | "
            f"{metric('WONN-L6T3', step, 'chrf2'):.4f} |\n"
        )
    markdown.extend([
        "\n## Stop decisions\n",
        f"- ELF: {terminal_status['Transformer ELF-B'].get('reason')} at {elf_step:,}.\n",
        f"- WONN: {terminal_status['WONN-L6T3'].get('reason')} at {wonn_step:,}.\n",
        "\n## Resource summary\n",
        "| Model | Params | Total train h | Extension train h | Median samples/s | Peak CUDA MiB |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])
    for label in specs:
        row = training[label]
        markdown.append(
            f"| {label} | {row['parameters']:,} | {row['total_training_seconds'] / 3600:.2f} | "
            f"{row['extension_training_seconds'] / 3600:.2f} | {row['median_samples_per_second']:.2f} | "
            f"{row['peak_allocated_cuda_mib']:.1f} |\n"
        )
    if source_diagnostics:
        markdown.extend([
            "\n## WONN source conditioning\n",
            "| Step | Correct chrF++ | Shuffled | Zero | Correct-shuffled | Correct-zero |\n",
            "|---:|---:|---:|---:|---:|---:|\n",
        ])
        for row in source_diagnostics:
            markdown.append(
                f"| {row['step']:,} | {row['correct_chrf2']:.4f} | "
                f"{row['shuffled_chrf2']:.4f} | {row['zero_chrf2']:.4f} | "
                f"{row['correct_minus_shuffled_chrf2']:+.4f} | "
                f"{row['correct_minus_zero_chrf2']:+.4f} |\n"
            )
    markdown.extend([
        "\n## Statistical interpretation\n",
        f"- Same-step paired bootstrap uses {same_step_bootstrap['resamples']} resamples on "
        f"{len(common_pair[0]['references'])} aligned validation examples.\n",
        f"- Endpoint paired bootstrap uses {terminal_bootstrap['resamples']} resamples on "
        f"{len(terminal_pair[0]['references'])} aligned validation examples.\n",
        "- The validation split is reused for monitoring, so this report is a new experimental baseline, not an unbiased test-set estimate.\n",
    ])
    (output_dir / "comparison.md").write_text("".join(markdown), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--label", required=True)
    monitor.add_argument("--base-run-dir", type=Path, required=True)
    monitor.add_argument("--extension-run-dir", type=Path, required=True)
    monitor.add_argument("--step", type=int, required=True)
    monitor.add_argument("--history-dir", type=Path, required=True)
    monitor.add_argument("--output", type=Path, required=True)
    monitor.add_argument("--bootstrap-resamples", type=int, default=200)

    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--elf-base", type=Path, required=True)
    report.add_argument("--wonn-base", type=Path, required=True)
    report.add_argument("--elf-step", type=int, required=True)
    report.add_argument("--wonn-step", type=int, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--bootstrap-resamples", type=int, default=1000)

    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    if args.command == "monitor":
        payload = evaluate_convergence(
            args.label,
            args.base_run_dir,
            args.extension_run_dir,
            args.step,
            args.history_dir,
            args.bootstrap_resamples,
        )
        _write_json(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            "decision": payload["decision"],
            "step": payload["step"],
            "output": str(args.output.resolve()),
        }))
        return
    payload = build_report(
        args.root,
        args.elf_base,
        args.wonn_base,
        args.elf_step,
        args.wonn_step,
        args.output_dir,
        args.bootstrap_resamples,
    )
    print(json.dumps({
        "status": payload["status"],
        "output": str((args.output_dir / "comparison.json").resolve()),
    }))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate a Phase 5 run and build dependency-free CSV/SVG analysis artifacts."""

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            records.append(record)
    return records


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _checkpoint_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _checkpoint_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _checkpoint_tensors(child)


def _audit_checkpoint(path: Path, expected_step: int) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = ("params", "ema_params1", "opt_state", "step", "epoch")
    missing = [field for field in required if field not in checkpoint]
    if missing:
        raise ValueError(f"{path} is missing checkpoint fields: {missing}")
    if int(checkpoint["step"]) != expected_step:
        raise ValueError(f"{path} records step {checkpoint['step']}, expected {expected_step}")
    tensors = list(_checkpoint_tensors(checkpoint))
    if not tensors:
        raise ValueError(f"{path} contains no tensors")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError(f"{path} contains non-finite tensor values")
    return {
        "step": expected_step,
        "size_bytes": path.stat().st_size,
        "tensor_count": len(tensors),
        "model_state_entries": len(checkpoint["params"]),
        "ema_state_entries": len(checkpoint["ema_params1"]),
        "optimizer_state_entries": len(checkpoint["opt_state"].get("state", {})),
    }


def _polyline_svg(title: str, x_label: str, series: list, width=960, height=520) -> str:
    """Return a small self-contained SVG line chart."""
    margin_left, margin_right, margin_top, margin_bottom = 86, 28, 58, 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    points = [(x, y) for _, values in series for x, y in values]
    if not points:
        raise ValueError(f"no finite points for chart {title!r}")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    y_padding = (y_max - y_min) * 0.08
    y_min -= y_padding
    y_max += y_padding

    def sx(value):
        return margin_left + (value - x_min) * plot_width / (x_max - x_min)

    def sy(value):
        return margin_top + (y_max - value) * plot_height / (y_max - y_min)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x_pixel = sx(x_value)
        y_pixel = sy(y_value)
        parts.extend([
            f'<line x1="{x_pixel:.1f}" y1="{margin_top}" x2="{x_pixel:.1f}" y2="{margin_top + plot_height}" stroke="#e5e7eb"/>',
            f'<text x="{x_pixel:.1f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.3g}</text>',
            f'<line x1="{margin_left}" y1="{y_pixel:.1f}" x2="{margin_left + plot_width}" y2="{y_pixel:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{margin_left - 12}" y="{y_pixel + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.3g}</text>',
        ])
    parts.extend([
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">{escape(x_label)}</text>',
    ])
    legend_x = margin_left + 12
    for index, (name, values) in enumerate(series):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = margin_top + 18 + index * 20
        parts.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{escape(name)}</text>',
        ])
    parts.append("</svg>\n")
    return "".join(parts)


def _series(rows: list, x_key: str, y_keys: list) -> list:
    result = []
    for y_key in y_keys:
        values = [
            (float(row[x_key]), float(row[y_key]))
            for row in rows
            if _finite_number(row.get(x_key)) and _finite_number(row.get(y_key))
        ]
        if values:
            result.append((y_key, values))
    return result


def summarize_run(
    run_dir: Path, expected_steps: list, batch_size: int, expected_samples: int,
    verify_checkpoints: bool = False,
):
    run_dir = run_dir.resolve()
    training_complete_path = run_dir / "training_complete.json"
    if not training_complete_path.is_file():
        raise FileNotFoundError(f"missing {training_complete_path}")
    training_complete = json.loads(training_complete_path.read_text(encoding="utf-8"))
    final_step = expected_steps[-1]
    if training_complete.get("status") != "complete":
        raise ValueError("training_complete.json does not report complete status")
    if training_complete.get("completed_optimizer_step") != final_step:
        raise ValueError("training_complete.json does not match the expected final optimizer step")

    checkpoint_audits = []
    for step in expected_steps:
        checkpoint = run_dir / f"checkpoint_{step}"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty checkpoint {checkpoint}")
        if verify_checkpoints:
            checkpoint_audits.append(_audit_checkpoint(checkpoint, step))

    training_rows_by_step = {}
    for row in _read_jsonl(run_dir / "train_metrics.jsonl"):
        step = row.get("optimizer_step")
        if isinstance(step, int) and not isinstance(step, bool):
            training_rows_by_step[step] = row
    training_rows = [training_rows_by_step[step] for step in sorted(training_rows_by_step)]
    if not training_rows or training_rows[-1].get("optimizer_step") != final_step:
        raise ValueError("train_metrics.jsonl does not end at the expected optimizer step")
    required_training_fields = {
        "samples_seen", "elapsed_training_seconds", "samples_per_second",
        "loss", "l2_loss", "ce_loss", "lr",
    }
    for field in required_training_fields:
        if not all(_finite_number(row.get(field)) for row in training_rows):
            raise ValueError(f"training metrics are missing finite {field!r} values")

    evaluation_rows = []
    for step in expected_steps:
        eval_dir = run_dir / "evaluations" / f"checkpoint_{step}"
        metric_paths = sorted(eval_dir.glob("*/metrics.jsonl"))
        generated_paths = sorted(eval_dir.glob(f"*/all_generated_*_{step}.jsonl"))
        if not metric_paths or len(metric_paths) != len(generated_paths):
            raise ValueError(f"incomplete evaluation artifacts for checkpoint {step}")
        for metric_path, generated_path in zip(metric_paths, generated_paths):
            with generated_path.open("r", encoding="utf-8") as generated_file:
                generated_count = sum(1 for _ in generated_file)
            if generated_count != expected_samples:
                raise ValueError(
                    f"{generated_path} has {generated_count} samples, expected {expected_samples}"
                )
            matching = [row for row in _read_jsonl(metric_path) if row.get("step") == step]
            if not matching:
                raise ValueError(f"{metric_path} has no metrics for checkpoint {step}")
            metric = matching[-1]
            required_eval_fields = ("bleu", "rouge1", "rouge2", "rougeL")
            if not all(_finite_number(metric.get(field)) for field in required_eval_fields):
                raise ValueError(f"{metric_path} contains non-finite evaluation metrics")
            train_row = training_rows_by_step.get(step)
            if train_row is None:
                raise ValueError(f"no training metric row exists at checkpoint {step}")
            evaluation_rows.append({
                "optimizer_step": step,
                "samples_seen": step * batch_size,
                "elapsed_training_seconds": train_row["elapsed_training_seconds"],
                "sampling_run": metric_path.parent.name,
                "num_samples": generated_count,
                **{field: metric[field] for field in required_eval_fields},
            })

    analysis_dir = run_dir / "analysis"
    training_fields = [
        "step", "optimizer_step", "samples_seen", "elapsed_training_seconds",
        "elapsed_run_seconds", "loss", "l2_loss", "ce_loss", "lr",
        "steps_per_second", "samples_per_second", "timestamp_utc",
    ]
    _write_csv(
        analysis_dir / "training_metrics.csv",
        [{field: row.get(field) for field in training_fields} for row in training_rows],
        training_fields,
    )
    evaluation_fields = [
        "optimizer_step", "samples_seen", "elapsed_training_seconds", "sampling_run",
        "num_samples", "bleu", "rouge1", "rouge2", "rougeL",
    ]
    _write_csv(analysis_dir / "evaluation_metrics.csv", evaluation_rows, evaluation_fields)

    (analysis_dir / "loss_vs_samples.svg").write_text(
        _polyline_svg(
            "Training losses vs new samples", "new samples",
            _series(training_rows, "samples_seen", ["loss", "l2_loss", "ce_loss"]),
        ), encoding="utf-8",
    )
    (analysis_dir / "loss_vs_time.svg").write_text(
        _polyline_svg(
            "Training losses vs training wall-clock", "training seconds",
            _series(training_rows, "elapsed_training_seconds", ["loss", "l2_loss", "ce_loss"]),
        ), encoding="utf-8",
    )
    (analysis_dir / "learning_rate.svg").write_text(
        _polyline_svg(
            "Learning rate schedule", "new samples",
            _series(training_rows, "samples_seen", ["lr"]),
        ), encoding="utf-8",
    )
    (analysis_dir / "throughput.svg").write_text(
        _polyline_svg(
            "Measured training throughput", "new samples",
            _series(training_rows, "samples_seen", ["samples_per_second"]),
        ), encoding="utf-8",
    )
    (analysis_dir / "evaluation_vs_samples.svg").write_text(
        _polyline_svg(
            "Checkpoint quality vs new samples", "new samples",
            _series(evaluation_rows, "samples_seen", ["bleu", "rouge1", "rouge2", "rougeL"]),
        ), encoding="utf-8",
    )
    (analysis_dir / "evaluation_vs_time.svg").write_text(
        _polyline_svg(
            "Checkpoint quality vs training wall-clock", "training seconds",
            _series(
                evaluation_rows, "elapsed_training_seconds",
                ["bleu", "rouge1", "rouge2", "rougeL"],
            ),
        ), encoding="utf-8",
    )

    evaluation_complete = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "checkpoint_steps": expected_steps,
        "num_samples_per_sampling_run": expected_samples,
        "checkpoint_audits": checkpoint_audits,
        "evaluations": evaluation_rows,
    }
    _write_json(run_dir / "evaluation_complete.json", evaluation_complete)
    run_complete = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "training_complete": "training_complete.json",
        "evaluation_complete": "evaluation_complete.json",
        "analysis_dir": "analysis",
        "checkpoint_steps": expected_steps,
    }
    _write_json(run_dir / "run_complete.json", run_complete)
    return run_complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-steps", default="2000,5000,10000,20000")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument(
        "--verify-checkpoints", action="store_true",
        help="Reload every checkpoint and reject missing or non-finite model/EMA/optimizer state.",
    )
    args = parser.parse_args()
    expected_steps = [int(value) for value in args.expected_steps.split(",")]
    if expected_steps != sorted(set(expected_steps)) or any(step <= 0 for step in expected_steps):
        raise ValueError("expected steps must be unique, positive, and increasing")
    summarize_run(
        args.run_dir, expected_steps, args.batch_size, args.expected_samples,
        verify_checkpoints=args.verify_checkpoints,
    )
    print(f"Phase 5 run validated and summarized: {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()

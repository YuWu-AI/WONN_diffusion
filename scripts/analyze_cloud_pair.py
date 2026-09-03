#!/usr/bin/env python
"""Validate and summarize the 20K four-GPU ELF/WONN cloud pair."""

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

import numpy as np
import sacrebleu
import yaml

ELF_LABEL = "Transformer ELF-B"
WONN_LABEL = "WONN-L12K768T3"
CHECKPOINT_STEPS = {
    20000: {
        ELF_LABEL: (10000, 15000, 20000),
        WONN_LABEL: (5000, 10000, 15000, 20000),
    },
}
EXPECTED_STEPS = CHECKPOINT_STEPS[20000]
EXPECTED_SAMPLES = 500


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _metric_bundle(hypotheses: list[str], references: list[str]) -> dict:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be non-empty and aligned")
    generated_words = np.asarray(
        [len(text.split()) for text in hypotheses], dtype=np.float64,
    )
    reference_words = np.asarray(
        [len(text.split()) for text in references], dtype=np.float64,
    )
    return {
        "bleu": sacrebleu.corpus_bleu(
            hypotheses, [references], lowercase=True, use_effective_order=True,
        ).score,
        "chrf2": sacrebleu.corpus_chrf(
            hypotheses, [references], word_order=2,
        ).score,
        "ter": sacrebleu.corpus_ter(
            hypotheses, [references], normalized=True, case_sensitive=False,
        ).score,
        "empty_rate_pct": (
            100.0 * sum(not text.strip() for text in hypotheses) / len(hypotheses)
        ),
        "unique_rate_pct": 100.0 * len(set(hypotheses)) / len(hypotheses),
        "mean_generated_words": float(generated_words.mean()),
        "mean_reference_words": float(reference_words.mean()),
        "length_ratio": float(
            generated_words.sum() / max(reference_words.sum(), 1.0)
        ),
    }


def _load_evaluation(run_dir: Path, step: int, expected_count: int) -> dict:
    eval_dir = run_dir / "evaluations" / f"checkpoint_{step}"
    generated_paths = sorted(eval_dir.glob("*/all_generated_*_*.jsonl"))
    metric_paths = sorted(eval_dir.glob("*/metrics.jsonl"))
    if len(generated_paths) != 1 or len(metric_paths) != 1:
        raise ValueError(
            f"{eval_dir} must contain exactly one generated file and one metrics file"
        )
    rows = _read_jsonl(generated_paths[0])
    if len(rows) != expected_count:
        raise ValueError(
            f"{generated_paths[0]} has {len(rows)} rows, expected {expected_count}"
        )
    if [row.get("id") for row in rows] != list(range(expected_count)):
        raise ValueError(f"{generated_paths[0]} IDs are not consecutive from zero")
    required_text = ("source", "reference", "generated")
    for row in rows:
        if any(not isinstance(row.get(field), str) for field in required_text):
            raise ValueError(
                f"{generated_paths[0]} lacks source/reference/generated text"
            )

    metric_rows = [
        row for row in _read_jsonl(metric_paths[0]) if row.get("step") == step
    ]
    if len(metric_rows) != 1:
        raise ValueError(
            f"{metric_paths[0]} must contain exactly one row for step {step}"
        )
    recorded = metric_rows[0]
    required_recorded = (
        "bleu", "rouge1", "rouge2", "rougeL", "num_samples",
        "generation_seconds", "decode_seconds", "samples_per_second",
    )
    if any(not _finite(recorded.get(field)) for field in required_recorded):
        raise ValueError(
            f"{metric_paths[0]} lacks finite evaluation/resource metrics"
        )
    if recorded.get("timing_scope") != "cuda_synchronized_sampler_and_decoder":
        raise ValueError(f"{metric_paths[0]} does not use synchronized CUDA timing")
    if int(recorded["num_samples"]) != expected_count:
        raise ValueError(f"{metric_paths[0]} records the wrong sample count")

    hypotheses = [row["generated"] for row in rows]
    references = [row["reference"] for row in rows]
    sources = [row["source"] for row in rows]
    recomputed = _metric_bundle(hypotheses, references)
    if not math.isclose(
        recomputed["bleu"], recorded["bleu"], rel_tol=0.0, abs_tol=1e-8,
    ):
        raise ValueError(f"{metric_paths[0]} BLEU does not match generated artifacts")
    return {
        "step": step,
        "rows": rows,
        "hypotheses": hypotheses,
        "references": references,
        "sources": sources,
        "recorded": recorded,
        "metrics": recomputed,
        "sampling_run": generated_paths[0].parent.name,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def validate_pair_configs(
    elf_config_path: Path,
    wonn_config_path: Path,
    effective_batch: int | None = None,
    world_size: int = 2,
    target_steps: int = 20000,
) -> dict:
    if world_size != 2:
        raise ValueError("cloud pair world_size must be 2")
    if target_steps not in CHECKPOINT_STEPS:
        raise ValueError("target steps must be 20000")
    elf = _load_yaml(elf_config_path)
    wonn = _load_yaml(wonn_config_path)
    default_save_steps = "5000,10000,15000,20000"
    selected_steps = CHECKPOINT_STEPS[target_steps]

    for label, config, expected_model in (
        (ELF_LABEL, elf, "ELF-B"),
        (WONN_LABEL, wonn, "ELF-WONN-B"),
    ):
        if config.get("model") != expected_model:
            raise ValueError(f"{label} config has model={config.get('model')!r}")
        if config.get("max_optimizer_steps") != 20000:
            raise ValueError(f"{label} must train to exactly 20000 optimizer steps")
        if config.get("save_optimizer_steps") != default_save_steps:
            raise ValueError(f"{label} has the wrong default checkpoint schedule")
        if config.get("grad_accum_steps") != 1:
            raise ValueError(f"{label} must use grad_accum_steps=1")
        if config.get("global_batch_size") != 24:
            raise ValueError(f"{label} must default to global_batch_size=24")
        if config.get("lr") != 0.0005:
            raise ValueError(f"{label} must use lr=0.0005")
        if config.get("lr_schedule") != "constant":
            raise ValueError(f"{label} must use a constant LR schedule")
        if config.get("warmup_steps") != 3000:
            raise ValueError(f"{label} must use 3000 warmup steps")
        if config.get("num_samples") != 500:
            raise ValueError(f"{label} must default to 500 evaluation samples")
        if config.get("resume") is not None or config.get("init_from") is not None:
            raise ValueError(f"{label} must start from random initialization")
        if config.get("seed") != 42:
            raise ValueError(f"{label} must use seed 42")

    if (
        wonn.get("wonn_num_layers") != 12
        or wonn.get("wonn_num_oscillators") != 768
        or wonn.get("wonn_num_inner_steps") != 3
        or wonn.get("wonn_num_heads") != 12
    ):
        raise ValueError("WONN config is not L12/K768/T3/H12")

    shared_fields = (
        "data_path", "eval_data_path", "data_revision", "eval_data_revision",
        "max_length", "max_input_length", "pad_token", "encoder_model_name",
        "encoder_revision", "tokenizer_revision", "latent_mean", "latent_std",
        "denoiser_p_mean", "denoiser_p_std", "denoiser_noise_scale", "t_eps",
        "time_schedule", "decoder_prob", "decoder_noise_scale", "decoder_p_mean",
        "decoder_p_std", "label_drop_prob", "self_cond_prob", "grad_accum_steps",
        "max_optimizer_steps", "blr", "lr", "lr_schedule", "warmup_steps",
        "optimizer", "seed", "sampling_configs_path", "num_samples",
    )
    mismatches = {
        field: {"elf": elf.get(field), "wonn": wonn.get(field)}
        for field in shared_fields
        if elf.get(field) != wonn.get(field)
    }
    if mismatches:
        raise ValueError(f"paired config mismatch: {mismatches}")

    configured_batches = {elf.get("global_batch_size"), wonn.get("global_batch_size")}
    if effective_batch is None:
        if len(configured_batches) != 1:
            raise ValueError("ELF and WONN global_batch_size values differ")
        effective_batch = configured_batches.pop()
    if (
        isinstance(effective_batch, bool)
        or not isinstance(effective_batch, int)
        or effective_batch <= 0
    ):
        raise ValueError("effective batch must be a positive integer")
    if effective_batch % world_size != 0:
        raise ValueError(
            f"effective batch {effective_batch} is not divisible by world size {world_size}"
        )

    return {
        "status": "valid",
        "terminal_optimizer_step": target_steps,
        "checkpoint_steps": {
            label: list(selected_steps[label]) for label in (ELF_LABEL, WONN_LABEL)
        },
        "world_size_per_model": world_size,
        "effective_batch_size": effective_batch,
        "batch_size_per_device": effective_batch // world_size,
        "grad_accum_steps": 1,
        "learning_rate": 0.0005,
        "warmup_steps": 3000,
        "seed": 42,
        "data_revision": elf["data_revision"],
        "eval_data_revision": elf["eval_data_revision"],
        "encoder_revision": elf["encoder_revision"],
        "tokenizer_revision": elf["tokenizer_revision"],
        "optimizer": elf["optimizer"],
    }


def _training_rows(run_dir: Path, checkpoint_steps: tuple[int, ...]) -> dict[int, dict]:
    rows = {}
    path = run_dir / "train_metrics.jsonl"
    previous_step = -1
    previous_samples = -1
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            step = row.get("optimizer_step")
            samples_seen = row.get("samples_seen")
            if (
                not isinstance(step, int) or isinstance(step, bool)
                or not isinstance(samples_seen, int) or isinstance(samples_seen, bool)
            ):
                raise ValueError(f"{path}:{line_number} lacks integer step/samples_seen")
            if step <= previous_step or samples_seen <= previous_samples:
                raise ValueError(f"{path} is not strictly monotonic at line {line_number}")
            if step in rows:
                raise ValueError(f"{path} repeats optimizer step {step}")
            rows[step] = row
            previous_step = step
            previous_samples = samples_seen
    missing = [step for step in checkpoint_steps if step not in rows]
    if missing:
        raise ValueError(f"{path} lacks checkpoint-step metrics: {missing}")
    return rows


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _panel_svg(title: str, x_label: str, series: list, x: int, y: int) -> str:
    width, height = 470, 285
    left, right, top, bottom = x + 58, x + width - 18, y + 35, y + height - 45
    points = [point for _, values in series for point in values]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = max((y_max - y_min) * 0.08, 0.1)
    y_min -= padding
    y_max += padding

    def px(value):
        return left + (value - x_min) * (right - left) / max(x_max - x_min, 1)

    def py(value):
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    colors = ("#2563eb", "#dc2626")
    parts = [
        f'<text x="{x + width / 2:.1f}" y="{y + 18}" text-anchor="middle" '
        f'font-size="15" font-weight="600">{escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444"/>',
        f'<text x="{(left + right) / 2:.1f}" y="{y + height - 8}" text-anchor="middle" '
        f'font-size="12">{escape(x_label)}</text>',
        f'<text x="{x + 14}" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
        f'font-size="12" transform="rotate(-90 {x + 14} {(top + bottom) / 2:.1f})">score</text>',
        f'<text x="{left}" y="{bottom + 17}" text-anchor="middle" font-size="10">{x_min:,.0f}</text>',
        f'<text x="{right}" y="{bottom + 17}" text-anchor="middle" font-size="10">{x_max:,.0f}</text>',
        f'<text x="{left - 7}" y="{bottom + 4}" text-anchor="end" font-size="10">{y_min:.1f}</text>',
        f'<text x="{left - 7}" y="{top + 4}" text-anchor="end" font-size="10">{y_max:.1f}</text>',
    ]
    for index, (_, values) in enumerate(series):
        encoded = " ".join(f"{px(xv):.2f},{py(yv):.2f}" for xv, yv in values)
        parts.append(
            f'<polyline points="{encoded}" fill="none" stroke="{colors[index]}" '
            'stroke-width="2.5" stroke-linejoin="round"/>'
        )
        parts.extend(
            f'<circle cx="{px(xv):.2f}" cy="{py(yv):.2f}" r="3" fill="{colors[index]}"/>'
            for xv, yv in values
        )
    return "".join(parts)


def _quality_chart(rows: list[dict], target_steps: int) -> str:
    by_model = {
        label: [row for row in rows if row["model"] == label]
        for label in (ELF_LABEL, WONN_LABEL)
    }
    panels = []
    for metric, title, x_field, x_label, x, y in (
        ("bleu", "BLEU vs optimizer step", "optimizer_step", "optimizer step", 20, 75),
        ("chrf2", "chrF++ vs optimizer step", "optimizer_step", "optimizer step", 530, 75),
        ("bleu", "BLEU vs samples seen", "samples_seen", "samples seen", 20, 375),
        ("chrf2", "chrF++ vs samples seen", "samples_seen", "samples seen", 530, 375),
    ):
        panels.append(_panel_svg(
            title,
            x_label,
            [
                (label, [(row[x_field], row[metric]) for row in by_model[label]])
                for label in (ELF_LABEL, WONN_LABEL)
            ],
            x,
            y,
        ))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="700" '
        'viewBox="0 0 1020 700"><rect width="100%" height="100%" fill="white"/>'
        '<text x="510" y="28" text-anchor="middle" font-size="20" font-weight="700">'
        f'WMT14 cloud pair quality curves ({target_steps // 1000}K)</text>'
        '<line x1="345" y1="51" x2="375" y2="51" stroke="#2563eb" stroke-width="3"/>'
        f'<text x="382" y="55" font-size="12">{ELF_LABEL}</text>'
        '<line x1="555" y1="51" x2="585" y2="51" stroke="#dc2626" stroke-width="3"/>'
        f'<text x="592" y="55" font-size="12">{WONN_LABEL}</text>'
        + "".join(panels)
        + "</svg>\n"
    )


def analyze_pair(
    elf_run_dir: Path,
    wonn_run_dir: Path,
    output_dir: Path,
    expected_samples: int = EXPECTED_SAMPLES,
) -> dict:
    run_specs = (
        (ELF_LABEL, elf_run_dir, "ELF-B"),
        (WONN_LABEL, wonn_run_dir, "ELF-WONN-B"),
    )
    completions = {}
    configs = {}
    for label, run_dir, model in run_specs:
        completion = _read_json(run_dir / "training_complete.json")
        if completion.get("status") != "complete":
            raise ValueError(f"{label} training is not complete")
        if completion.get("model") != model:
            raise ValueError(f"{label} completion marker has the wrong model")
        completions[label] = completion
        configs[label] = _load_yaml(run_dir / "config.yml")

    targets = {config.get("max_optimizer_steps") for config in configs.values()}
    if len(targets) != 1 or next(iter(targets)) not in CHECKPOINT_STEPS:
        raise ValueError("resolved configs must share the 20000 target")
    target_steps = targets.pop()
    checkpoint_steps = CHECKPOINT_STEPS[target_steps]
    for label in (ELF_LABEL, WONN_LABEL):
        expected_save_steps = ",".join(str(step) for step in checkpoint_steps[label])
        if configs[label].get("save_optimizer_steps") != expected_save_steps:
            raise ValueError(f"{label} resolved config has the wrong checkpoint schedule")
    training = {
        label: _training_rows(run_dir, checkpoint_steps[label])
        for label, run_dir, _ in run_specs
    }
    for label in (ELF_LABEL, WONN_LABEL):
        if completions[label].get("completed_optimizer_step") != target_steps:
            raise ValueError(f"{label} training is not complete at {target_steps}")

    effective_batches = {
        completion.get("effective_batch_size") for completion in completions.values()
    }
    if len(effective_batches) != 1:
        raise ValueError("ELF and WONN effective batch sizes differ")
    effective_batch = effective_batches.pop()
    if not isinstance(effective_batch, int) or effective_batch <= 0:
        raise ValueError("completion markers lack a valid effective batch size")
    if effective_batch != 24:
        raise ValueError("paired run must use effective batch 24")

    fairness_fields = (
        "global_batch_size", "batch_size", "grad_accum_steps", "lr", "blr", "lr_schedule",
        "warmup_steps", "optimizer", "seed", "data_revision", "eval_data_revision",
        "encoder_revision", "tokenizer_revision", "sampling_configs_path",
    )
    mismatches = {
        field: {
            "elf": configs[ELF_LABEL].get(field),
            "wonn": configs[WONN_LABEL].get(field),
        }
        for field in fairness_fields
        if configs[ELF_LABEL].get(field) != configs[WONN_LABEL].get(field)
    }
    if mismatches:
        raise ValueError(f"resolved paired config mismatch: {mismatches}")
    fixed_contract = {
        "grad_accum_steps": 1,
        "lr": 0.0005,
        "lr_schedule": "constant",
        "warmup_steps": 3000,
        "seed": 42,
        "num_samples": 500,
    }
    for label in (ELF_LABEL, WONN_LABEL):
        wrong = {
            field: {"actual": configs[label].get(field), "expected": expected}
            for field, expected in fixed_contract.items()
            if configs[label].get(field) != expected
        }
        if wrong:
            raise ValueError(f"{label} violates the fixed paired contract: {wrong}")
    if configs[ELF_LABEL].get("global_batch_size") != effective_batch:
        raise ValueError("resolved config and completion effective batch disagree")
    if (
        configs[WONN_LABEL].get("wonn_num_layers") != 12
        or configs[WONN_LABEL].get("wonn_num_oscillators") != 768
        or configs[WONN_LABEL].get("wonn_num_inner_steps") != 3
        or configs[WONN_LABEL].get("wonn_num_heads") != 12
    ):
        raise ValueError("resolved WONN config is not L12/K768/T3/H12")
    for label in (ELF_LABEL, WONN_LABEL):
        if configs[label].get("resume") is not None or configs[label].get("init_from") is not None:
            raise ValueError(f"{label} resolved config is not from scratch")
        if completions[label].get("samples_seen") != target_steps * effective_batch:
            raise ValueError(f"{label} completion marker has the wrong samples_seen")
        if completions[label].get("batch_size_per_device") != configs[label].get("batch_size"):
            raise ValueError(f"{label} completion marker has the wrong per-device batch")
        if completions[label].get("learning_rate") != configs[label].get("lr"):
            raise ValueError(f"{label} completion marker has the wrong learning rate")
        if completions[label].get("world_size") != 2:
            raise ValueError(f"{label} completion marker must use world size 2")
    if completions[ELF_LABEL]["world_size"] != completions[WONN_LABEL]["world_size"]:
        raise ValueError("ELF and WONN world sizes differ")

    rows = []
    paired_inputs = {}
    for label, run_dir, _ in run_specs:
        for step in checkpoint_steps[label]:
            checkpoint = run_dir / f"checkpoint_{step}"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty checkpoint {checkpoint}")
            marker = _read_json(
                run_dir / "evaluations" / f"checkpoint_{step}" / "evaluation_complete.json"
            )
            if (
                marker.get("status") != "complete"
                or marker.get("model") != label
                or marker.get("step") != step
                or marker.get("num_samples") != expected_samples
            ):
                raise ValueError(f"{label} checkpoint {step} evaluation is incomplete")
            evaluation = _load_evaluation(run_dir, step, expected_samples)
            pair_key = (evaluation["sources"], evaluation["references"])
            if step in paired_inputs and paired_inputs[step] != pair_key:
                raise ValueError(f"ELF/WONN evaluation inputs differ at step {step}")
            paired_inputs[step] = pair_key
            train_row = training[label][step]
            required_training = (
                "loss", "l2_loss", "ce_loss", "lr", "samples_seen",
                "elapsed_training_seconds", "samples_per_second",
            )
            if any(not _finite(train_row.get(field)) for field in required_training):
                raise ValueError(f"{label} has non-finite training metrics at step {step}")
            metrics = evaluation["metrics"]
            recorded = evaluation["recorded"]
            rows.append({
                "model": label,
                "optimizer_step": step,
                "samples_seen": train_row["samples_seen"],
                "loss": train_row["loss"],
                "l2_loss": train_row["l2_loss"],
                "ce_loss": train_row["ce_loss"],
                "learning_rate": train_row["lr"],
                "training_samples_per_second": train_row["samples_per_second"],
                "elapsed_training_seconds": train_row["elapsed_training_seconds"],
                "bleu": metrics["bleu"],
                "chrf2": metrics["chrf2"],
                "ter": metrics["ter"],
                "empty_rate_pct": metrics["empty_rate_pct"],
                "unique_rate_pct": metrics["unique_rate_pct"],
                "length_ratio": metrics["length_ratio"],
                "evaluation_samples_per_second": recorded["samples_per_second"],
                "sampler_latency_seconds": recorded["generation_seconds"],
                "decode_latency_seconds": recorded["decode_seconds"],
                "evaluation_peak_allocated_cuda_mib": recorded.get(
                    "peak_allocated_cuda_mib"
                ),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = list(rows[0])
    with (output_dir / "metrics_by_checkpoint.csv").open(
        "w", encoding="utf-8", newline="",
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "quality_curves.svg").write_text(
        _quality_chart(rows, target_steps), encoding="utf-8",
    )

    terminal = {
        label: next(
            row for row in rows
            if row["model"] == label and row["optimizer_step"] == target_steps
        )
        for label in (ELF_LABEL, WONN_LABEL)
    }
    payload = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_contract": {
            "terminal_optimizer_step": target_steps,
            "checkpoint_steps": {
                label: list(checkpoint_steps[label])
                for label in (ELF_LABEL, WONN_LABEL)
            },
            "samples_per_model_per_checkpoint": expected_samples,
            "effective_batch_size": effective_batch,
            "seed": configs[ELF_LABEL]["seed"],
            "data_revision": configs[ELF_LABEL]["data_revision"],
            "eval_data_revision": configs[ELF_LABEL]["eval_data_revision"],
            "encoder_revision": configs[ELF_LABEL]["encoder_revision"],
            "tokenizer_revision": configs[ELF_LABEL]["tokenizer_revision"],
            "learning_rate": configs[ELF_LABEL]["lr"],
        },
        "models": {
            label: {
                "parameters": completions[label].get("model_parameters"),
                "world_size": completions[label]["world_size"],
                "batch_size_per_device": configs[label].get("batch_size"),
                "elapsed_training_seconds": completions[label].get(
                    "elapsed_training_seconds"
                ),
                "peak_allocated_cuda_mib": completions[label].get(
                    "peak_allocated_cuda_mib"
                ),
                "terminal_metrics": terminal[label],
            }
            for label in (ELF_LABEL, WONN_LABEL)
        },
        "metrics_by_checkpoint": rows,
        "artifacts": {
            "table_csv": "metrics_by_checkpoint.csv",
            "quality_chart": "quality_curves.svg",
            "markdown": "comparison.md",
        },
    }
    _write_json(output_dir / "comparison.json", payload)
    markdown = [
        f"# WMT14 {target_steps // 1000}K cloud pair\n\n",
        f"Both models use effective batch {effective_batch} and seed 42; ELF lacks the already-passed 5K checkpoint.\n\n",
        "| Model | BLEU | chrF++ | TER | Empty % | Unique % | Length ratio |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for label in (ELF_LABEL, WONN_LABEL):
        row = terminal[label]
        markdown.append(
            f"| {label} | {row['bleu']:.4f} | {row['chrf2']:.4f} | "
            f"{row['ter']:.4f} | {row['empty_rate_pct']:.2f} | "
            f"{row['unique_rate_pct']:.2f} | {row['length_ratio']:.4f} |\n"
        )
    markdown.extend(["\n## Curves\n\n", "![Quality curves](quality_curves.svg)\n"])
    (output_dir / "comparison.md").write_text("".join(markdown), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-configs")
    validate.add_argument("--elf-config", type=Path, required=True)
    validate.add_argument("--wonn-config", type=Path, required=True)
    validate.add_argument("--effective-batch", type=int)
    validate.add_argument("--world-size", type=int, required=True)
    validate.add_argument("--target-steps", type=int, default=20000)
    validate.add_argument("--output", type=Path)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--elf-run-dir", type=Path, required=True)
    analyze.add_argument("--wonn-run-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLES)
    args = parser.parse_args()

    if args.command == "validate-configs":
        payload = validate_pair_configs(
            args.elf_config, args.wonn_config, args.effective_batch, args.world_size,
            args.target_steps,
        )
        if args.output:
            _write_json(args.output, payload)
    else:
        payload = analyze_pair(
            args.elf_run_dir, args.wonn_run_dir, args.output_dir,
            args.expected_samples,
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

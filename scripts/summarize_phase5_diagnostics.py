#!/usr/bin/env python
"""Summarize Phase 5 checkpoint diagnostics into reviewed tables."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase5_50k import _metric_bundle, _paired_bootstrap


RUN_PATTERN = re.compile(r"ode-steps(?P<steps>\d+)-cfg(?P<cfg>[0-9.]+)-ts_logit_normal-cond")


def _read_json(path: Path):
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


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _generated_file(directory: Path) -> Path:
    matches = sorted(directory.glob("all_generated_*_50000.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one generated JSONL in {directory}, got {matches}")
    return matches[0]


def _paired_payload(rows: list[dict]) -> dict:
    return {
        "sources": [row["source"] for row in rows],
        "references": [row["reference"] for row in rows],
        "hypotheses": [row["generated"] for row in rows],
    }


def _evaluate_generated_rows(rows: list[dict]) -> dict:
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("generated IDs must be ordered and contiguous")
    return {
        "num_samples": len(rows),
        **_metric_bundle(
            [row["generated"] for row in rows],
            [row["reference"] for row in rows],
        ),
    }


def _summarize_sampler_root(root: Path, stage: str) -> tuple[list[dict], dict]:
    results = []
    paired = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = RUN_PATTERN.fullmatch(directory.name)
        if not match:
            continue
        generated_rows = _read_jsonl(_generated_file(directory))
        metric_rows = _read_jsonl(directory / "metrics.jsonl")
        if len(metric_rows) != 1:
            raise ValueError(f"expected one metrics row in {directory}")
        metrics = _evaluate_generated_rows(generated_rows)
        result = {
            "stage": stage,
            "steps": int(match.group("steps")),
            "cfg": float(match.group("cfg")),
            **metrics,
            "generation_seconds": float(metric_rows[0]["generation_seconds"]),
            "decode_seconds": float(metric_rows[0]["decode_seconds"]),
            "samples_per_second": float(metric_rows[0]["samples_per_second"]),
            "peak_allocated_cuda_mib": float(metric_rows[0]["peak_allocated_cuda_mib"]),
        }
        results.append(result)
        paired[(result["steps"], result["cfg"])] = _paired_payload(generated_rows)
    if not results:
        raise ValueError(f"no sampler results found in {root}")
    results.sort(key=lambda row: (-row["bleu"], -row["chrf2"], row["steps"], row["cfg"]))
    return results, paired


def _conditioning_tables(diagnostics: dict, bootstrap_resamples: int):
    metric_rows = []
    bootstrap_rows = []
    for model, model_payload in diagnostics["models"].items():
        model_key = model.lower().replace("-", "_")
        condition_dir = Path(diagnostics["_diagnostics_root"]) / model_key / "conditioning"
        paired = {
            variant: _paired_payload(_read_jsonl(condition_dir / f"{variant}.jsonl"))
            for variant in ("correct", "shuffled", "zero")
        }
        for variant in ("correct", "shuffled", "zero"):
            metrics = model_payload["conditioning"][variant]
            metric_rows.append({
                "model": model,
                "condition": variant,
                **{key: metrics[key] for key in (
                    "bleu", "chrf2", "ter", "empty_rate_pct", "unique_rate_pct",
                    "length_ratio", "generation_seconds", "samples_per_second",
                )},
                "final_latent_rms_delta_vs_correct": metrics.get(
                    "final_latent_rms_delta_vs_correct", 0.0
                ),
                "decoded_token_disagreement_pct_vs_correct": metrics.get(
                    "decoded_token_disagreement_pct_vs_correct", 0.0
                ),
            })
        for variant in ("shuffled", "zero"):
            bootstrap = _paired_bootstrap(
                paired["correct"], paired[variant],
                resamples=bootstrap_resamples,
                seed=42,
            )
            for metric_key, label in (
                ("bleu_delta_wonn_minus_elf", "BLEU"),
                ("chrf2_delta_wonn_minus_elf", "chrF++"),
            ):
                values = bootstrap[metric_key]
                bootstrap_rows.append({
                    "model": model,
                    "contrast": f"{variant}_minus_correct",
                    "metric": label,
                    "mean_delta": values["mean"],
                    "ci95_low": values["ci95_low"],
                    "ci95_high": values["ci95_high"],
                    "probability_gt_zero": values["probability_delta_gt_zero"],
                    "resamples": bootstrap_resamples,
                })
    return metric_rows, bootstrap_rows


def _flatten_model_tables(diagnostics: dict):
    decoder_rows = []
    time_rows = []
    rollout_rows = []
    for model, payload in diagnostics["models"].items():
        for probe in ("clean_x0", "training_noised_x0"):
            decoder_rows.append({"model": model, "probe": probe, **payload["decoder_probe"][probe]})
        for row in payload["time_bins"]:
            time_rows.append({"model": model, **row})
        for row in payload["oracle_start_rollout"]:
            rollout_rows.append({"model": model, **row})
    return decoder_rows, time_rows, rollout_rows


def _loss_alignment(final_comparison: dict, decoder_rows: list[dict]) -> list[dict]:
    probe_by_model = {
        row["model"]: row
        for row in decoder_rows if row["probe"] == "training_noised_x0"
    }
    terminal_eval = {
        row["model"]: row
        for row in final_comparison["evaluations"]
        if row["step"] == 50000
    }
    rows = []
    for model in ("ELF-B", "WONN-L6T3"):
        training = final_comparison["training"][model]
        rows.append({
            "model": model,
            "logged_train_total_loss": training["final_loss"],
            "logged_train_l2_loss": training["final_l2_loss"],
            "logged_train_ce_loss": training["final_ce_loss"],
            "heldout_decoder_probe_ce": probe_by_model[model]["ce_loss"],
            "heldout_decoder_probe_bleu": probe_by_model[model]["bleu"],
            "free_sampling_bleu_3000": terminal_eval[model]["bleu"],
            "free_sampling_chrf2_3000": terminal_eval[model]["chrf2"],
        })
    return rows


def _sampler_confirmation_bootstrap(paired: dict, resamples: int) -> list[dict]:
    cheap = paired[(16, 1.0)]
    expensive = paired[(128, 2.0)]
    bootstrap = _paired_bootstrap(cheap, expensive, resamples=resamples, seed=42)
    rows = []
    for metric_key, label in (
        ("bleu_delta_wonn_minus_elf", "BLEU"),
        ("chrf2_delta_wonn_minus_elf", "chrF++"),
    ):
        values = bootstrap[metric_key]
        rows.append({
            "contrast": "128-step/CFG2 minus 16-step/CFG1",
            "metric": label,
            "mean_delta": values["mean"],
            "ci95_low": values["ci95_low"],
            "ci95_high": values["ci95_high"],
            "probability_gt_zero": values["probability_delta_gt_zero"],
            "resamples": resamples,
        })
    return rows


def summarize(args) -> dict:
    diagnostics = _read_json(args.diagnostics_root / "diagnostics.json")
    if diagnostics.get("status") != "complete":
        raise ValueError("checkpoint diagnostics are not complete")
    diagnostics["_diagnostics_root"] = str(args.diagnostics_root)
    final_comparison = _read_json(args.final_comparison)
    sweep_rows, _ = _summarize_sampler_root(args.sampler_sweep_root, "exploration_128")
    confirm_rows, confirm_paired = _summarize_sampler_root(
        args.sampler_confirm_root, "confirmation_1000"
    )
    conditioning_rows, conditioning_bootstrap = _conditioning_tables(
        diagnostics, args.bootstrap_resamples
    )
    decoder_rows, time_rows, rollout_rows = _flatten_model_tables(diagnostics)
    loss_rows = _loss_alignment(final_comparison, decoder_rows)
    sampler_bootstrap = _sampler_confirmation_bootstrap(
        confirm_paired, args.bootstrap_resamples
    )

    tables = {
        "sampler_sweep": sweep_rows,
        "sampler_confirm": confirm_rows,
        "sampler_confirm_bootstrap": sampler_bootstrap,
        "conditioning": conditioning_rows,
        "conditioning_bootstrap": conditioning_bootstrap,
        "decoder_probe": decoder_rows,
        "time_bins": time_rows,
        "oracle_start_rollout": rollout_rows,
        "loss_alignment": loss_rows,
    }
    for name, rows in tables.items():
        _write_csv(args.output_dir / f"{name}.csv", rows)

    best_sweep = sweep_rows[0]
    confirm_by_key = {(row["steps"], row["cfg"]): row for row in confirm_rows}
    cheap = confirm_by_key[(16, 1.0)]
    expensive = confirm_by_key[(128, 2.0)]
    result = {
        "status": "complete",
        "bootstrap_resamples": args.bootstrap_resamples,
        "headline": {
            "best_exploration_sampler": best_sweep,
            "cheap_confirm": cheap,
            "expensive_confirm": expensive,
            "expensive_to_cheap_generation_time_ratio": (
                expensive["generation_seconds"] / cheap["generation_seconds"]
            ),
        },
        "tables": tables,
    }
    _atomic_write_json(args.output_dir / "diagnostic_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-root", type=Path,
        default=Path("outputs/phase5/formal50k/analysis/diagnostics_v1"),
    )
    parser.add_argument(
        "--sampler-sweep-root", type=Path,
        default=Path("outputs/phase5/formal50k/analysis/sampler_sweep_wonn_128"),
    )
    parser.add_argument(
        "--sampler-confirm-root", type=Path,
        default=Path("outputs/phase5/formal50k/analysis/sampler_confirm_wonn_1000"),
    )
    parser.add_argument(
        "--final-comparison", type=Path,
        default=Path("outputs/phase5/formal50k/analysis/final50k/comparison.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/phase5/formal50k/analysis/diagnostic_readout"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(args)
    print(json.dumps({
        "status": result["status"],
        "output": str((args.output_dir / "diagnostic_summary.json").resolve()),
        "best_exploration_sampler": result["headline"]["best_exploration_sampler"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

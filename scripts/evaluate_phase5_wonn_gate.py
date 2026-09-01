#!/usr/bin/env python
"""Evaluate the WONN-only 10K pilot and 20K continuation gates."""

import argparse
import json
from pathlib import Path

from analyze_phase5_50k import (
    MODEL_SPECS,
    _load_run,
    _read_json,
    _write_json,
)


STAGES = {
    "pilot10k": {
        "target_step": 10000,
        "checkpoint_steps": (5000, 10000),
        "evaluation_steps": (5000, 10000),
        "expected_samples": {5000: 1000, 10000: 1000},
        "previous_step": 5000,
        "minimum_bleu": 0.1,
        "minimum_chrf2_exclusive": 13.793431945999787,
    },
    "gate20k": {
        "target_step": 20000,
        "checkpoint_steps": (5000, 10000, 15000, 20000),
        "evaluation_steps": (10000, 20000),
        "expected_samples": {10000: 1000, 20000: 1000},
        "previous_step": 10000,
        "minimum_bleu": 1.0,
        "minimum_chrf2_exclusive": 14.723611391748296,
    },
}


def _conditioning_metrics(diagnostics: dict) -> dict:
    if diagnostics.get("status") != "complete":
        raise ValueError("conditioning diagnostics are not complete")
    if diagnostics.get("num_samples") != 256 or diagnostics.get("seed") != 42:
        raise ValueError("conditioning diagnostics use the wrong sample count or seed")
    models = diagnostics.get("models", {})
    if set(models) != {"WONN-L6T3"}:
        raise ValueError("diagnostics must contain only WONN-L6T3")
    conditioning = models["WONN-L6T3"].get("conditioning", {})
    for variant in ("correct", "shuffled", "zero", "shuffle_construction"):
        if variant not in conditioning:
            raise ValueError(f"diagnostics lack conditioning variant {variant}")
    if not conditioning["shuffle_construction"].get("fixed_target_offsets"):
        raise ValueError("shuffled-source diagnostic changed target offsets")
    return conditioning


def _evaluate_gate(run: dict, diagnostics: dict, stage: dict) -> dict:
    target_step = stage["target_step"]
    previous_step = stage["previous_step"]
    previous = run["evaluations"][previous_step]["metrics"]
    current = run["evaluations"][target_step]["metrics"]
    conditioning = _conditioning_metrics(diagnostics)
    correct_chrf2 = conditioning["correct"]["chrf2"]
    shuffled_chrf2 = conditioning["shuffled"]["chrf2"]
    zero_chrf2 = conditioning["zero"]["chrf2"]
    checks = {
        "wonn_empty_rate_at_most_10_pct": current["empty_rate_pct"] <= 10.0,
        "wonn_unique_rate_at_least_90_pct": current["unique_rate_pct"] >= 90.0,
        "wonn_length_ratio_between_0_5_and_1_5": (
            0.5 <= current["length_ratio"] <= 1.5
        ),
        "wonn_bleu_reaches_stage_minimum": (
            current["bleu"] >= stage["minimum_bleu"]
        ),
        "wonn_chrf2_exceeds_legacy_same_step": (
            current["chrf2"] > stage["minimum_chrf2_exclusive"]
        ),
        "wonn_bleu_improves_from_previous_stage": (
            current["bleu"] > previous["bleu"]
        ),
        "wonn_chrf2_improves_from_previous_stage": (
            current["chrf2"] > previous["chrf2"]
        ),
        "correct_source_beats_shuffled_source_by_chrf2_0_25": (
            correct_chrf2 - shuffled_chrf2 >= 0.25
        ),
        "correct_source_beats_zero_source_by_chrf2_0_25": (
            correct_chrf2 - zero_chrf2 >= 0.25
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "maximum_empty_rate_pct": 10.0,
            "minimum_unique_rate_pct": 90.0,
            "length_ratio_range": [0.5, 1.5],
            "minimum_bleu": stage["minimum_bleu"],
            "minimum_chrf2_exclusive": stage["minimum_chrf2_exclusive"],
            "minimum_conditioning_chrf2_delta": 0.25,
        },
        "observed": {
            "previous_step": previous_step,
            "target_step": target_step,
            "previous_metrics": previous,
            "target_metrics": current,
            "conditioning_chrf2": {
                "correct": correct_chrf2,
                "shuffled": shuffled_chrf2,
                "zero": zero_chrf2,
                "correct_minus_shuffled": correct_chrf2 - shuffled_chrf2,
                "correct_minus_zero": correct_chrf2 - zero_chrf2,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()

    manifest_path = args.root / "experiment_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "ready":
        raise ValueError(f"{manifest_path} is not ready")
    stage = STAGES[args.stage]
    spec = MODEL_SPECS["WONN-L6T3"]
    run = _load_run(
        args.root / spec["directory"], "WONN-L6T3", spec, stage,
        args.verify_checkpoints,
    )
    diagnostics = _read_json(args.diagnostics)
    gate = _evaluate_gate(run, diagnostics, stage)
    payload = {
        "status": "complete",
        "stage": args.stage,
        "experiment_manifest": manifest,
        "training": run["training_summary"],
        "gate": gate,
    }
    report_path = args.output_dir / "comparison.json"
    _write_json(report_path, payload)
    markdown = [
        f"# WONN {args.stage} gate\n",
        f"Gate passed: **{gate['passed']}**\n",
    ]
    for name, passed in gate["checks"].items():
        markdown.append(f"- {name}: {passed}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.md").write_text(
        "".join(markdown), encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "stage": args.stage,
        "gate_passed": gate["passed"],
        "output": str(report_path.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

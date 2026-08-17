#!/usr/bin/env python
"""Capture the immutable source, data, dependency, and hardware run contract."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess

import torch
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_manifest(repo_root: Path, config_paths: list[Path]) -> dict:
    repo_root = repo_root.resolve()
    resolved_configs = [path.resolve() for path in config_paths]
    configs = {}
    for path in resolved_configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        configs[path.name] = {
            "sha256": _sha256(path),
            "model": payload.get("model"),
            "wonn_num_inner_steps": payload.get("wonn_num_inner_steps"),
            "data_path": payload.get("data_path"),
            "data_revision": payload.get("data_revision"),
            "eval_data_path": payload.get("eval_data_path"),
            "eval_data_revision": payload.get("eval_data_revision"),
            "encoder_model_name": payload.get("encoder_model_name"),
            "encoder_revision": payload.get("encoder_revision"),
            "tokenizer_revision": payload.get("tokenizer_revision"),
            "seed": payload.get("seed"),
            "batch_size": payload.get("batch_size"),
            "max_optimizer_steps": payload.get("max_optimizer_steps"),
        }
    required_pins = (
        "data_revision", "eval_data_revision", "encoder_revision",
        "tokenizer_revision",
    )
    for name, config in configs.items():
        missing = [field for field in required_pins if not config.get(field)]
        if missing:
            raise ValueError(f"{name} lacks immutable revisions: {missing}")

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if git_status:
        raise ValueError("formal experiment manifest requires a clean worktree")

    gpu = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    lock_path = repo_root / "requirements-lock.txt"
    sampling_path = repo_root / "src/configs/sampling_configs/cond_sampling_configs.yml"
    return {
        "status": "ready",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit,
        "git_worktree_clean": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "gpu": gpu,
        "packages": _package_versions((
            "datasets", "huggingface-hub", "numpy", "PyYAML",
            "sacrebleu", "transformers", "muon-optimizer",
        )),
        "requirements_lock_sha256": _sha256(lock_path),
        "sampling_config": {
            "path": str(sampling_path.relative_to(repo_root)),
            "sha256": _sha256(sampling_path),
        },
        "training_configs": configs,
        "metric_contract": {
            "bleu": "sacrebleu corpus_bleu lowercase effective_order",
            "chrf2": "sacrebleu corpus_chrf word_order=2",
            "ter": "sacrebleu corpus_ter normalized case-insensitive",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()
    payload = build_manifest(args.repo_root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": payload["status"],
        "source_commit": payload["source_commit"],
        "output": str(args.output.resolve()),
    }))


if __name__ == "__main__":
    main()

"""Resolve native-run and consolidated Phase 5 artifact paths."""

from pathlib import Path


def is_consolidated_elf_run(run_dir: Path) -> bool:
    return all(
        (run_dir / directory).is_dir()
        for directory in ("checkpoints", "training", "provenance")
    )


def checkpoint_path(run_dir: Path, step: int) -> Path:
    parent = run_dir / "checkpoints" if is_consolidated_elf_run(run_dir) else run_dir
    return parent / f"checkpoint_{step}"


def retained_checkpoint_steps(run_dir: Path, requested_steps) -> list[int]:
    steps = list(requested_steps)
    if is_consolidated_elf_run(run_dir):
        return [step for step in steps if step % 10000 == 0]
    return steps


def completion_path(run_dir: Path, step: int) -> Path:
    if is_consolidated_elf_run(run_dir):
        return run_dir / "provenance" / f"training_complete_{step}.json"
    return run_dir / "training_complete.json"


def config_path(run_dir: Path, step: int) -> Path:
    if not is_consolidated_elf_run(run_dir):
        return run_dir / "config.yml"
    segment = "00000_50000" if step <= 50000 else "50000_90000"
    return run_dir / "provenance" / f"config_{segment}.yml"


def source_commit_path(run_dir: Path, step: int) -> Path:
    if not is_consolidated_elf_run(run_dir):
        return run_dir / "source_commit"
    segment = "00000_50000" if step <= 50000 else "50000_90000"
    return run_dir / "provenance" / f"source_commit_{segment}.txt"


def training_metric_paths(run_dir: Path, terminal_step: int) -> list[Path]:
    if not is_consolidated_elf_run(run_dir):
        return [run_dir / "train_metrics.jsonl"]
    paths = [run_dir / "training" / "train_metrics_00000_50000.jsonl"]
    if terminal_step > 50000:
        paths.append(run_dir / "training" / "train_metrics_50000_90000.jsonl")
    return paths


def terminal_status_path(run_dir: Path, step: int) -> Path:
    if is_consolidated_elf_run(run_dir):
        return run_dir / "provenance" / f"terminal_status_{step}.json"
    return run_dir / "terminal_status.json"

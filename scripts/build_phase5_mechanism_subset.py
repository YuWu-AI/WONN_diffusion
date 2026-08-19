#!/usr/bin/env python
"""Build the deterministic short-sentence dataset for Phase 5 mechanism runs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def select_short_indices(
    dataset,
    *,
    train_size: int,
    heldout_size: int,
    max_source_tokens: int,
    max_target_tokens: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Select disjoint rows after a seeded permutation of the source dataset."""
    required = train_size + heldout_size
    if train_size <= 0 or heldout_size <= 0:
        raise ValueError("train_size and heldout_size must be positive")
    if max_source_tokens <= 0 or max_target_tokens <= 0:
        raise ValueError("token limits must be positive")
    selected = []
    permutation = np.random.default_rng(seed).permutation(len(dataset))
    for index in permutation:
        row = dataset[index]
        source_length = int(
            row.get("condition_sequence_length", len(row["condition_input_ids"]))
        )
        target_length = int(row.get("sequence_length", len(row["input_ids"])))
        if source_length <= max_source_tokens and target_length <= max_target_tokens:
            selected.append(int(index))
            if len(selected) == required:
                break
    if len(selected) != required:
        raise ValueError(
            f"found {len(selected)} eligible rows, expected {required}"
        )
    return selected[:train_size], selected[train_size:]


def _indices_sha256(indices: list[int]) -> str:
    payload = "\n".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _materialize(dataset, indices: list[int], tokenizer, output: Path) -> None:
    flatten_cache = output.parent / f".{output.name}.flatten.arrow"
    try:
        subset = dataset.select(indices).flatten_indices(
            cache_file_name=str(flatten_cache)
        )
        subset = subset.add_column("index", indices)
        source_text = tokenizer.batch_decode(
            subset["condition_input_ids"], skip_special_tokens=True
        )
        target_text = tokenizer.batch_decode(
            subset["input_ids"], skip_special_tokens=True
        )
        subset = subset.add_column("input", source_text)
        subset = subset.add_column("target", target_text)
        subset.reset_format()
        subset.save_to_disk(str(output))
    finally:
        flatten_cache.unlink(missing_ok=True)


def build_subset(args) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    from utils.data_utils import load_dataset_split

    dataset = load_dataset_split(args.source_dataset, revision=args.source_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    train_indices, heldout_indices = select_short_indices(
        dataset,
        train_size=args.train_size,
        heldout_size=args.heldout_size,
        max_source_tokens=args.max_source_tokens,
        max_target_tokens=args.max_target_tokens,
        seed=args.seed,
    )

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_root.with_name(args.output_root.name + ".in_progress")
    if temporary.exists():
        raise FileExistsError(f"stale temporary subset directory: {temporary}")
    temporary.mkdir()
    try:
        _materialize(dataset, train_indices, tokenizer, temporary / "train")
        _materialize(dataset, heldout_indices, tokenizer, temporary / "heldout")
        manifest = {
            "status": "complete",
            "source_dataset": args.source_dataset,
            "source_revision": args.source_revision,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
            "seed": args.seed,
            "train_size": len(train_indices),
            "heldout_size": len(heldout_indices),
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "train_indices_sha256": _indices_sha256(train_indices),
            "heldout_indices_sha256": _indices_sha256(heldout_indices),
            "train_indices": train_indices,
            "heldout_indices": heldout_indices,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset",
        default="embedded-language-flows/wmt14_de-en_train_t5",
    )
    parser.add_argument(
        "--source-revision",
        default="aee21115965ee2b9d3d6d02ec2a72f4f998476d9",
    )
    parser.add_argument("--tokenizer", default="t5-small")
    parser.add_argument(
        "--tokenizer-revision",
        default="df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "data/phase5_mechanism",
    )
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--heldout-size", type=int, default=2_000)
    parser.add_argument("--max-source-tokens", type=int, default=32)
    parser.add_argument("--max-target-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_subset(args)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output_root.resolve()),
        "train_size": manifest["train_size"],
        "heldout_size": manifest["heldout_size"],
    }))


if __name__ == "__main__":
    main()

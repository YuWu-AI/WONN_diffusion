#!/usr/bin/env python
"""Profile formal ELF/WONN training steps on the fixed WMT14 validation stream."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
# Phase 5 inputs are pinned and pre-cached before profiling. Avoid network
# probes contaminating setup time or making an otherwise local run flaky.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.config import apply_config_overrides, load_config_from_yaml
from modules.model_factory import build_model
from modules.t5_encoder import get_encoder
from train_step import train_step
from utils.data_utils import (
    get_dataloader,
    get_pad_token_id,
    load_dataset_split,
    prepare_batch,
)
from utils.train_utils import TrainState, get_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=("ELF-B", "ELF-WONN-B"))
    parser.add_argument("--decoder-prob", required=True, type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-override", action="append", default=[])
    return parser.parse_args()


def _next_batch(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("Phase 5 profiling requires a visible CUDA device.", file=sys.stderr)
        return 2
    if not 0.0 <= args.decoder_prob <= 1.0:
        raise ValueError("decoder-prob must be in [0, 1]")

    config = load_config_from_yaml(args.config)
    config = apply_config_overrides(config, args.config_override)
    config.model = args.model
    config.decoder_prob = args.decoder_prob
    config.global_batch_size = None
    config.batch_size = args.batch_size
    config.grad_accum_steps = 1
    config.use_wandb = False

    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    dataset = load_dataset_split(config.eval_data_path)
    dataloader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=False,
        seed=config.seed,
    )

    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    vocab_size = len(tokenizer)
    model = build_model(
        config,
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        vocab_size=vocab_size,
    ).to(device)
    lr = config.lr if config.lr is not None and config.lr > 0 else config.blr * args.batch_size / 256
    optimizer = get_optimizer(model, config, lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        ema_params1=TrainState.init_ema(model),
        dropout_generator=generator,
    )

    iterator = iter(dataloader)
    # Count eager operations before compiling. Keeping the compiled graph and
    # an eager backward alive together can exceed memory at otherwise-stable
    # batch sizes, while compilation does not change the mathematical work.
    batch, iterator = _next_batch(iterator, dataloader)
    batch = prepare_batch(batch, config, generator=generator)
    with FlopCounterMode(display=False) as flop_counter:
        state, _ = train_step(state, encoder=encoder, batch=batch, config=config)
    estimated_flops = flop_counter.get_total_flops()

    if config.compile_train:
        state = state.replace(model=torch.compile(state.model))

    for _ in range(args.warmup_steps):
        batch, iterator = _next_batch(iterator, dataloader)
        batch = prepare_batch(batch, config, generator=generator)
        state, _ = train_step(state, encoder=encoder, batch=batch, config=config)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = []
    for _ in range(args.measure_steps):
        batch, iterator = _next_batch(iterator, dataloader)
        batch = prepare_batch(batch, config, generator=generator)
        state, step_metrics = train_step(
            state, encoder=encoder, batch=batch, config=config
        )
        metrics.append(step_metrics)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    averages = {
        key: float(torch.stack([item[key] for item in metrics]).mean().cpu())
        for key in ("loss", "l2_loss", "ce_loss")
    }
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result = {
        "model": args.model,
        "decoder_prob": args.decoder_prob,
        "seed": config.seed,
        "batch_size": args.batch_size,
        "sequence_length": config.max_length,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.measure_steps,
        "steps_per_second": args.measure_steps / elapsed,
        "samples_per_second": args.measure_steps * args.batch_size / elapsed,
        "padded_positions_per_second": (
            args.measure_steps * args.batch_size * config.max_length / elapsed
        ),
        "peak_allocated_mib": peak_allocated / (1024 ** 2),
        "peak_reserved_mib": peak_reserved / (1024 ** 2),
        "estimated_flops_per_train_step": estimated_flops,
        "estimated_flops_scope": "PyTorch supported ops, full train_step including frozen encoder and optimizer",
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "use_bf16": bool(config.use_bf16),
        "compile_train": bool(config.compile_train),
        "wonn_num_layers": config.wonn_num_layers if args.model == "ELF-WONN-B" else None,
        "wonn_num_inner_steps": config.wonn_num_inner_steps if args.model == "ELF-WONN-B" else None,
        "wonn_num_oscillators": config.wonn_num_oscillators if args.model == "ELF-WONN-B" else None,
        "average_metrics": averages,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

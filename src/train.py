#!/usr/bin/env python
"""Training script for the ELF."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    save_checkpoint, load_checkpoint, load_warmstart_checkpoint,
    find_latest_checkpoint,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
    attach_lr_scheduler,
)
from generation import run_generation
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs, SamplingConfig
from modules.model_factory import build_model
from modules.denoiser_objectives import (
    denoiser_objectives_enabled,
    validate_denoiser_objective_config,
)
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from train_step import train_step

try:
    import wandb
except ImportError:
    wandb = None

# Logging: no timestamps; suppress noisy checkpoint loggers; unbuffered stdout
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)


def _init_distributed():
    """Initialize torch.distributed if launched via torchrun."""
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (PyTorch).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file to override defaults.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def _resolve_step_schedule(
    num_train_steps: int,
    grad_accum_steps: int,
    max_optimizer_steps,
    save_optimizer_steps,
):
    """Resolve an exact optimizer-step budget and requested save points."""
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    num_optimizer_steps = num_train_steps // grad_accum_steps
    if max_optimizer_steps is not None:
        if max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive when provided")
        num_optimizer_steps = min(num_optimizer_steps, max_optimizer_steps)

    requested_steps = set()
    if save_optimizer_steps:
        requested_steps = {
            int(value.strip())
            for value in save_optimizer_steps.split(",")
            if value.strip()
        }
        if any(step <= 0 or step > num_optimizer_steps for step in requested_steps):
            raise ValueError(
                "save_optimizer_steps must be positive and no larger than the training budget"
            )
    return num_optimizer_steps, num_optimizer_steps * grad_accum_steps, requested_steps


def _resume_position(resume_step: int, steps_per_epoch: int):
    """Map an exact train step to its epoch and in-epoch batch offset."""
    if resume_step < 0:
        raise ValueError("resume_step must be non-negative")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    return divmod(resume_step, steps_per_epoch)


def _requested_checkpoint_step(
    global_step: int, grad_accum_steps: int, requested_optimizer_steps,
):
    """Return the completed requested optimizer step, or None."""
    if global_step % grad_accum_steps != 0:
        return None
    optimizer_step = global_step // grad_accum_steps
    return optimizer_step if optimizer_step in requested_optimizer_steps else None


def _reconcile_metrics_file(metrics_path: str, resume_step: int):
    """Keep one latest valid metric per completed step up to a resume point."""
    summary = {"lines": 0, "kept": 0, "duplicates": 0, "discarded": 0}
    if not os.path.isfile(metrics_path):
        return summary

    records_by_step = {}
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            summary["lines"] += 1
            try:
                record = json.loads(line)
                step = record["step"]
                if isinstance(step, bool) or not isinstance(step, int):
                    raise ValueError("metric step must be an integer")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                summary["discarded"] += 1
                continue
            if step <= 0 or step > resume_step:
                summary["discarded"] += 1
                continue
            if step in records_by_step:
                summary["duplicates"] += 1
            records_by_step[step] = record

    temporary_path = f"{metrics_path}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as metrics_file:
            for step in sorted(records_by_step):
                metrics_file.write(json.dumps(records_by_step[step]) + "\n")
        os.replace(temporary_path, metrics_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    summary["kept"] = len(records_by_step)
    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_marker(path: str, payload: dict):
    """Atomically write a machine-readable run lifecycle marker."""
    temporary_path = f"{path}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as marker_file:
            json.dump(payload, marker_file, ensure_ascii=False, indent=2, sort_keys=True)
            marker_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _elapsed_training_offset(metrics_path: str) -> float:
    """Recover recorded training time so a resumed run keeps a monotonic axis."""
    if not os.path.isfile(metrics_path):
        return 0.0
    latest = 0.0
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            try:
                value = float(json.loads(line).get("elapsed_training_seconds", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            latest = max(latest, value)
    return latest


def _finalize_training(
    *, state, encoder, eval_dataset, tokenizer, config, generator,
    local_batch_size: int, global_step: int, training_complete_payload=None,
):
    """Persist the terminal state, mark training complete, and optionally evaluate."""
    state.step = global_step
    final_checkpoint = os.path.abspath(
        os.path.join(config.output_dir, f"checkpoint_{global_step}")
    )
    if os.path.isfile(final_checkpoint):
        log_for_0(f"Final checkpoint already exists at {final_checkpoint}")
    else:
        save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
        log_for_0(f"Final checkpoint saved to {config.output_dir}")

    if training_complete_payload is not None and _rank() == 0:
        payload = dict(training_complete_payload)
        payload["checkpoint"] = final_checkpoint
        _write_json_marker(
            os.path.join(config.output_dir, "training_complete.json"), payload,
        )
        log_for_0("Pure training complete; wrote training_complete.json")

    if not bool(getattr(config, "final_eval", True)):
        log_for_0("Final generation disabled; checkpoint evaluation is delegated to the run pipeline")
        return

    log_for_0("\n" + "=" * 60)
    log_for_0("Final Generation")
    log_for_0("=" * 60)
    run_generation(
        state=state, encoder=encoder, eval_dataset=eval_dataset,
        tokenizer=tokenizer, config=config, generator=generator,
        local_batch_size=local_batch_size,
    )


def run_training(config, *, force_cpu: bool = False):
    validate_denoiser_objective_config(config)
    process_started_perf = time.perf_counter()
    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device(f"cuda:{local_rank}")
    rank = _rank()
    world = _world_size()

    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (PyTorch)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max sequence length: {config.max_length}")
    log_for_0(f"Output dir: {config.output_dir}")
    log_for_0(f"HF Repo ID: {config.hf_repo_id}")
    log_for_0(f"Batch size per device: {config.batch_size}")
    log_for_0(f"Number of epochs: {config.epochs}")
    log_for_0(f"PyTorch device: {device}, world_size={world}")
    log_for_0(f"BF16 autocast: {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"torch.compile: {bool(getattr(config, 'compile_train', True)) and device.type == 'cuda'}")
    log_for_0(f"Gradient checkpointing: {bool(getattr(config, 'gradient_checkpointing', True))}")
    log_for_0("=" * 60)

    if config.use_wandb and rank == 0 and wandb is not None:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
        )
        resume_suffix = f" (resume={config.wandb_resume}, id={config.wandb_run_name})"
        log_for_0(f"Wandb initialized: {wandb.run.url}{resume_suffix}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # Per-rank seed so stochastic draws (decoder/denoiser branch coin,
    # timesteps, noise) diverge across ranks. A shared seed would make every
    # rank take the same branch in lockstep, producing spiky decoder gradients
    # instead of an evenly-mixed CE/L2 reduction.
    g = torch.Generator(device="cpu").manual_seed(config.seed + rank)

    # TF32 for fp32 matmuls on Ampere/Hopper (no hyperparameter change).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name,
        revision=getattr(config, "tokenizer_revision", None),
    )
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)

    log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(
        config.encoder_model_name, torch.float32,
        revision=getattr(config, "encoder_revision", None),
    )
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    log_for_0(f"Creating {config.model} model...")
    # Use the full tokenizer length for CE heads; tokenizer.vocab_size can exclude
    # added special tokens that still appear in tokenized Qwen targets.
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    log_for_0(f"Tokenizer vocab: CE head={vocab_size}")
    model = build_model(
        config,
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        vocab_size=vocab_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_for_0(f"Model parameters: {total_params:,}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_for_0(f"Total trainable parameters: {total_trainable:,}")

    # Keep initialization identical across ranks, then make runtime stochastic
    # ops (e.g. dropout) rank-specific.
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)

    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size: {config.global_batch_size}")
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // world
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * world
        local_batch_size = config.batch_size
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")

    steps_per_epoch = len(train_dataset) // total_batch_size
    num_train_steps = steps_per_epoch * config.epochs
    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    # Gradient accumulation: LR schedule is parameterized in optimizer steps
    grad_accum_steps = config.grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps
    num_optimizer_steps, target_train_steps, save_optimizer_steps = _resolve_step_schedule(
        num_train_steps=num_train_steps,
        grad_accum_steps=grad_accum_steps,
        max_optimizer_steps=config.max_optimizer_steps,
        save_optimizer_steps=config.save_optimizer_steps,
    )

    # Effective learning rate (scaled with effective batch size, including grad accum)
    if config.lr is None or config.lr <= 0:
        if config.lr is not None:
            log_for_0(f"Configured lr={config.lr} is non-positive; recomputing from blr={config.blr}")
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"World={world} | batch local={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total_train={target_train_steps}, "
        f"warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )
    if grad_accum_steps > 1:
        log_for_0(
            f"Grad accum={grad_accum_steps}, effective batch={total_batch_size * grad_accum_steps}, "
            f"optimizer steps={num_optimizer_steps}"
        )

    lr_fn = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(model, config, lr=config.lr, grad_accum_steps=grad_accum_steps)
    lr_scheduler = attach_lr_scheduler(optimizer, lr_fn)

    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        ema_params1=TrainState.init_ema(model),
        step=0, epoch=0, dropout_generator=g,
    )

    init_from = getattr(config, "init_from", None)
    if config.resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")

    # Auto-resume only when neither an explicit resume nor a warm-start was requested.
    if not config.resume and not init_from:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto_ckpt}")

    resume_step = 0
    warmstart_report = None
    if init_from:
        try:
            state, warmstart_report = load_warmstart_checkpoint(init_from, state)
            log_for_0(
                f"Initialized model/EMA from checkpoint step "
                f"{warmstart_report['source_step']}; optimizer and schedule start fresh"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to warm-start training from {init_from!r}") from e
    elif config.resume:
        try:
            ckpt_path = config.resume
            if "checkpoint_" not in ckpt_path:
                ckpt_path = find_latest_checkpoint(ckpt_path) or ckpt_path
            state, resume_step = load_checkpoint(ckpt_path, state)
            log_for_0(
                f"Resumed from step {resume_step} "
                f"(checkpoint epoch metadata {float(state.epoch):.2f})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to resume training from {config.resume!r}") from e

    start_epoch, steps_to_skip_in_epoch = _resume_position(resume_step, steps_per_epoch)
    resume_epoch_fractional = resume_step / steps_per_epoch
    state.epoch = resume_epoch_fractional

    # torch.compile before DDP so only the inner module is compiled and
    # checkpoint I/O (which uses unwrap_model -> _orig_mod) still works.
    if device.type == "cuda" and bool(getattr(config, "compile_train", True)):
        log_for_0("Compiling ELF model with torch.compile (first step will be slower)...")
        state = state.replace(model=torch.compile(state.model))

    if world > 1:
        # find_unused_parameters=False is safe: 0-mult sinks in train_step
        # (`0 * net_out.sum()` for CE, `0 * decoder_logits.sum()` for L2)
        # keep every head in the autograd graph on every step.
        state = state.replace(model=DDP(
            state.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        ))

    os.makedirs(config.output_dir, exist_ok=True)

    metrics_path = os.path.join(config.output_dir, "train_metrics.jsonl")
    if rank == 0:
        metrics_summary = _reconcile_metrics_file(metrics_path, resume_step)
        if metrics_summary["lines"]:
            log_for_0(
                "Reconciled train metrics at resume step "
                f"{resume_step}: kept={metrics_summary['kept']}, "
                f"duplicates={metrics_summary['duplicates']}, "
                f"discarded={metrics_summary['discarded']}"
            )
    if world > 1:
        dist.barrier()

    elapsed_training_offset = _elapsed_training_offset(metrics_path) if rank == 0 else 0.0

    if rank == 0:
        config_dict = {
            k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
            for k, v in vars(config).items()
        }
        config_path = os.path.join(config.output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        log_for_0(f"Config saved to {config_path}")

    train_dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=(world > 1),
        seed=config.seed,
    )

    log_for_0("\n" + "=" * 60)
    log_for_0("Checkpoint and Evaluation Schedule")
    log_for_0("=" * 60)
    log_for_0(
        f"Steps/epoch={steps_per_epoch}, epochs={config.epochs}, total={steps_per_epoch * config.epochs} | "
        f"save every {config.save_freq} epoch(s), eval every {config.eval_freq} epoch(s)"
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")

    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    global_step = resume_step
    state.step = global_step
    training_started_perf = time.perf_counter()
    training_started_at = _utc_now()

    if rank == 0:
        _write_json_marker(
            os.path.join(
                config.output_dir,
                "training_started.json" if resume_step == 0 else "training_resumed.json",
            ),
            {
                "status": "running",
                "started_at_utc": training_started_at,
                "resume_train_step": resume_step,
                "target_train_step": target_train_steps,
                "target_optimizer_step": num_optimizer_steps,
                "batch_size_per_device": local_batch_size,
                "world_size": world,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": total_batch_size * grad_accum_steps,
                "model": config.model,
                "model_parameters": total_params,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "gpu_total_memory_bytes": (
                    torch.cuda.get_device_properties(device).total_memory
                    if device.type == "cuda" else None
                ),
                "init_from": init_from,
                "warmstart_report": warmstart_report,
                "resume": config.resume,
            },
        )

    if global_step >= target_train_steps:
        log_for_0(
            f"Training budget already reached: step {global_step} >= {target_train_steps}."
        )

    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()
    # Track last save point for fractional save_freq; use fractional epoch from
    # checkpoint to avoid re-saving immediately after resume.
    last_save_epoch = resume_epoch_fractional if resume_step > 0 else float(start_epoch)

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Free device buffers from previous epoch before allocating new ones, to avoid
        # transient OOM at epoch boundaries.
        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = (resume_step - start_epoch * steps_per_epoch) if (epoch == start_epoch and resume_step > 0) else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=rank != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            if global_step >= target_train_steps:
                break
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and step_in_epoch < steps_to_skip_in_epoch:
                continue
            is_first_step = global_step == resume_step
            if is_first_step:
                log_for_0("Performing initial training step, this may take longer...")
                first_step_start = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            batch = prepare_batch(batch, config, generator=g)
            state, metrics = train_step(state, encoder=encoder, batch=batch, config=config)

            # Sync only on first step to measure compile/execution time and peak memory;
            # float() on the loss below already forces a device-to-host sync.
            if is_first_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - first_step_start
                log_for_0(f"First training step completed in {elapsed:.3f}s")
                if device.type == "cuda":
                    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    log_for_0(f"First training step peak allocated CUDA memory: {peak_mib:.1f} MiB")

            global_step += 1
            state.epoch = global_step / steps_per_epoch
            train_metrics.append(metrics)
            epoch_pbar.update(1)

            if global_step % config.log_freq == 0:
                loss_mean = torch.stack([m["loss"] for m in train_metrics]).mean()
                metric_totals = {
                    key: torch.stack([m[key] for m in train_metrics]).sum()
                    for key in (
                        "l2_loss_sum", "l2_token_count",
                        "ce_loss_sum", "ce_token_count",
                    )
                }
                stacked = torch.stack([loss_mean, *metric_totals.values()])
                # Average each metric across DDP ranks before logging — done
                # once per log_freq so we never sync on every train step.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                    stacked = stacked / dist.get_world_size()
                reduced = dict(zip(("loss", *metric_totals), stacked.tolist()))
                avg_loss = float(reduced["loss"])
                avg_l2 = float(reduced["l2_loss_sum"]) / max(
                    float(reduced["l2_token_count"]), 1.0
                )
                avg_ce = float(reduced["ce_loss_sum"]) / max(
                    float(reduced["ce_token_count"]), 1.0
                )
                aux_log = {}
                if denoiser_objectives_enabled(config):
                    aux_totals = {
                        key: torch.stack([
                            m.get(key, torch.zeros_like(m["loss"]))
                            for m in train_metrics
                        ]).sum()
                        for key in (
                            "token_loss_sum", "token_count",
                            "contrastive_loss_sum", "contrastive_count",
                            "aux_example_count",
                        )
                    }
                    aux_means = torch.stack([
                        torch.stack([
                            m.get("aux_loss", torch.zeros_like(m["loss"]))
                            for m in train_metrics
                        ]).mean(),
                        torch.stack([
                            m.get("aux_scale", torch.zeros_like(m["loss"]))
                            for m in train_metrics
                        ]).mean(),
                        *aux_totals.values(),
                    ])
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(aux_means, op=dist.ReduceOp.SUM)
                        aux_means[:2] = aux_means[:2] / dist.get_world_size()
                    aux_values = aux_means.tolist()
                    aux_log = {
                        "aux_loss": float(aux_values[0]),
                        "aux_scale": float(aux_values[1]),
                        "token_loss": float(aux_values[2]) / max(float(aux_values[3]), 1.0),
                        "source_contrastive_loss": (
                            float(aux_values[4]) / max(float(aux_values[5]), 1.0)
                        ),
                        "aux_examples": float(aux_values[6]),
                    }
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = state.optimizer.param_groups[0]["lr"]
                elapsed_training_seconds = (
                    elapsed_training_offset + time.perf_counter() - training_started_perf
                )

                postfix_dict = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2:.4f}", "ce": f"{avg_ce:.4f}",
                    "sps": f"{steps_per_sec:.1f}", "lr": f"{current_lr:.2e}",
                }
                if aux_log:
                    postfix_dict["aux"] = f"{aux_log['aux_loss']:.4f}"
                log_for_0(postfix_dict)
                epoch_pbar.set_postfix(**postfix_dict)

                if rank == 0:
                    aux_message = (
                        f", aux={aux_log['aux_loss']:.4f}, token={aux_log['token_loss']:.4f}, "
                        f"source={aux_log['source_contrastive_loss']:.4f}"
                        if aux_log else ""
                    )
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"l2={avg_l2:.4f}, ce={avg_ce:.4f}{aux_message}, "
                        f"lr={current_lr:.2e}, steps/sec={steps_per_sec:.2f}"
                    )
                    if config.use_wandb and wandb is not None:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb_payload = {
                                "train_loss": avg_loss, "train_l2_loss": avg_l2,
                                "train_ce_loss": avg_ce, "lr": current_lr,
                                "epoch": current_epoch_progress, "step": global_step,
                            }
                            wandb_payload.update({
                                f"train_{key}": value for key, value in aux_log.items()
                            })
                            wandb.log(wandb_payload, step=global_step)
                        except Exception:
                            pass
                    with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                        metric_payload = {
                            "step": global_step,
                            "optimizer_step": global_step // grad_accum_steps,
                            "loss": avg_loss,
                            "l2_loss": avg_l2,
                            "ce_loss": avg_ce,
                            "lr": current_lr,
                            "steps_per_second": steps_per_sec,
                            "samples_per_second": steps_per_sec * total_batch_size,
                            "samples_seen": global_step * total_batch_size,
                            "elapsed_training_seconds": elapsed_training_seconds,
                            "elapsed_run_seconds": time.perf_counter() - process_started_perf,
                            "timestamp_utc": _utc_now(),
                        }
                        metric_payload.update(aux_log)
                        metrics_file.write(json.dumps(metric_payload) + "\n")

                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            optimizer_step = _requested_checkpoint_step(
                global_step, grad_accum_steps, save_optimizer_steps,
            )
            if optimizer_step is not None:
                save_checkpoint(
                    state, config.output_dir, global_step,
                    hf_repo_id=config.hf_repo_id,
                )
                log_for_0(
                    f"Saved requested optimizer-step checkpoint {optimizer_step} "
                    f"(train step {global_step})"
                )

            # Intra-epoch checkpoint saving (fractional save_freq, e.g., 0.1 epoch)
            if 0 < config.save_freq < 1:
                progress = epoch + (global_step - epoch * steps_per_epoch) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
                    log_for_0(f"Saved checkpoint at epoch {progress:.2f} (step {global_step})")
                    last_save_epoch = progress

        if global_step >= target_train_steps:
            epoch_pbar.close()
            break

        epoch_pbar.close()
        current_epoch = epoch + 1
        state.epoch = current_epoch

        if config.save_freq >= 1 and current_epoch % config.save_freq == 0:
            save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
            log_for_0(f"Saved checkpoint at epoch {current_epoch} (step {global_step})")

        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config, generator=g,
                local_batch_size=local_batch_size,
            )
            last_log_step = global_step
            last_log_time = time.time()

    if global_step != target_train_steps:
        raise RuntimeError(
            f"Training ended at step {global_step}, expected {target_train_steps}"
        )

    training_completed_at = _utc_now()
    elapsed_training_seconds = (
        elapsed_training_offset + time.perf_counter() - training_started_perf
    )
    peak_allocated_cuda_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    _finalize_training(
        state=state, encoder=encoder, eval_dataset=eval_dataset,
        tokenizer=tokenizer, config=config, generator=g,
        local_batch_size=local_batch_size, global_step=global_step,
        training_complete_payload={
            "status": "complete",
            "started_at_utc": training_started_at,
            "completed_at_utc": training_completed_at,
            "completed_train_step": global_step,
            "completed_optimizer_step": global_step // grad_accum_steps,
            "samples_seen": global_step * total_batch_size,
            "effective_batch_size": total_batch_size * grad_accum_steps,
            "model": config.model,
            "model_parameters": total_params,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_cuda_mib": peak_allocated_cuda_mib,
            "elapsed_training_seconds": elapsed_training_seconds,
            "elapsed_run_seconds": time.perf_counter() - process_started_perf,
            "init_from": init_from,
            "warmstart_report": warmstart_report,
            "final_evaluation_in_training_process": bool(getattr(config, "final_eval", True)),
        },
    )
    if config.use_wandb and rank == 0 and wandb is not None:
        wandb.finish()


def main():
    """CLI entry point: parse args, load config, then run training."""
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config, force_cpu=args.use_cpu)


if __name__ == "__main__":
    main()

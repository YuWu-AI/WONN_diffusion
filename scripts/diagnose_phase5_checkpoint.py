#!/usr/bin/env python
"""Diagnose Phase 5 checkpoint sampling, conditioning, and flow behavior."""

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase5_50k import _metric_bundle
from configs.config import SamplingConfig, load_config_from_yaml
from generation import _build_eval_model
from modules.model_factory import build_model
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_checkpoint
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    _dlm_decode_batch,
    _generate_samples_single_batch,
    mask_after_eos,
    shift_left,
)
from utils.sampling_utils import _forward_sample, add_noise, restore_cond
from utils.train_utils import TrainState, get_optimizer


MODEL_SPECS = {
    "ELF-B": {
        "config": "src/configs/training_configs/train_de-en_ELF-B-phase5-50k.yml",
        "checkpoint": "elf_b_seed42_b12/checkpoint_50000",
    },
    "WONN-L6T3": {
        "config": "src/configs/training_configs/train_de-en-WONN-L6T3-phase5-50k.yml",
        "checkpoint": "wonn_l6t3_seed42_b12/checkpoint_50000",
    },
}
DEFAULT_T_VALUES = (0.05, 0.15, 0.30, 0.50, 0.75, 0.90)
DEFAULT_ROLLOUT_STARTS = (0.0, 0.25, 0.50, 0.75)


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _all_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    return fieldnames


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not 0.0 <= item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("values must be comma-separated numbers in [0, 1)")
    return values


def _checkpoint_step(checkpoint: Path) -> int:
    name = checkpoint.name
    if not name.startswith("checkpoint_"):
        raise ValueError(f"checkpoint path must end in checkpoint_<step>: {checkpoint}")
    try:
        step = int(name.rsplit("_", 1)[1])
    except ValueError as exc:
        raise ValueError(
            f"checkpoint path must end in checkpoint_<step>: {checkpoint}"
        ) from exc
    if step <= 0:
        raise ValueError(f"checkpoint step must be positive: {checkpoint}")
    return step


def _fixed_time_steps(
    num_steps: int,
    p_mean: float,
    p_std: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_steps == 1:
        return torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn((num_steps - 1,), generator=generator, dtype=torch.float32)
    interior = torch.sigmoid(logits * p_std + p_mean).sort().values
    steps = torch.cat((torch.zeros(1), interior, torch.ones(1)))
    return steps.to(device=device, dtype=dtype)


def _rollout_steps(base_steps: torch.Tensor, start_t: float) -> torch.Tensor:
    if not 0.0 <= start_t < 1.0:
        raise ValueError("start_t must be in [0, 1)")
    if start_t == 0.0:
        return base_steps
    suffix = base_steps[base_steps > start_t]
    start = torch.tensor([start_t], dtype=base_steps.dtype, device=base_steps.device)
    if suffix.numel() == 0 or float(suffix[-1]) < 1.0:
        suffix = torch.cat((suffix, torch.ones_like(start)))
    return torch.cat((start, suffix))


def _nearest_length_permutation(lengths: torch.Tensor) -> torch.Tensor:
    """Return a deterministic wrong-source permutation with small length gaps."""
    count = int(lengths.numel())
    if count < 2:
        raise ValueError("at least two rows are required to shuffle conditioning")
    order = torch.argsort(lengths.to(torch.int64)).tolist()
    permutation = list(range(count))
    even_end = count if count % 2 == 0 else count - 3
    for offset in range(0, even_end, 2):
        left, right = order[offset], order[offset + 1]
        permutation[left], permutation[right] = right, left
    if count % 2:
        first, second, third = order[-3:]
        permutation[first] = second
        permutation[second] = third
        permutation[third] = first
    return torch.tensor(permutation, dtype=torch.long, device=lengths.device)


def _shuffled_condition(
    cond_seq: torch.Tensor,
    cond_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    lengths = cond_mask.to(torch.int64).sum(dim=1)
    permutation = _nearest_length_permutation(lengths)
    shuffled = torch.zeros_like(cond_seq)
    copied = 0
    requested = int(lengths.sum().item())
    length_deltas = []
    for destination, source in enumerate(permutation.tolist()):
        destination_len = int(lengths[destination].item())
        source_len = int(lengths[source].item())
        copy_len = min(destination_len, source_len)
        if copy_len:
            shuffled[destination, :copy_len] = cond_seq[source, :copy_len]
        copied += copy_len
        length_deltas.append(abs(destination_len - source_len))
    metadata = {
        "mean_abs_source_length_delta": float(np.mean(length_deltas)),
        "copied_slot_fraction": copied / max(requested, 1),
        "fixed_target_offsets": True,
    }
    return shuffled, metadata


def _to_device(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            target_dtype = dtype if value.is_floating_point() else value.dtype
            result[key] = value.to(device=device, dtype=target_dtype)
        else:
            result[key] = value
    return result


def _prepare_batches(
    dataset,
    tokenizer,
    encoder,
    config,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    dataloader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=False,
    )
    prepared = []
    seen = 0
    for batch in dataloader:
        if seen >= num_samples:
            break
        keep = min(int(batch["input_ids"].shape[0]), num_samples - seen)
        input_ids = torch.as_tensor(np.asarray(batch["input_ids"][:keep]), device=device).long()
        encoder_mask = torch.as_tensor(
            np.asarray(batch["encoder_attention_mask"][:keep]), device=device
        ).float()
        cond_mask = torch.as_tensor(
            np.asarray(batch["cond_seq_mask"][:keep]), device=device
        ).float()
        with torch.inference_mode():
            x0 = encode_text(
                input_ids=input_ids,
                attention_mask=encoder_mask,
                encoder=encoder,
                latent_mean=config.latent_mean,
                latent_std=config.latent_std,
            )
        indices = batch.get("index", list(range(seen, seen + keep)))[:keep]
        prepared.append({
            "input_ids": input_ids.cpu(),
            "x0": x0.float().cpu(),
            "cond_mask": cond_mask.cpu(),
            "indices": [int(item) for item in indices],
            "sources": list(batch["input"][:keep]),
            "references": list(batch["target"][:keep]),
        })
        seen += keep
    if seen != num_samples:
        raise ValueError(f"prepared {seen} examples, expected {num_samples}")
    return prepared


def _load_eval_model(
    config,
    checkpoint: Path,
    text_encoder_dim: int,
    vocab_size: int,
    device,
    expected_step: int,
):
    model = build_model(
        config,
        text_encoder_dim=text_encoder_dim,
        max_length=config.max_length,
        vocab_size=vocab_size,
    ).to(device)
    optimizer = get_optimizer(model, config, lr=1e-4)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema_params1=TrainState.init_ema(model),
        step=0,
        epoch=0.0,
        dropout_generator=torch.Generator(device="cpu").manual_seed(config.seed),
    )
    state, loaded_step = load_checkpoint(str(checkpoint), state)
    if loaded_step != expected_step:
        raise ValueError(f"expected checkpoint step {expected_step}, got {loaded_step}")
    eval_model = _build_eval_model(state, use_compile=False).to(device).eval()
    del state, optimizer, model
    gc.collect()
    return eval_model


def _decode_ids(
    latent: torch.Tensor,
    model,
    config,
    cond_mask: torch.Tensor,
    tokenizer,
) -> tuple[torch.Tensor, list[str]]:
    predicted = _dlm_decode_batch(
        z=latent,
        model=model,
        t_final_val=1.0,
        config=config,
        self_cond_cfg_scale=1.0,
    )
    cond_lengths = cond_mask.to(torch.int32).sum(dim=1)
    generation_length = config.max_length - config.max_input_length
    predicted = shift_left(predicted, cond_lengths, 0)[:, :generation_length]
    predicted = mask_after_eos(
        predicted,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
    )
    texts = [
        tokenizer.decode(row.detach().cpu().numpy(), skip_special_tokens=True)
        for row in predicted
    ]
    return predicted, texts


def _target_token_accuracy(
    full_predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    cond_mask: torch.Tensor,
) -> tuple[int, int]:
    target_mask = cond_mask == 0
    correct = ((full_predicted_ids == target_ids) & target_mask).sum().item()
    return int(correct), int(target_mask.sum().item())


def _sampling_config() -> SamplingConfig:
    return SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[64],
        cfgs=[2],
        self_cond_cfg_scales=[1],
        time_schedule="logit_normal",
    )


def _initial_noise(shape, scale: float, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return noise.to(device=device, dtype=dtype)


def _sample_rows(indices, sources, references, generated) -> list[dict]:
    return [
        {
            "id": int(index),
            "source": source,
            "reference": reference,
            "generated": hypothesis,
        }
        for index, source, reference, hypothesis in zip(
            indices, sources, references, generated
        )
    ]


@torch.inference_mode()
def _conditioning_diagnostic(
    model,
    prepared_batches: list[dict],
    config,
    tokenizer,
    device,
    seed: int,
    output_dir: Path,
) -> tuple[dict, dict]:
    dtype = next(model.parameters()).dtype
    sampling_config = _sampling_config()
    generated = {name: [] for name in ("correct", "shuffled", "zero")}
    artifact_rows = {name: [] for name in generated}
    latent_stats = {name: {"squared_delta": 0.0, "count": 0} for name in ("shuffled", "zero")}
    token_disagreement = {name: {"different": 0, "count": 0} for name in ("shuffled", "zero")}
    shuffle_metadata = []
    correct_cache = []
    total_generation_seconds = {name: 0.0 for name in generated}

    for batch_index, cpu_batch in enumerate(prepared_batches):
        batch = _to_device(cpu_batch, device, dtype)
        x0 = batch["x0"]
        cond_mask = batch["cond_mask"]
        shuffled, metadata = _shuffled_condition(x0, cond_mask)
        shuffle_metadata.append(metadata)
        variants = {
            "correct": x0,
            "shuffled": shuffled,
            "zero": torch.zeros_like(x0),
        }
        t_steps = _fixed_time_steps(
            64, config.denoiser_p_mean, config.denoiser_p_std,
            seed + 10_000 + batch_index, device, dtype,
        )
        initial = _initial_noise(
            x0.shape, config.denoiser_noise_scale,
            seed + 20_000 + batch_index, device, dtype,
        )
        batch_outputs = {}
        batch_ids = {}
        for variant, cond_seq in variants.items():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            latent = _generate_samples_single_batch(
                model=model,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                z=initial.clone(),
                t_steps=t_steps,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
                config=config,
                sampling_config=sampling_config,
                cfg_scale=2.0,
                self_cond_cfg_scale=1.0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_generation_seconds[variant] += time.perf_counter() - started
            predicted_ids, texts = _decode_ids(latent, model, config, cond_mask, tokenizer)
            generated[variant].extend(texts)
            artifact_rows[variant].extend(_sample_rows(
                cpu_batch["indices"], cpu_batch["sources"],
                cpu_batch["references"], texts,
            ))
            batch_outputs[variant] = latent
            batch_ids[variant] = predicted_ids
        target_mask = (cond_mask == 0).unsqueeze(-1)
        for variant in ("shuffled", "zero"):
            delta = (batch_outputs[variant] - batch_outputs["correct"])[target_mask.expand_as(x0)]
            latent_stats[variant]["squared_delta"] += float((delta.float() ** 2).sum().item())
            latent_stats[variant]["count"] += int(delta.numel())
            different = batch_ids[variant] != batch_ids["correct"]
            token_disagreement[variant]["different"] += int(different.sum().item())
            token_disagreement[variant]["count"] += int(different.numel())
        correct_cache.append({
            "indices": cpu_batch["indices"],
            "sources": cpu_batch["sources"],
            "references": cpu_batch["references"],
            "generated": list(generated["correct"][-len(cpu_batch["indices"]):]),
        })

    references = [reference for batch in prepared_batches for reference in batch["references"]]
    metrics = {}
    for variant in generated:
        bundle = _metric_bundle(generated[variant], references)
        bundle["generation_seconds"] = total_generation_seconds[variant]
        bundle["samples_per_second"] = len(references) / max(total_generation_seconds[variant], 1e-12)
        if variant != "correct":
            stat = latent_stats[variant]
            bundle["final_latent_rms_delta_vs_correct"] = math.sqrt(
                stat["squared_delta"] / max(stat["count"], 1)
            )
            disagreement = token_disagreement[variant]
            bundle["decoded_token_disagreement_pct_vs_correct"] = (
                100.0 * disagreement["different"] / max(disagreement["count"], 1)
            )
            bundle["output_chrf_vs_correct"] = _metric_bundle(
                generated[variant], generated["correct"]
            )["chrf2"]
        metrics[variant] = bundle
        _write_jsonl(output_dir / "conditioning" / f"{variant}.jsonl", artifact_rows[variant])

    metrics["shuffle_construction"] = {
        "mean_abs_source_length_delta": float(np.mean([
            item["mean_abs_source_length_delta"] for item in shuffle_metadata
        ])),
        "mean_copied_slot_fraction": float(np.mean([
            item["copied_slot_fraction"] for item in shuffle_metadata
        ])),
        "fixed_target_offsets": True,
    }
    return metrics, {"correct_batches": correct_cache}


@torch.inference_mode()
def _decoder_probe(model, prepared_batches, config, tokenizer, device, seed: int) -> dict:
    dtype = next(model.parameters()).dtype
    probe_names = ("clean_x0", "training_noised_x0")
    generated = {name: [] for name in probe_names}
    ce_sums = {name: 0.0 for name in probe_names}
    token_counts = {name: 0 for name in probe_names}
    correct_counts = {name: 0 for name in probe_names}
    references = []

    for batch_index, cpu_batch in enumerate(prepared_batches):
        batch = _to_device(cpu_batch, device, dtype)
        x0 = batch["x0"]
        cond_mask = batch["cond_mask"]
        target_ids = batch["input_ids"].long()
        references.extend(cpu_batch["references"])
        generator = torch.Generator(device="cpu").manual_seed(seed + 30_000 + batch_index)
        lambda_logits = (
            torch.randn(x0.shape[:2], generator=generator, dtype=torch.float32)
            * config.decoder_p_std + config.decoder_p_mean
        )
        decoder_lambda = torch.sigmoid(lambda_logits).unsqueeze(-1).to(device=device, dtype=dtype)
        decoder_noise = (
            torch.randn(x0.shape, generator=generator, dtype=torch.float32)
            * config.decoder_noise_scale
        ).to(device=device, dtype=dtype)
        inputs = {
            "clean_x0": x0,
            "training_noised_x0": decoder_lambda * x0 + (1.0 - decoder_lambda) * decoder_noise,
        }
        target_mask = cond_mask == 0
        sc_batch = torch.ones((x0.shape[0],), device=device, dtype=dtype)
        t_final = torch.ones((x0.shape[0],), device=device, dtype=dtype)
        for name, latent in inputs.items():
            model_input = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
            use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                _, logits = model(
                    model_input,
                    t_final,
                    deterministic=True,
                    self_cond_cfg_scale=sc_batch,
                    decoder_step_active=True,
                )
            per_token = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape_as(target_ids)
            ce_sums[name] += float((per_token * target_mask).sum().item())
            token_counts[name] += int(target_mask.sum().item())
            full_predicted = logits.argmax(dim=-1)
            correct, count = _target_token_accuracy(full_predicted, target_ids, cond_mask)
            correct_counts[name] += correct
            if count != int(target_mask.sum().item()):
                raise AssertionError("decoder token count mismatch")
            cond_lengths = cond_mask.to(torch.int32).sum(dim=1)
            generation_length = config.max_length - config.max_input_length
            shifted = shift_left(full_predicted, cond_lengths, 0)[:, :generation_length]
            shifted = mask_after_eos(
                shifted,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
            )
            generated[name].extend([
                tokenizer.decode(row.detach().cpu().numpy(), skip_special_tokens=True)
                for row in shifted
            ])

    results = {}
    for name in probe_names:
        results[name] = {
            "ce_loss": ce_sums[name] / max(token_counts[name], 1),
            "target_token_accuracy_pct": 100.0 * correct_counts[name] / max(token_counts[name], 1),
            **_metric_bundle(generated[name], references),
        }
    results["ce_logging_zero_batch_probability"] = (
        (1.0 - config.decoder_prob) ** config.batch_size
    )
    return results


def _masked_vector_stats(predicted, target, mask) -> dict:
    expanded = mask.unsqueeze(-1).expand_as(predicted)
    pred = predicted[expanded].float()
    truth = target[expanded].float()
    error = pred - truth
    dot = float((pred * truth).sum().item())
    pred_sq = float((pred * pred).sum().item())
    truth_sq = float((truth * truth).sum().item())
    return {
        "error_squared_sum": float((error * error).sum().item()),
        "pred_truth_dot": dot,
        "pred_squared_sum": pred_sq,
        "truth_squared_sum": truth_sq,
        "count": int(pred.numel()),
    }


@torch.inference_mode()
def _time_bin_diagnostic(
    model,
    prepared_batches,
    config,
    tokenizer,
    device,
    seed: int,
    t_values: tuple[float, ...],
) -> list[dict]:
    dtype = next(model.parameters()).dtype
    references = [reference for batch in prepared_batches for reference in batch["references"]]
    accumulators = {
        value: {
            "velocity": {key: 0.0 for key in (
                "error_squared_sum", "pred_truth_dot", "pred_squared_sum",
                "truth_squared_sum", "count",
            )},
            "x_error_squared_sum": 0.0,
            "x_count": 0,
            "conditioning_delta_squared_sum": 0.0,
            "conditioning_count": 0,
            "velocity_pred_squared_sum": 0.0,
            "generated": [],
        }
        for value in t_values
    }

    for batch_index, cpu_batch in enumerate(prepared_batches):
        batch = _to_device(cpu_batch, device, dtype)
        x0 = batch["x0"]
        cond_mask = batch["cond_mask"]
        target_mask = cond_mask == 0
        base_noise = _initial_noise(
            x0.shape, config.denoiser_noise_scale,
            seed + 40_000 + batch_index, device, dtype,
        ) / config.denoiser_noise_scale
        for t_value in t_values:
            t_batch = torch.full((x0.shape[0],), t_value, device=device, dtype=dtype)
            z = add_noise(x0, base_noise, t_batch, config, cond_seq_mask=cond_mask.unsqueeze(-1))
            x_prev = restore_cond(torch.zeros_like(z), x0, cond_mask)
            use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                v_pred, x_pred = _forward_sample(
                    model=model,
                    z=z,
                    t_batch=t_batch,
                    x_pred_prev=x_prev,
                    config=config,
                    cfg_scale=2.0,
                    self_cond_cfg_scale=1.0,
                    cond_seq=x0,
                    cond_seq_mask=cond_mask,
                )
                zero_cond = torch.zeros_like(x0)
                z_zero = restore_cond(z, zero_cond, cond_mask)
                v_zero, _ = _forward_sample(
                    model=model,
                    z=z_zero,
                    t_batch=t_batch,
                    x_pred_prev=restore_cond(torch.zeros_like(z), zero_cond, cond_mask),
                    config=config,
                    cfg_scale=2.0,
                    self_cond_cfg_scale=1.0,
                    cond_seq=zero_cond,
                    cond_seq_mask=cond_mask,
                )
            ideal_v = (x0 - z) / max(1.0 - t_value, config.t_eps)
            stats = _masked_vector_stats(v_pred, ideal_v, target_mask)
            accumulator = accumulators[t_value]
            for key, value in stats.items():
                accumulator["velocity"][key] += value
            expanded = target_mask.unsqueeze(-1).expand_as(x0)
            x_error = (x_pred - x0)[expanded].float()
            accumulator["x_error_squared_sum"] += float((x_error ** 2).sum().item())
            accumulator["x_count"] += int(x_error.numel())
            conditioning_delta = (v_pred - v_zero)[expanded].float()
            accumulator["conditioning_delta_squared_sum"] += float(
                (conditioning_delta ** 2).sum().item()
            )
            accumulator["conditioning_count"] += int(conditioning_delta.numel())
            velocity_pred = v_pred[expanded].float()
            accumulator["velocity_pred_squared_sum"] += float((velocity_pred ** 2).sum().item())
            _, texts = _decode_ids(x_pred, model, config, cond_mask, tokenizer)
            accumulator["generated"].extend(texts)

    rows = []
    for t_value in t_values:
        accumulator = accumulators[t_value]
        velocity = accumulator["velocity"]
        cosine = velocity["pred_truth_dot"] / max(
            math.sqrt(velocity["pred_squared_sum"] * velocity["truth_squared_sum"]),
            1e-12,
        )
        condition_rms = math.sqrt(
            accumulator["conditioning_delta_squared_sum"]
            / max(accumulator["conditioning_count"], 1)
        )
        velocity_rms = math.sqrt(
            accumulator["velocity_pred_squared_sum"]
            / max(accumulator["conditioning_count"], 1)
        )
        decoded_metrics = _metric_bundle(accumulator["generated"], references)
        rows.append({
            "t": t_value,
            "velocity_mse": velocity["error_squared_sum"] / max(velocity["count"], 1),
            "velocity_cosine": cosine,
            "x_pred_mse": accumulator["x_error_squared_sum"] / max(accumulator["x_count"], 1),
            "conditioning_velocity_rms_delta": condition_rms,
            "conditioning_delta_to_velocity_ratio": condition_rms / max(velocity_rms, 1e-12),
            "decoded_bleu": decoded_metrics["bleu"],
            "decoded_chrf2": decoded_metrics["chrf2"],
            "decoded_ter": decoded_metrics["ter"],
            "decoded_empty_rate_pct": decoded_metrics["empty_rate_pct"],
        })
    return rows


@torch.inference_mode()
def _rollout_diagnostic(
    model,
    prepared_batches,
    config,
    tokenizer,
    device,
    seed: int,
    starts: tuple[float, ...],
    correct_metrics: dict,
) -> list[dict]:
    dtype = next(model.parameters()).dtype
    sampling_config = _sampling_config()
    references = [reference for batch in prepared_batches for reference in batch["references"]]
    generated = {value: [] for value in starts if value > 0.0}
    step_counts = {value: [] for value in starts if value > 0.0}
    rows = []
    if 0.0 in starts:
        rows.append({
            "start_t": 0.0,
            "mean_remaining_ode_steps": 64.0,
            **{key: correct_metrics[key] for key in (
                "bleu", "chrf2", "ter", "empty_rate_pct", "length_ratio",
            )},
        })
    for batch_index, cpu_batch in enumerate(prepared_batches):
        batch = _to_device(cpu_batch, device, dtype)
        x0 = batch["x0"]
        cond_mask = batch["cond_mask"]
        unit_noise = _initial_noise(
            x0.shape, 1.0, seed + 20_000 + batch_index, device, dtype,
        )
        base_steps = _fixed_time_steps(
            64, config.denoiser_p_mean, config.denoiser_p_std,
            seed + 10_000 + batch_index, device, dtype,
        )
        for start_t in starts:
            if start_t == 0.0:
                continue
            t_batch = torch.full((x0.shape[0],), start_t, device=device, dtype=dtype)
            latent_start = add_noise(
                x0, unit_noise, t_batch, config, cond_seq_mask=cond_mask.unsqueeze(-1)
            )
            t_steps = _rollout_steps(base_steps, start_t)
            latent = _generate_samples_single_batch(
                model=model,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                z=latent_start,
                t_steps=t_steps,
                cond_seq=x0,
                cond_seq_mask=cond_mask,
                config=config,
                sampling_config=sampling_config,
                cfg_scale=2.0,
                self_cond_cfg_scale=1.0,
            )
            _, texts = _decode_ids(latent, model, config, cond_mask, tokenizer)
            generated[start_t].extend(texts)
            step_counts[start_t].append(int(t_steps.numel() - 1))
    for start_t in starts:
        if start_t == 0.0:
            continue
        rows.append({
            "start_t": start_t,
            "mean_remaining_ode_steps": float(np.mean(step_counts[start_t])),
            **_metric_bundle(generated[start_t], references),
        })
    return sorted(rows, key=lambda row: row["start_t"])


def diagnose_model(
    label: str,
    model,
    prepared_batches,
    config,
    tokenizer,
    device,
    seed: int,
    t_values: tuple[float, ...],
    rollout_starts: tuple[float, ...],
    output_dir: Path,
) -> dict:
    model_output = output_dir / label.lower().replace("-", "_")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    conditioning, _ = _conditioning_diagnostic(
        model, prepared_batches, config, tokenizer, device, seed, model_output,
    )
    decoder_probe = _decoder_probe(
        model, prepared_batches, config, tokenizer, device, seed,
    )
    time_bins = _time_bin_diagnostic(
        model, prepared_batches, config, tokenizer, device, seed, t_values,
    )
    rollout = _rollout_diagnostic(
        model, prepared_batches, config, tokenizer, device, seed,
        rollout_starts, conditioning["correct"],
    )
    elapsed = time.perf_counter() - started
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    result = {
        "model": label,
        "num_samples": sum(len(batch["indices"]) for batch in prepared_batches),
        "seed": seed,
        "sampling": {
            "method": "ode",
            "steps": 64,
            "cfg": 2.0,
            "self_cond_cfg": 1.0,
            "time_schedule": "logit_normal",
        },
        "conditioning": conditioning,
        "decoder_probe": decoder_probe,
        "time_bins": time_bins,
        "oracle_start_rollout": rollout,
        "elapsed_seconds": elapsed,
        "peak_allocated_cuda_mib": peak_mib,
    }
    _atomic_write_json(model_output / "diagnostics.json", result)
    _write_csv(
        model_output / "time_bins.csv",
        time_bins,
        _all_fieldnames(time_bins),
    )
    _write_csv(
        model_output / "oracle_start_rollout.csv",
        rollout,
        _all_fieldnames(rollout),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/phase5/redesign_v2/formal50k"),
        help="Phase 5 formal run root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", default="ELF-B,WONN-L6T3",
        help="Comma-separated model labels.",
    )
    parser.add_argument(
        "--model-spec", action="append", default=[],
        metavar="LABEL|CONFIG|CHECKPOINT",
        help="Evaluate a custom model; repeatable and mutually exclusive with --models.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Override the first model config's evaluation dataset.",
    )
    parser.add_argument("--dataset-revision", type=str, default=None)
    parser.add_argument(
        "--t-values", type=_parse_floats,
        default=DEFAULT_T_VALUES,
    )
    parser.add_argument(
        "--rollout-starts", type=_parse_floats,
        default=DEFAULT_ROLLOUT_STARTS,
    )
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 2:
        raise ValueError("num-samples must be at least 2 for shuffled conditioning")
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 for shuffled conditioning")
    if args.model_spec:
        if args.models != "ELF-B,WONN-L6T3":
            raise ValueError("--model-spec cannot be combined with a custom --models value")
        specs = {}
        for raw_spec in args.model_spec:
            parts = raw_spec.split("|")
            if len(parts) != 3 or not all(part.strip() for part in parts):
                raise ValueError(
                    "model specs must use LABEL|CONFIG|CHECKPOINT"
                )
            label, config_path, checkpoint_path = (part.strip() for part in parts)
            if label in specs:
                raise ValueError(f"duplicate model label: {label}")
            specs[label] = {"config": config_path, "checkpoint": checkpoint_path}
        selected_models = tuple(specs)
    else:
        selected_models = tuple(
            item.strip() for item in args.models.split(",") if item.strip()
        )
        unknown = sorted(set(selected_models) - set(MODEL_SPECS))
        if unknown:
            raise ValueError(f"unknown models: {unknown}")
        specs = {label: MODEL_SPECS[label] for label in selected_models}
    output_path = args.output_dir / "diagnostics.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite")

    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    first_config_path = Path(specs[selected_models[0]]["config"])
    if not first_config_path.is_absolute():
        first_config_path = REPO_ROOT / first_config_path
    first_config = load_config_from_yaml(str(first_config_path))
    tokenizer = AutoTokenizer.from_pretrained(
        first_config.tokenizer_name or first_config.encoder_model_name,
        revision=first_config.tokenizer_revision,
    )
    encoder_config, encoder = get_encoder(
        first_config.encoder_model_name,
        torch.float32,
        revision=first_config.encoder_revision,
    )
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    dataset_path = args.dataset_path or first_config.eval_data_path
    dataset_revision = (
        args.dataset_revision
        if args.dataset_revision is not None
        else first_config.eval_data_revision
    )
    dataset = load_dataset_split(dataset_path, revision=dataset_revision)
    prepared_batches = _prepare_batches(
        dataset, tokenizer, encoder, first_config,
        args.num_samples, args.batch_size, device,
    )
    encoder = encoder.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "status": "running",
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "models": {},
        "t_values": list(args.t_values),
        "rollout_starts": list(args.rollout_starts),
    }
    _atomic_write_json(output_path, payload)
    try:
        for label in selected_models:
            spec = specs[label]
            config_path = Path(spec["config"])
            if not config_path.is_absolute():
                config_path = REPO_ROOT / config_path
            checkpoint_path = Path(spec["checkpoint"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = args.root / checkpoint_path
            config = load_config_from_yaml(str(config_path))
            expected_step = _checkpoint_step(checkpoint_path)
            model = _load_eval_model(
                config,
                checkpoint_path,
                encoder_config.d_model,
                len(tokenizer),
                device,
                expected_step,
            )
            payload["models"][label] = diagnose_model(
                label, model, prepared_batches, config, tokenizer, device,
                args.seed, args.t_values, args.rollout_starts, args.output_dir,
            )
            _atomic_write_json(output_path, payload)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        payload["status"] = "complete"
        _atomic_write_json(output_path, payload)
    except Exception:
        payload["status"] = "failed"
        _atomic_write_json(output_path, payload)
        raise

    print(json.dumps({
        "status": payload["status"],
        "output": str(output_path.resolve()),
        "models": list(payload["models"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

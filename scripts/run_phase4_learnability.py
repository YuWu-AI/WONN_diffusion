#!/usr/bin/env python
"""Deterministic Phase 4 learnability experiments for the formal WONN-ELF.

The fixed-batch objectives deliberately reset all stochastic draws before each
step. Loss changes therefore measure optimization of one exact task rather than
variation from newly sampled flow times, noise, or decoder branch assignments.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from configs.config import load_config_from_yaml
from modules.model_factory import build_model
from modules.t5_encoder import get_encoder
from train_step import train_step
from utils.data_utils import (
    get_dataloader,
    get_pad_token_id,
    load_dataset_split,
    prepare_batch,
)
from utils.train_utils import TrainState


@dataclass
class ObjectiveResult:
    parameters: int
    initial_loss: float
    final_loss: float
    loss_ratio: float
    initial_l2: float
    final_l2: float
    initial_ce: float
    final_ce: float
    omega_alpha_delta: float
    theta_embedding_delta: float
    omega_input_projection_delta: float
    omega_output_projection_delta: float


@dataclass
class ConditionalResult:
    initial_loss: float
    final_loss: float
    accuracy: float
    swapped_accuracy: float
    prediction_change_fraction: float
    correct_margin: float
    swapped_margin: float


class FrozenBatchEncoder(nn.Module):
    """Return cached real T5 outputs for one exact batch."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("expected_input_ids", input_ids.detach().clone())
        self.register_buffer(
            "expected_attention_mask", attention_mask.detach().clone()
        )
        self.register_buffer("latents", latents.detach().clone())

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        del deterministic
        if input_ids.shape != self.expected_input_ids.shape or not torch.equal(
            input_ids, self.expected_input_ids
        ):
            raise ValueError("FrozenBatchEncoder received a different batch")
        if attention_mask is None or not torch.equal(
            attention_mask, self.expected_attention_mask
        ):
            raise ValueError("FrozenBatchEncoder received a different attention mask")
        return self.latents


def _float_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach()) for key, value in metrics.items()}


def _build_real_batch(config, device: torch.device, batch_size: int):
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
    dataset = load_dataset_split(config.eval_data_path).select(range(batch_size))
    loader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length,
        distributed=False,
    )
    batch = prepare_batch(
        next(iter(loader)), config, generator=torch.Generator().manual_seed(0)
    )
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        latents = encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["encoder_attention_mask"].float(),
            deterministic=True,
        ).float()
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokenizer, batch, latents, encoder_config.d_model


def _build_model_state(config, device, text_encoder_dim, vocab_size, seed):
    torch.manual_seed(seed)
    model = build_model(
        config,
        text_encoder_dim=text_encoder_dim,
        max_length=config.max_length,
        vocab_size=vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.phase4_lr, weight_decay=0.0
    )
    state = TrainState(
        model=model,
        optimizer=optimizer,
        ema_params1=TrainState.init_ema(model),
        dropout_generator=torch.Generator(device="cpu"),
    )
    return state


def _parameter_snapshot(model) -> Dict[str, torch.Tensor]:
    empty = torch.zeros(1)
    transitions = getattr(model, "omega_transitions", ())
    if not transitions:
        return {
            "alpha": empty,
            "theta_embedding": empty,
            "input_projection": empty,
            "output_projection": empty,
        }
    transition = transitions[0]
    return {
        "alpha": transition.raw_alpha.detach().float().cpu().clone(),
        "theta_embedding": (
            transition.theta_embedding.weight.detach().float().cpu().clone()
        ),
        "input_projection": (
            transition.input_projection.weight.detach().float().cpu().clone()
        ),
        "output_projection": (
            transition.output_projection.weight.detach().float().cpu().clone()
        ),
    }


def run_fixed_objective(
    *,
    config,
    batch,
    latents,
    text_encoder_dim,
    vocab_size,
    decoder_prob,
    branch_seed,
    steps,
    model_seed,
    device,
):
    config.decoder_prob = decoder_prob
    config.self_cond_prob = 0.0
    config.label_drop_prob = 0.0
    state = _build_model_state(config, device, text_encoder_dim, vocab_size, model_seed)
    frozen_encoder = FrozenBatchEncoder(
        batch["input_ids"], batch["encoder_attention_mask"].float(), latents
    ).to(device)
    parameters_before = _parameter_snapshot(state.model)

    first = last = None
    for _ in range(steps):
        torch.manual_seed(10_000 + branch_seed)
        state.dropout_generator.manual_seed(branch_seed)
        state, metrics = train_step(
            state, encoder=frozen_encoder, batch=batch, config=config
        )
        values = _float_metrics(metrics)
        first = values if first is None else first
        last = values

    parameters_after = _parameter_snapshot(state.model)
    result = ObjectiveResult(
        parameters=sum(parameter.numel() for parameter in state.model.parameters()),
        initial_loss=first["loss"],
        final_loss=last["loss"],
        loss_ratio=last["loss"] / max(first["loss"], 1e-12),
        initial_l2=first["l2_loss"],
        final_l2=last["l2_loss"],
        initial_ce=first["ce_loss"],
        final_ce=last["ce_loss"],
        omega_alpha_delta=float(
            (parameters_after["alpha"] - parameters_before["alpha"]).abs()
        ),
        theta_embedding_delta=float(
            (
                parameters_after["theta_embedding"]
                - parameters_before["theta_embedding"]
            ).norm()
        ),
        omega_input_projection_delta=float(
            (
                parameters_after["input_projection"]
                - parameters_before["input_projection"]
            ).norm()
        ),
        omega_output_projection_delta=float(
            (
                parameters_after["output_projection"]
                - parameters_before["output_projection"]
            ).norm()
        ),
    )
    return state, result


def run_conditional_task(
    *,
    state,
    normalized_latents,
    config,
    steps,
    device,
):
    model = state.model
    model.train()
    source_length = 8
    target_slice = slice(source_length, source_length + 8)
    inputs = torch.zeros(
        (2, config.max_length, normalized_latents.shape[-1]),
        device=device,
        dtype=normalized_latents.dtype,
    )
    inputs[:, :source_length] = normalized_latents[:2, :source_length]
    attention_mask = torch.ones((2, config.max_length), device=device)
    labels = torch.tensor([10, 20], device=device).view(2, 1).expand(-1, 8)
    t = torch.ones(2, device=device)
    mode = torch.ones(2, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.phase4_lr)

    def evaluate(source_inputs):
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            _, logits = model(
                source_inputs,
                t,
                attention_mask=attention_mask,
                deterministic=True,
                decoder_step_active=mode,
            )
        selected = logits[:, target_slice]
        loss = F.cross_entropy(selected.reshape(-1, selected.shape[-1]), labels.reshape(-1))
        predictions = selected.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
        correct = selected.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        alternative_labels = labels.flip(0)
        alternative = selected.gather(
            -1, alternative_labels.unsqueeze(-1)
        ).squeeze(-1)
        return loss, predictions, accuracy, (correct - alternative).mean()

    initial_loss, _, _, _ = evaluate(inputs)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            _, logits = model(
                inputs,
                t,
                attention_mask=attention_mask,
                deterministic=False,
                decoder_step_active=mode,
            )
            selected = logits[:, target_slice]
            loss = F.cross_entropy(
                selected.reshape(-1, selected.shape[-1]), labels.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_loss, predictions, accuracy, margin = evaluate(inputs)
    swapped = inputs.clone()
    swapped[:, :source_length] = swapped.flip(0)[:, :source_length]
    _, swapped_predictions, swapped_accuracy, swapped_margin = evaluate(swapped)
    return ConditionalResult(
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        accuracy=float(accuracy),
        swapped_accuracy=float(swapped_accuracy),
        prediction_change_fraction=float(
            (predictions != swapped_predictions).float().mean()
        ),
        correct_margin=float(margin),
        swapped_margin=float(swapped_margin),
    )


def _verify(results):
    failures = []
    comparison = results["wmt14_subset_comparison"]["models"]
    checks = {
        "denoising loss ratio <= 0.25": results["denoising"]["loss_ratio"] <= 0.25,
        "decoding loss ratio <= 0.50": results["decoding"]["loss_ratio"] <= 0.50,
        "mixed loss ratio <= 0.60": results["mixed"]["loss_ratio"] <= 0.60,
        "mixed L2 decreased": results["mixed"]["final_l2"] < results["mixed"]["initial_l2"],
        "mixed CE decreased": results["mixed"]["final_ce"] < results["mixed"]["initial_ce"],
        "omega alpha changed": results["mixed"]["omega_alpha_delta"] > 0.0,
        "theta embedding changed": results["mixed"]["theta_embedding_delta"] > 0.0,
        "omega input projection changed": (
            results["mixed"]["omega_input_projection_delta"] > 0.0
        ),
        "omega output projection changed": (
            results["mixed"]["omega_output_projection_delta"] > 0.0
        ),
        "conditional accuracy == 1": results["conditional"]["accuracy"] == 1.0,
        "source swap breaks labels": results["conditional"]["swapped_accuracy"] <= 0.25,
        "source swap changes predictions": results["conditional"]["prediction_change_fraction"] >= 0.75,
        "source swap reverses margin": results["conditional"]["correct_margin"] > 0.0
        and results["conditional"]["swapped_margin"] < 0.0,
        "WONN subset curve decreased": comparison["ELF-WONN-B"]["loss_ratio"] < 0.60,
        "ELF subset curve decreased": comparison["ELF-B"]["loss_ratio"] < 0.60,
        "WONN subset L2 and CE decreased": comparison["ELF-WONN-B"]["final_l2"]
        < comparison["ELF-WONN-B"]["initial_l2"]
        and comparison["ELF-WONN-B"]["final_ce"]
        < comparison["ELF-WONN-B"]["initial_ce"],
        "ELF subset L2 and CE decreased": comparison["ELF-B"]["final_l2"]
        < comparison["ELF-B"]["initial_l2"]
        and comparison["ELF-B"]["final_ce"]
        < comparison["ELF-B"]["initial_ce"],
    }
    for description, passed in checks.items():
        if not passed:
            failures.append(description)
    results["checks"] = checks
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--conditional-steps", type=int, default=120)
    parser.add_argument("--comparison-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.conditional_steps <= 0 or args.comparison_steps <= 0:
        parser.error("all step counts must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    if not torch.cuda.is_available():
        print("Phase 4 formal learnability verification requires CUDA.", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(
        str(REPO_ROOT / "src/configs/training_configs/train_de-en_ELF-WONN-B.yml")
    )
    config.use_bf16 = True
    config.gradient_checkpointing = True
    config.ema_decay1 = 0.9999
    config.phase4_lr = args.lr
    tokenizer, batch, latents, text_encoder_dim = _build_real_batch(
        config, device, batch_size=5
    )
    torch.cuda.reset_peak_memory_stats(device)
    experiment_start = time.perf_counter()

    experiments = {
        "denoising": (0.0, 0, 1),
        "decoding": (1.0, 0, 1),
        "mixed": (0.2, 1, 5),
    }
    results = {}
    mixed_state = None
    for index, (name, (probability, branch_seed, batch_size)) in enumerate(
        experiments.items()
    ):
        subset = {
            key: value[:batch_size] if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        state, result = run_fixed_objective(
            config=config,
            batch=subset,
            latents=latents[:batch_size],
            text_encoder_dim=text_encoder_dim,
            vocab_size=len(tokenizer),
            decoder_prob=probability,
            branch_seed=branch_seed,
            steps=args.steps,
            model_seed=100 + index,
            device=device,
        )
        results[name] = asdict(result)
        if name == "mixed":
            mixed_state = state
        else:
            del state
            torch.cuda.empty_cache()

    normalized_latents = latents / config.latent_std
    conditional = run_conditional_task(
        state=mixed_state,
        normalized_latents=normalized_latents,
        config=config,
        steps=args.conditional_steps,
        device=device,
    )
    results["conditional"] = asdict(conditional)
    del mixed_state
    torch.cuda.empty_cache()

    comparison = {}
    comparison_batch = {
        key: value[:5] if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    for model_name in ("ELF-WONN-B", "ELF-B"):
        config.model = model_name
        comparison_state, comparison_result = run_fixed_objective(
            config=config,
            batch=comparison_batch,
            latents=latents[:5],
            text_encoder_dim=text_encoder_dim,
            vocab_size=len(tokenizer),
            decoder_prob=0.2,
            branch_seed=1,
            steps=args.comparison_steps,
            model_seed=200,
            device=device,
        )
        comparison[model_name] = asdict(comparison_result)
        del comparison_state
        torch.cuda.empty_cache()
    results["wmt14_subset_comparison"] = {
        "steps": args.comparison_steps,
        "batch_size": 5,
        "decoder_prob": 0.2,
        "models": comparison,
    }
    torch.cuda.synchronize(device)
    results["runtime"] = {
        "seconds": time.perf_counter() - experiment_start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "device": torch.cuda.get_device_name(device),
    }
    failures = _verify(results)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if failures:
        print("Phase 4 checks failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

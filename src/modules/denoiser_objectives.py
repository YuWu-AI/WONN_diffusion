"""Optional semantic objectives for denoiser-predicted clean latents."""

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.func import functional_call

from utils.train_utils import unwrap_model


def validate_denoiser_objective_config(config) -> None:
    """Reject invalid auxiliary-objective settings before training starts."""
    nonnegative = (
        "denoiser_token_loss_weight",
        "denoiser_source_contrastive_weight",
        "denoiser_aux_start_step",
        "denoiser_aux_warmup_steps",
        "denoiser_source_contrastive_margin",
    )
    for name in nonnegative:
        if float(getattr(config, name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    max_t = float(getattr(config, "denoiser_aux_max_t", 0.25))
    if not 0.0 <= max_t <= 1.0:
        raise ValueError("denoiser_aux_max_t must be in [0, 1]")
    contrastive_prob = float(
        getattr(config, "denoiser_source_contrastive_prob", 1.0)
    )
    if not 0.0 <= contrastive_prob <= 1.0:
        raise ValueError("denoiser_source_contrastive_prob must be in [0, 1]")


def denoiser_objectives_enabled(config) -> bool:
    return (
        float(getattr(config, "denoiser_token_loss_weight", 0.0)) > 0.0
        or float(getattr(config, "denoiser_source_contrastive_weight", 0.0)) > 0.0
    )


def _auxiliary_scale(step: int, start_step: int, warmup_steps: int) -> float:
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step - start_step + 1) / warmup_steps)


def _nearest_length_permutation(lengths: torch.Tensor) -> torch.Tensor:
    """Create a deterministic derangement while keeping source lengths close."""
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


def _replace_condition_with_permuted_source(
    latent: torch.Tensor,
    cond_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep target slots fixed and replace each clean source with another row."""
    lengths = cond_mask.to(torch.int64).sum(dim=1)
    permutation = _nearest_length_permutation(lengths)
    shuffled = latent.clone()
    for destination, source in enumerate(permutation.tolist()):
        destination_len = int(lengths[destination].item())
        source_len = int(lengths[source].item())
        shuffled[destination, :destination_len] = 0
        copy_len = min(destination_len, source_len)
        if copy_len:
            shuffled[destination, :copy_len] = latent[source, :copy_len]
    return shuffled


def _ema_decoder_logits(
    *,
    model,
    ema_params: Dict[str, torch.Tensor],
    latent: torch.Tensor,
    self_cond_cfg_scale: torch.Tensor,
) -> torch.Tensor:
    """Decode with detached EMA weights while retaining gradients to latent."""
    inner = unwrap_model(model)
    parameters = {
        name: ema_params[name].detach()
        for name, _ in inner.named_parameters()
    }
    buffers = {
        name: value.detach()
        for name, value in inner.named_buffers()
    }
    model_input = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    t_final = torch.ones(
        (latent.shape[0],), device=latent.device, dtype=latent.dtype
    )
    decoder_active = torch.ones_like(t_final)
    _, logits = functional_call(
        inner,
        (parameters, buffers),
        (model_input, t_final),
        {
            "deterministic": True,
            "self_cond_cfg_scale": self_cond_cfg_scale,
            "decoder_step_active": decoder_active,
        },
        strict=True,
    )
    if logits is None:
        raise RuntimeError("EMA decoder did not return token logits")
    return logits.float()


def _per_example_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    mask = target_mask.to(per_token.dtype)
    sums = (per_token * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    means = sums / counts.clamp(min=1.0)
    return means, sums, counts


def compute_denoiser_auxiliary_loss(
    *,
    model,
    ema_params: Dict[str, torch.Tensor],
    x_pred: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    cond_mask: torch.Tensor,
    timesteps: torch.Tensor,
    denoiser_rows: torch.Tensor,
    label_drop_mask: torch.Tensor,
    self_cond_cfg_scale: torch.Tensor,
    config,
    step: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the weighted auxiliary loss and auditable numerator metrics."""
    validate_denoiser_objective_config(config)
    zero = x_pred.sum() * 0.0
    scale = _auxiliary_scale(
        step,
        int(getattr(config, "denoiser_aux_start_step", 0)),
        int(getattr(config, "denoiser_aux_warmup_steps", 0)),
    )
    if scale == 0.0 or not denoiser_objectives_enabled(config):
        return zero, {}

    eligible = (
        denoiser_rows.bool()
        & ~label_drop_mask.bool()
        & (timesteps <= float(getattr(config, "denoiser_aux_max_t", 0.25)))
        & (target_mask.sum(dim=1) > 0)
    )
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel() == 0:
        metrics = {
            "aux_loss": zero.detach(),
            "aux_scale": torch.tensor(scale, device=x_pred.device),
            "token_loss_sum": zero.detach(),
            "token_count": torch.zeros((), device=x_pred.device),
            "contrastive_loss_sum": zero.detach(),
            "contrastive_count": torch.zeros((), device=x_pred.device),
            "aux_example_count": torch.zeros((), device=x_pred.device),
        }
        return zero, metrics

    selected_x = x_pred.index_select(0, eligible_indices)
    selected_targets = targets.index_select(0, eligible_indices)
    selected_target_mask = target_mask.index_select(0, eligible_indices)
    selected_cond_mask = cond_mask.index_select(0, eligible_indices)
    selected_cfg = self_cond_cfg_scale.index_select(0, eligible_indices)
    correct_logits = _ema_decoder_logits(
        model=model,
        ema_params=ema_params,
        latent=selected_x,
        self_cond_cfg_scale=selected_cfg,
    )
    correct_means, token_sums, token_counts = _per_example_cross_entropy(
        correct_logits, selected_targets, selected_target_mask
    )
    token_sum = token_sums.sum()
    token_count = token_counts.sum()
    token_loss = token_sum / token_count.clamp(min=1.0)

    contrastive_sum = zero
    contrastive_count = torch.zeros((), device=x_pred.device)
    contrastive_weight = float(
        getattr(config, "denoiser_source_contrastive_weight", 0.0)
    )
    if contrastive_weight > 0.0 and eligible_indices.numel() >= 2:
        probability = float(
            getattr(config, "denoiser_source_contrastive_prob", 1.0)
        )
        selected_by_probability = torch.rand(
            (eligible_indices.numel(),), generator=generator, device="cpu"
        ) < probability
        contrast_indices = torch.nonzero(
            selected_by_probability, as_tuple=False
        ).flatten().to(x_pred.device)
        if contrast_indices.numel() >= 2:
            contrast_x = selected_x.index_select(0, contrast_indices)
            contrast_cond_mask = selected_cond_mask.index_select(0, contrast_indices)
            wrong_x = _replace_condition_with_permuted_source(
                contrast_x, contrast_cond_mask
            )
            wrong_logits = _ema_decoder_logits(
                model=model,
                ema_params=ema_params,
                latent=wrong_x,
                self_cond_cfg_scale=selected_cfg.index_select(0, contrast_indices),
            )
            wrong_means, _, _ = _per_example_cross_entropy(
                wrong_logits,
                selected_targets.index_select(0, contrast_indices),
                selected_target_mask.index_select(0, contrast_indices),
            )
            margin = float(
                getattr(config, "denoiser_source_contrastive_margin", 0.1)
            )
            contrastive_values = F.relu(
                margin
                + correct_means.index_select(0, contrast_indices)
                - wrong_means
            )
            contrastive_sum = contrastive_values.sum()
            contrastive_count = torch.tensor(
                float(contrastive_values.numel()), device=x_pred.device
            )

    token_weight = float(getattr(config, "denoiser_token_loss_weight", 0.0))
    contrastive_loss = contrastive_sum / contrastive_count.clamp(min=1.0)
    auxiliary = scale * (
        token_weight * token_loss + contrastive_weight * contrastive_loss
    )
    if not math.isfinite(scale):
        raise ValueError("auxiliary loss scale must be finite")
    metrics = {
        "aux_loss": auxiliary.detach(),
        "aux_scale": torch.tensor(scale, device=x_pred.device),
        "token_loss_sum": token_sum.detach(),
        "token_count": token_count.detach(),
        "contrastive_loss_sum": contrastive_sum.detach(),
        "contrastive_count": contrastive_count.detach(),
        "aux_example_count": torch.tensor(
            float(eligible_indices.numel()), device=x_pred.device
        ),
    }
    return auxiliary, metrics

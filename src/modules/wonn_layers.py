"""Winfree oscillator primitives for the ELF-WONN backbone."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import (
    DEFAULT_BIAS_INIT,
    DEFAULT_KERNEL_INIT,
    RMSNorm,
    _make_linear,
)


def wrap_phase(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without detaching gradients."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def phase_features(theta: torch.Tensor) -> torch.Tensor:
    """Return a periodic Euclidean representation [..., 2 * K]."""
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)


class GroupedPhaseMap(nn.Module):
    """Independent phase-safe sensitivity/influence map for each head."""

    def __init__(self, num_heads: int, oscillators_per_head: int):
        super().__init__()
        self.num_heads = num_heads
        self.oscillators_per_head = oscillators_per_head
        in_features = 2 * oscillators_per_head
        self.weight = nn.Parameter(
            torch.empty(num_heads, oscillators_per_head, in_features)
        )
        self.bias = nn.Parameter(torch.empty(num_heads, oscillators_per_head))
        for head_weight in self.weight:
            DEFAULT_KERNEL_INIT(head_weight)
        DEFAULT_BIAS_INIT(self.bias)

    def forward(self, grouped_phase_features: torch.Tensor) -> torch.Tensor:
        output = torch.einsum(
            "bnhd,hod->bnho", grouped_phase_features, self.weight
        )
        return torch.tanh(output + self.bias)


class AttentiveWinfreeCoupling(nn.Module):
    """Attention-weighted Winfree influence without a Transformer V residual."""

    def __init__(
        self,
        num_oscillators: int,
        num_heads: int,
        qk_head_dim: int,
        coupling_mode: str = "learned",
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if num_oscillators % num_heads != 0:
            raise ValueError("num_oscillators must be divisible by num_heads")
        if qk_head_dim % 2 != 0:
            raise ValueError("qk_head_dim must be even for RoPE")
        if coupling_mode not in {"learned", "fixed_trig"}:
            raise ValueError("coupling_mode must be 'learned' or 'fixed_trig'")

        self.num_oscillators = num_oscillators
        self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim
        self.oscillators_per_head = num_oscillators // num_heads
        self.coupling_mode = coupling_mode
        self.attn_drop = attn_drop

        phase_width = 2 * num_oscillators
        self.phase_norm = RMSNorm(phase_width)
        self.q_proj = _make_linear(
            phase_width, num_heads * qk_head_dim, bias=True
        )
        self.k_proj = _make_linear(
            phase_width, num_heads * qk_head_dim, bias=True
        )
        self.q_norm = RMSNorm(qk_head_dim)
        self.k_norm = RMSNorm(qk_head_dim)
        if coupling_mode == "learned":
            self.sensitivity = GroupedPhaseMap(
                num_heads, self.oscillators_per_head
            )
            self.influence = GroupedPhaseMap(
                num_heads, self.oscillators_per_head
            )

    def _grouped_phase_features(self, theta: torch.Tensor) -> torch.Tensor:
        batch, length, _ = theta.shape
        grouped = theta.reshape(
            batch, length, self.num_heads, self.oscillators_per_head
        )
        return torch.cat([torch.sin(grouped), torch.cos(grouped)], dim=-1)

    @staticmethod
    def _attention_mask(
        attention_mask: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        if attention_mask.dim() == 2:
            mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 4:
            mask = attention_mask
        else:
            raise ValueError("attention_mask must have rank 2, 3, or 4")
        return mask.to(device=scores.device, dtype=torch.bool)

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        rope_fn: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, length, _ = theta.shape
        features = self.phase_norm(phase_features(theta))
        q = self.q_proj(features).reshape(
            batch, length, self.num_heads, self.qk_head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(features).reshape(
            batch, length, self.num_heads, self.qk_head_dim
        ).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.qk_head_dim
        )
        if attention_mask is not None:
            mask = self._attention_mask(attention_mask, scores)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)

        grouped_features = self._grouped_phase_features(theta)
        if self.coupling_mode == "learned":
            sensitivity = self.sensitivity(grouped_features)
            influence = self.influence(grouped_features)
        else:
            grouped_theta = theta.reshape(
                batch, length, self.num_heads, self.oscillators_per_head
            )
            sensitivity = torch.cos(grouped_theta)
            influence = torch.sin(grouped_theta)

        dropped_weights = weights
        if self.attn_drop > 0.0:
            dropped_weights = F.dropout(
                weights, p=self.attn_drop, training=not deterministic
            )
        message = torch.einsum("bhij,bjhc->bihc", dropped_weights, influence)
        grouped_omega = omega.reshape(
            batch, length, self.num_heads, self.oscillators_per_head
        )
        phase_velocity = grouped_omega + sensitivity * message

        diagnostics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            diagnostics = {
                "attention_entropy": -(
                    weights * safe_weights.log()
                ).sum(dim=-1).mean(),
                "coupling_message_rms": message.float().square().mean().sqrt(),
            }
        return phase_velocity.reshape(batch, length, -1), diagnostics


class WONNLayer(nn.Module):
    """One slow-frequency layer containing recurrent fast-phase steps."""

    def __init__(
        self,
        num_oscillators: int,
        num_heads: int,
        qk_head_dim: int,
        num_inner_steps: int,
        step_init: float = 0.1,
        step_max: float = 0.25,
        coupling_mode: str = "learned",
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if num_inner_steps <= 0:
            raise ValueError("num_inner_steps must be positive")
        if not 0.0 < step_init < step_max:
            raise ValueError("step_init must lie strictly between 0 and step_max")
        self.num_oscillators = num_oscillators
        self.num_inner_steps = num_inner_steps
        self.step_max = step_max
        raw_step = math.log(step_init / (step_max - step_init))
        self.raw_step = nn.Parameter(torch.tensor(raw_step, dtype=torch.float32))
        self.coupling = AttentiveWinfreeCoupling(
            num_oscillators=num_oscillators,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            coupling_mode=coupling_mode,
            attn_drop=attn_drop,
        )
        transition_width = 3 * num_oscillators
        self.frequency_norm = RMSNorm(transition_width)
        self.frequency_transition = _make_linear(
            transition_width, num_oscillators, bias=True
        )
        self.frequency_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def step_size(self) -> torch.Tensor:
        return self.step_max * torch.sigmoid(self.raw_step)

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        rope_fn: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        update_squares = []
        attention_entropies = []
        message_rms_values = []
        for _ in range(self.num_inner_steps):
            velocity, coupling_diagnostics = self.coupling(
                theta,
                omega,
                rope_fn=rope_fn,
                attention_mask=attention_mask,
                deterministic=deterministic,
                collect_diagnostics=collect_diagnostics,
            )
            phase_update = self.step_size.to(velocity.dtype) * velocity
            theta = wrap_phase(theta + phase_update)
            if collect_diagnostics:
                update_squares.append(phase_update.float().square().mean())
                attention_entropies.append(
                    coupling_diagnostics["attention_entropy"]
                )
                message_rms_values.append(
                    coupling_diagnostics["coupling_message_rms"]
                )

        transition_input = torch.cat([phase_features(theta), omega], dim=-1)
        frequency_delta = torch.tanh(
            self.frequency_transition(self.frequency_norm(transition_input))
        )
        frequency_update = torch.tanh(self.frequency_gate).to(omega.dtype) * frequency_delta
        omega = omega + frequency_update

        diagnostics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            diagnostics = {
                "phase_update_rms": torch.stack(update_squares).mean().sqrt(),
                "frequency_norm": omega.float().square().mean().sqrt(),
                "frequency_update_rms": frequency_update.float().square().mean().sqrt(),
                "attention_entropy": torch.stack(attention_entropies).mean(),
                "coupling_message_rms": torch.stack(message_rms_values).mean(),
            }
        return theta, omega, diagnostics

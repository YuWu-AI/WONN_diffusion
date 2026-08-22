"""Winfree oscillator primitives for the ELF-WONN backbone."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import RMSNorm, _make_linear


def wrap_phase(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without detaching gradients."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def phase_features(theta: torch.Tensor) -> torch.Tensor:
    """Return a periodic Euclidean representation [..., 2 * K]."""
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)


def _reset_grouped_linear(weight: torch.Tensor, bias: torch.Tensor) -> None:
    """Match ``groups=K`` pointwise-convolution default initialization."""
    for group_weight, group_bias in zip(weight, bias):
        nn.init.kaiming_uniform_(group_weight, a=math.sqrt(5))
        fan_in = group_weight.shape[-1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(group_bias, -bound, bound)


class PerOscillatorMLP(nn.Module):
    """Independent periodic MLPs, one for each oscillator channel."""

    def __init__(self, num_oscillators: int, hidden_features: int):
        super().__init__()
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        self.num_oscillators = num_oscillators
        self.hidden_features = hidden_features
        self.input_weight = nn.Parameter(
            torch.empty(num_oscillators, hidden_features, 2)
        )
        self.input_bias = nn.Parameter(
            torch.empty(num_oscillators, hidden_features)
        )
        self.output_weight = nn.Parameter(
            torch.empty(num_oscillators, 1, hidden_features)
        )
        self.output_bias = nn.Parameter(torch.empty(num_oscillators, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_grouped_linear(self.input_weight, self.input_bias)
        _reset_grouped_linear(self.output_weight, self.output_bias)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.shape[-1] != self.num_oscillators:
            raise ValueError(
                f"expected {self.num_oscillators} oscillators, got {theta.shape[-1]}"
            )
        periodic = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        hidden = torch.einsum(
            "bnki,khi->bnkh", periodic, self.input_weight
        ) + self.input_bias
        hidden = F.relu(hidden)
        output = torch.einsum(
            "bnkh,koh->bnko", hidden, self.output_weight
        ) + self.output_bias
        return output.squeeze(-1)


class ThetaEmbedding(nn.Module):
    """Independent periodic affine embedding for every oscillator."""

    def __init__(self, num_oscillators: int):
        super().__init__()
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        self.num_oscillators = num_oscillators
        self.weight = nn.Parameter(torch.empty(num_oscillators, 1, 2))
        self.bias = nn.Parameter(torch.empty(num_oscillators, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_grouped_linear(self.weight, self.bias)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.shape[-1] != self.num_oscillators:
            raise ValueError(
                f"expected {self.num_oscillators} oscillators, got {theta.shape[-1]}"
            )
        periodic = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        embedded = torch.einsum("bnki,koi->bnko", periodic, self.weight)
        return (embedded + self.bias).squeeze(-1)


class AttentiveWinfreeCoupling(nn.Module):
    """Complete QKV/O attention field followed by Winfree phase velocity."""

    def __init__(
        self,
        num_oscillators: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_oscillators % num_heads != 0:
            raise ValueError("num_oscillators must be divisible by num_heads")
        head_dim = num_oscillators // num_heads
        if head_dim % 2 != 0:
            raise ValueError("num_oscillators / num_heads must be even for 1D RoPE")
        if not 0.0 <= attn_drop < 1.0:
            raise ValueError("attn_drop must lie in [0, 1)")
        if not 0.0 <= proj_drop < 1.0:
            raise ValueError("proj_drop must lie in [0, 1)")

        self.num_oscillators = num_oscillators
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.sensitivity = PerOscillatorMLP(num_oscillators, hidden_features=2)
        self.influence = PerOscillatorMLP(num_oscillators, hidden_features=4)
        self.W_qkv = nn.Linear(num_oscillators, 3 * num_oscillators, bias=True)
        self.W_o = nn.Linear(num_oscillators, num_oscillators, bias=True)
        self.output_norm = RMSNorm(num_oscillators)

    def _key_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch: int,
        length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2 or attention_mask.shape != (batch, length):
            raise ValueError(
                "attention_mask must have shape [batch, sequence] and mask keys only"
            )
        valid_keys = attention_mask.to(device=device, dtype=torch.bool)
        invalid_samples = (~valid_keys.any(dim=-1)).nonzero(as_tuple=False).flatten()
        if invalid_samples.numel() > 0:
            indices = invalid_samples.detach().cpu().tolist()
            raise ValueError(f"attention_mask has no valid keys for samples {indices}")
        return valid_keys[:, None, None, :]

    def _project_qkv(
        self, influence: torch.Tensor, rope_fn: Optional[nn.Module]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = influence.shape
        q, k, v = self.W_qkv(influence).chunk(3, dim=-1)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        return q, k, v

    def _explicit_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        dropped_weights = F.dropout(
            weights,
            p=self.attn_drop if not deterministic else 0.0,
            training=not deterministic,
        )
        return torch.matmul(dropped_weights, v), weights

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        rope_fn: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if theta.shape != omega.shape:
            raise ValueError("theta and omega must have the same shape")
        if theta.shape[-1] != self.num_oscillators:
            raise ValueError(
                f"expected {self.num_oscillators} oscillators, got {theta.shape[-1]}"
            )
        batch, length, _ = theta.shape
        influence = self.influence(theta)
        q, k, v = self._project_qkv(influence, rope_fn)
        key_mask = self._key_mask(attention_mask, batch, length, theta.device)

        weights = None
        if collect_diagnostics:
            message, weights = self._explicit_attention(
                q, k, v, key_mask, deterministic
            )
        else:
            message = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_mask,
                dropout_p=self.attn_drop if not deterministic else 0.0,
            )

        message = message.transpose(1, 2).reshape(batch, length, -1)
        field = F.relu(self.output_norm(self.W_o(message)))
        field = F.dropout(
            field,
            p=self.proj_drop if not deterministic else 0.0,
            training=not deterministic,
        )
        phase_velocity = omega + self.sensitivity(theta) * field

        diagnostics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            diagnostics = {
                "attention_entropy": -(
                    weights * safe_weights.log()
                ).sum(dim=-1).mean(),
                "coupling_field_rms": field.float().square().mean().sqrt(),
            }
        return phase_velocity, diagnostics


class WONNLayer(nn.Module):
    """One coupling layer containing recurrent fast-phase steps."""

    def __init__(
        self,
        num_oscillators: int,
        num_heads: int,
        num_inner_steps: int,
        step_init: float = 0.1,
        step_max: float = 0.25,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
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
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        update_squares = []
        attention_entropies = []
        field_rms_values = []
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
                field_rms_values.append(
                    coupling_diagnostics["coupling_field_rms"]
                )

        diagnostics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            diagnostics = {
                "phase_update_rms": torch.stack(update_squares).mean().sqrt(),
                "attention_entropy": torch.stack(attention_entropies).mean(),
                "coupling_field_rms": torch.stack(field_rms_values).mean(),
            }
        return theta, diagnostics


class OmegaTransition(nn.Module):
    """Per-token residual frequency update between adjacent WONN layers."""

    def __init__(
        self,
        num_oscillators: int,
        alpha_init: float = 0.1,
        alpha_max: float = 0.25,
    ):
        super().__init__()
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must lie strictly between 0 and alpha_max")
        self.num_oscillators = num_oscillators
        self.alpha_max = alpha_max
        raw_alpha = math.log(alpha_init / (alpha_max - alpha_init))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))
        self.theta_embedding = ThetaEmbedding(num_oscillators)
        self.transition_norm = RMSNorm(2 * num_oscillators)
        self.input_projection = _make_linear(
            2 * num_oscillators, num_oscillators, bias=True
        )
        self.output_projection = _make_linear(
            num_oscillators, num_oscillators, bias=True
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if theta.shape != omega.shape:
            raise ValueError("theta and omega must have the same shape")
        theta_embedding = self.theta_embedding(theta)
        transition_input = self.transition_norm(
            torch.cat([theta_embedding, omega], dim=-1)
        )
        hidden = F.relu(self.input_projection(transition_input))
        delta_omega = self.output_projection(hidden)
        omega_update = self.alpha.to(omega.dtype) * delta_omega
        next_omega = omega + omega_update

        diagnostics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            diagnostics = {
                "omega_norm": next_omega.float().square().mean().sqrt(),
                "omega_delta_rms": delta_omega.float().square().mean().sqrt(),
                "omega_update_rms": omega_update.float().square().mean().sqrt(),
                "alpha": self.alpha.float(),
            }
        return theta, next_omega, diagnostics

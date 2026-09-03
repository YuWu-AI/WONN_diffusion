"""ELF-compatible language model backed by Winfree oscillator dynamics."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    BottleneckTextProj,
    DEFAULT_BIAS_INIT,
    DEFAULT_KERNEL_INIT,
    FinalLayer,
    NORMAL_INIT_002,
    RMSNorm,
    TextRotaryEmbeddingFast,
    TimestepEmbedder,
    _make_linear,
)
from modules.wonn_layers import OmegaTransition, WONNLayer, phase_features


class WONNELF(nn.Module):
    """WONN backbone with the same public forward contract as :class:`ELF`."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        num_oscillators: int = 384,
        num_layers: int = 6,
        num_inner_steps: int = 2,
        num_heads: int = 12,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        step_init: float = 0.1,
        step_max: float = 0.25,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_oscillators % num_heads != 0:
            raise ValueError("num_oscillators must be divisible by num_heads")
        if (num_oscillators // num_heads) % 2 != 0:
            raise ValueError("num_oscillators / num_heads must be even for 1D RoPE")
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive")

        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.num_oscillators = num_oscillators
        self.hidden_size = 2 * num_oscillators
        self.depth = num_layers
        self.num_inner_steps = num_inner_steps
        self.num_heads = num_heads
        self.head_dim = num_oscillators // num_heads
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.proj_drop = proj_drop
        self.gradient_checkpointing = gradient_checkpointing

        self.self_cond_proj = _make_linear(
            2 * text_encoder_dim, text_encoder_dim, bias=True
        )
        self.text_proj = BottleneckTextProj(
            text_encoder_dim, self.hidden_size, bottleneck_dim
        )

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.t_emb_tokens = nn.Parameter(
            torch.empty(1, num_time_tokens, self.hidden_size)
        )
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(self.hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, self.hidden_size)
            )
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                torch.empty(1, num_model_mode_tokens, self.hidden_size)
            )
            NORMAL_INIT_002(self.mode_tokens)

        prefix_total = (
            num_model_mode_tokens + num_time_tokens + num_self_cond_cfg_tokens
        )
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=self.head_dim,
            pt_seq_len=max_length,
            num_empty_token=prefix_total,
        )

        self.adapter_norm = RMSNorm(self.hidden_size)
        self.phase_projection = _make_linear(
            self.hidden_size, 2 * num_oscillators, bias=True
        )
        self.frequency_projection = _make_linear(
            self.hidden_size, num_oscillators, bias=True
        )

        self.layers = nn.ModuleList(
            [
                WONNLayer(
                    num_oscillators=num_oscillators,
                    num_heads=num_heads,
                    num_inner_steps=num_inner_steps,
                    step_init=step_init,
                    step_max=step_max,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(num_layers)
            ]
        )
        self.omega_transitions = nn.ModuleList(
            [OmegaTransition(num_oscillators) for _ in range(num_layers - 1)]
        )

        self.final_layer = FinalLayer(
            self.hidden_size, patch_size=1, out_channels=text_encoder_dim
        )
        self.proj_kernel = nn.Parameter(
            torch.empty(self.hidden_size, text_encoder_dim)
        )
        self.proj_bias = nn.Parameter(torch.empty(text_encoder_dim))
        self.unembed_kernel = nn.Parameter(
            torch.empty(text_encoder_dim, vocab_size)
        )
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def build_context(
        self,
        t: torch.Tensor,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
    ) -> list:
        batch = t.shape[0]
        time_embedding = self.t_embedder(t)
        prefix_tokens = [
            self.t_emb_tokens.expand(batch, -1, -1)
            + time_embedding.unsqueeze(1)
        ]
        if self.num_self_cond_cfg_tokens > 0:
            sc_tokens = self.self_cond_cfg_tokens.expand(batch, -1, -1)
            if self_cond_cfg_scale is not None:
                scale_embedding = self.self_cond_cfg_embedder(
                    self_cond_cfg_scale
                )
                sc_tokens = sc_tokens + scale_embedding.unsqueeze(1)
            prefix_tokens.append(sc_tokens)
        return prefix_tokens

    def _prepare_sequence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        self_cond_cfg_scale: Optional[torch.Tensor],
        decoder_step_active,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        batch = x.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            elif x.shape[-1] != self.text_encoder_dim:
                raise ValueError(
                    "x must have text_encoder_dim or 2 * text_encoder_dim channels"
                )
            x = self.text_proj(x.float())
            context_tokens = self.build_context(t, self_cond_cfg_scale)

        mode_length = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(batch, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif (
                isinstance(decoder_step_active, torch.Tensor)
                and decoder_step_active.dim() > 0
            ):
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(
                    -1, 1, 1
                )
            else:
                active_gate = float(decoder_step_active)
            x = torch.cat([mode_tokens * active_gate, x], dim=1)
            mode_length = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(
                    (batch, mode_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        context_length = sum(tokens.shape[1] for tokens in context_tokens)
        if context_tokens:
            x = torch.cat([torch.cat(context_tokens, dim=1), x], dim=1)
            if attention_mask is not None:
                context_mask = torch.ones(
                    (batch, context_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    [context_mask, attention_mask], dim=1
                )
        return x, attention_mask, context_length + mode_length

    def _initialize_states(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.adapter_norm(hidden)
        phase_pairs = self.phase_projection(hidden).reshape(
            *hidden.shape[:-1], self.num_oscillators, 2
        )
        # atan2(0, 0) has a finite forward value but undefined derivatives.
        # Padding/masked training rows can produce exactly zero projected
        # pairs, so map only those pairs to the neutral unit direction before
        # the fp32 polar conversion. Non-zero pairs retain the original phase.
        phase_pairs_f32 = phase_pairs.float()
        radius_sq = phase_pairs_f32.square().sum(dim=-1, keepdim=True)
        phase_pairs_f32 = phase_pairs_f32 * torch.rsqrt(radius_sq + 1e-6)
        neutral = torch.zeros_like(phase_pairs_f32)
        neutral[..., 0] = 1.0
        phase_pairs_f32 = torch.where(radius_sq > 0, phase_pairs_f32, neutral)
        theta = torch.atan2(
            phase_pairs_f32[..., 1], phase_pairs_f32[..., 0]
        ).to(phase_pairs.dtype)
        omega = self.frequency_projection(hidden)
        return theta, omega

    def _run_dynamics(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        deterministic: bool,
        collect_diagnostics: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        diagnostics: Dict[str, torch.Tensor] = {}
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not collect_diagnostics
        )
        for index, layer in enumerate(self.layers):
            transition = (
                self.omega_transitions[index]
                if index < len(self.omega_transitions)
                else None
            )
            if use_checkpoint:
                def _stage_forward(
                    phase: torch.Tensor,
                    frequency: torch.Tensor,
                    layer: WONNLayer = layer,
                    transition: Optional[OmegaTransition] = transition,
                ):
                    next_phase, _ = layer(
                        phase,
                        frequency,
                        rope_fn=self.feat_rope,
                        attention_mask=attention_mask,
                        deterministic=deterministic,
                    )
                    next_frequency = frequency
                    if transition is not None:
                        next_phase, next_frequency, _ = transition(
                            next_phase, frequency
                        )
                    return next_phase, next_frequency

                theta, omega = checkpoint(
                    _stage_forward, theta, omega, use_reentrant=False
                )
            else:
                theta, layer_diagnostics = layer(
                    theta,
                    omega,
                    rope_fn=self.feat_rope,
                    attention_mask=attention_mask,
                    deterministic=deterministic,
                    collect_diagnostics=collect_diagnostics,
                )
                if collect_diagnostics:
                    diagnostics.update(
                        {
                            f"layer_{index}/{key}": value
                            for key, value in layer_diagnostics.items()
                        }
                    )
                if transition is not None:
                    theta, omega, transition_diagnostics = transition(
                        theta,
                        omega,
                        collect_diagnostics=collect_diagnostics,
                    )
                    if collect_diagnostics:
                        diagnostics.update(
                            {
                                f"transition_{index}/{key}": value
                                for key, value in transition_diagnostics.items()
                            }
                        )
        if collect_diagnostics:
            mean_sin = torch.sin(theta.float()).mean()
            mean_cos = torch.cos(theta.float()).mean()
            diagnostics.update(
                {
                    "kuramoto_order": torch.sqrt(
                        mean_sin.square() + mean_cos.square()
                    ),
                    "phase_finite_fraction": torch.isfinite(theta).float().mean(),
                    "frequency_finite_fraction": torch.isfinite(omega).float().mean(),
                }
            )
        return theta, omega, diagnostics

    def _forward_impl(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        collect_diagnostics: bool = False,
    ):
        hidden, attention_mask, control_length = self._prepare_sequence(
            x,
            t,
            attention_mask,
            self_cond_cfg_scale,
            decoder_step_active,
        )
        theta, omega = self._initialize_states(hidden)
        theta, _, diagnostics = self._run_dynamics(
            theta,
            omega,
            attention_mask=attention_mask,
            deterministic=deterministic,
            collect_diagnostics=collect_diagnostics,
        )
        hidden = phase_features(theta)[:, control_length:]
        if self.proj_drop > 0.0:
            hidden = F.dropout(
                hidden, p=self.proj_drop, training=not deterministic
            )

        with torch.amp.autocast("cuda", enabled=False):
            hidden_f32 = hidden.float()
            decoder_logits = None
            if decoder_step_active is not None:
                decoder_hidden = F.gelu(
                    hidden_f32 @ self.proj_kernel + self.proj_bias,
                    approximate="tanh",
                )
                decoder_logits = (
                    decoder_hidden @ self.unembed_kernel + self.unembed_bias
                )
            output = self.final_layer(hidden_f32)
        return output, decoder_logits, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output, decoder_logits, _ = self._forward_impl(
            x,
            t,
            attention_mask=attention_mask,
            deterministic=deterministic,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,
        )
        return output, decoder_logits

    def forward_with_diagnostics(self, *args, **kwargs):
        """Run the same stateless forward while returning scalar dynamics data."""
        kwargs["collect_diagnostics"] = True
        return self._forward_impl(*args, **kwargs)


def WONN_ELF_B(**kwargs):
    defaults = {
        "num_oscillators": 384,
        "num_layers": 6,
        "num_inner_steps": 2,
        "num_heads": 12,
    }
    defaults.update(kwargs)
    return WONNELF(**defaults)

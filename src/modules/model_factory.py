"""Backbone registry and config-aware model construction."""

from modules.model import ELF_models
from modules.wonn_model import WONN_ELF_B


MODEL_FACTORIES = {
    **ELF_models,
    "ELF-WONN-B": WONN_ELF_B,
}


def build_model(
    config,
    *,
    text_encoder_dim: int,
    max_length: int,
    vocab_size: int,
):
    if config.model not in MODEL_FACTORIES:
        available = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"Unknown model {config.model!r}; available: {available}")

    common_kwargs = {
        "text_encoder_dim": text_encoder_dim,
        "max_length": max_length,
        "attn_drop": config.attn_dropout,
        "proj_drop": config.proj_dropout,
        "num_time_tokens": config.num_time_tokens,
        "num_self_cond_cfg_tokens": config.num_self_cond_cfg_tokens,
        "vocab_size": vocab_size,
        "num_model_mode_tokens": config.num_model_mode_tokens,
        "bottleneck_dim": config.bottleneck_dim,
        "gradient_checkpointing": bool(
            getattr(config, "gradient_checkpointing", False)
        ),
    }
    if config.model == "ELF-WONN-B":
        common_kwargs.update(
            {
                "num_oscillators": config.wonn_num_oscillators,
                "num_layers": config.wonn_num_layers,
                "num_inner_steps": config.wonn_num_inner_steps,
                "num_heads": config.wonn_num_heads,
                "qk_head_dim": config.wonn_qk_head_dim,
                "step_init": config.wonn_step_init,
                "step_max": config.wonn_step_max,
                "coupling_mode": config.wonn_coupling_mode,
            }
        )
        return MODEL_FACTORIES[config.model](**common_kwargs)
    return MODEL_FACTORIES[config.model](**common_kwargs)

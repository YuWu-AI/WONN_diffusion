"""Backbone factories shared by the reusable contract test mixins."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modules.model import ELF
from modules.wonn_model import WONNELF


def make_tiny_elf(
    *, self_cond_tokens=0, gradient_checkpointing=False, dropout=0.0, depth=2
):
    torch.manual_seed(7)
    return ELF(
        text_encoder_dim=16,
        max_length=6,
        hidden_size=32,
        depth=depth,
        num_heads=4,
        mlp_ratio=2.0,
        attn_drop=dropout,
        proj_drop=dropout,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=self_cond_tokens,
        num_model_mode_tokens=1,
        vocab_size=23,
        gradient_checkpointing=gradient_checkpointing,
    )


def make_tiny_wonn(
    *, self_cond_tokens=0, gradient_checkpointing=False, dropout=0.0, depth=2
):
    torch.manual_seed(7)
    return WONNELF(
        text_encoder_dim=16,
        max_length=6,
        num_oscillators=16,
        num_layers=depth,
        num_inner_steps=1,
        num_heads=4,
        attn_drop=dropout,
        proj_drop=dropout,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=self_cond_tokens,
        num_model_mode_tokens=1,
        vocab_size=23,
        gradient_checkpointing=gradient_checkpointing,
    )

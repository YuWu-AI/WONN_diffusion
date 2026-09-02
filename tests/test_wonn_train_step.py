import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs.config import Config
from tests.contract_factories import make_tiny_wonn
from train_step import train_step
from utils.train_utils import TrainState


class DummyEncoder(nn.Module):
    def __init__(self, vocab_size=23, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        del attention_mask, deterministic
        return self.embedding(input_ids)


class WONNTrainStepTest(unittest.TestCase):
    def test_mixed_objective_train_step_updates_wonn(self):
        torch.manual_seed(53)
        model = make_tiny_wonn(gradient_checkpointing=True).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = TrainState(
            model=model,
            optimizer=optimizer,
            ema_params1=TrainState.init_ema(model),
            dropout_generator=torch.Generator().manual_seed(59),
        )
        config = Config()
        config.use_bf16 = False
        config.max_length = 6
        config.pad_token = "pad"
        config.latent_mean = 0.0
        config.latent_std = 1.0
        config.num_self_cond_cfg_tokens = 1
        config.self_cond_prob = 0.5
        config.decoder_prob = 0.5
        config.label_drop_prob = 0.0
        config.grad_accum_steps = 1

        batch = {
            "input_ids": torch.randint(0, 23, (2, 6)),
            "encoder_attention_mask": torch.ones(2, 6, 6),
            "attention_mask": torch.ones(2, 6),
            "cond_seq_mask": torch.tensor(
                [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
                dtype=torch.float32,
            ),
        }
        before = model.phase_projection.weight.detach().clone()
        state, metrics = train_step(
            state,
            encoder=DummyEncoder(),
            batch=batch,
            config=config,
        )
        self.assertEqual(state.step, 1)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, model.phase_projection.weight))

if __name__ == "__main__":
    unittest.main()

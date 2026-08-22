import tempfile
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.contract_factories import make_tiny_wonn
from utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.train_utils import TrainState


class CheckpointArchitectureTest(unittest.TestCase):
    @staticmethod
    def _state(model):
        return TrainState(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            ema_params1=TrainState.init_ema(model),
        )

    def test_legacy_wonn_checkpoint_fails_without_partial_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(make_tiny_wonn())
            state.step = 10
            save_checkpoint(state, temporary, step=10)
            checkpoint_path = Path(temporary) / "checkpoint_10"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint["params"]["layers.0.coupling.q_proj.weight"] = (
                checkpoint["params"].pop("layers.0.coupling.W_qkv.weight")
            )
            torch.save(checkpoint, checkpoint_path)

            restored_state = self._state(make_tiny_wonn())
            with self.assertRaisesRegex(
                ValueError, "incompatible.*old WONN checkpoints.*not.*partially loaded"
            ):
                load_checkpoint(str(checkpoint_path), restored_state)


if __name__ == "__main__":
    unittest.main()

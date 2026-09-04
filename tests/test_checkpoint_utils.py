import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

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

    def test_checkpoint_filename_can_use_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(torch.nn.Linear(2, 2))
            state.step = 160000
            save_checkpoint(state, temporary, step=5000)
            checkpoint_path = Path(temporary) / "checkpoint_5000"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 160000)
            self.assertEqual(payload["checkpoint_step"], 5000)

            restored = self._state(torch.nn.Linear(2, 2))
            restored, resume_step = load_checkpoint(str(checkpoint_path), restored)
            self.assertEqual(resume_step, 160000)
            self.assertEqual(restored.step, 160000)

    def test_failed_checkpoint_write_never_leaves_a_final_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(torch.nn.Linear(2, 2))

            def fail_after_partial_write(_payload, path):
                Path(path).write_bytes(b"partial")
                raise OSError("simulated interrupted write")

            with mock.patch(
                "utils.checkpoint_utils.torch.save", side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save_checkpoint(state, temporary, step=10)

            self.assertFalse((Path(temporary) / "checkpoint_10").exists())
            self.assertFalse((Path(temporary) / ".checkpoint_10.tmp").exists())

    def test_checkpoint_label_must_match_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(torch.nn.Linear(2, 2))
            save_checkpoint(state, temporary, step=10)
            checkpoint = Path(temporary) / "checkpoint_10"
            checkpoint.rename(Path(temporary) / "checkpoint_20")

            with self.assertRaisesRegex(ValueError, "checkpoint label mismatch"):
                load_checkpoint(
                    str(Path(temporary) / "checkpoint_20"),
                    self._state(torch.nn.Linear(2, 2)),
                )


if __name__ == "__main__":
    unittest.main()

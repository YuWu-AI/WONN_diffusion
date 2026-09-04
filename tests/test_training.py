import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train import (
    _elapsed_training_offset,
    _finalize_training,
    _reconcile_metrics_file,
    _requested_checkpoint_step,
    _resolve_step_schedule,
    _resolve_warmup_optimizer_steps,
    _resume_position,
)
from configs.config import resolve_batch_sizes
from utils.checkpoint_utils import (
    load_checkpoint,
    load_warmstart_checkpoint,
    save_checkpoint,
)
from utils.train_utils import TrainState

class TrainingScheduleTest(unittest.TestCase):
    def test_global_batch_must_divide_world_size(self):
        self.assertEqual(resolve_batch_sizes(48, None, 2), (24, 48))
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            resolve_batch_sizes(49, None, 2)

    def test_per_device_batch_resolves_total_batch(self):
        self.assertEqual(resolve_batch_sizes(None, 12, 4), (12, 48))

    def test_global_batch_includes_gradient_accumulation(self):
        self.assertEqual(resolve_batch_sizes(512, None, 1, 32), (16, 16))
        with self.assertRaisesRegex(ValueError, r"world_size \* grad_accum_steps"):
            resolve_batch_sizes(500, None, 1, 32)

    def test_warmup_steps_are_optimizer_steps(self):
        self.assertEqual(_resolve_warmup_optimizer_steps(5000, None, 100, 32), 5000)
        self.assertEqual(_resolve_warmup_optimizer_steps(-1, 2.0, 160, 32), 10)

    def test_exact_optimizer_budget_and_requested_checkpoints(self):
        optimizer_steps, train_steps, save_steps = _resolve_step_schedule(
            num_train_steps=1000,
            grad_accum_steps=4,
            max_optimizer_steps=100,
            save_optimizer_steps="20, 50,100",
        )
        self.assertEqual(optimizer_steps, 100)
        self.assertEqual(train_steps, 400)
        self.assertEqual(save_steps, {20, 50, 100})

    def test_requested_checkpoint_outside_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no larger than the training budget"):
            _resolve_step_schedule(1000, 4, 100, "101")

    def test_staged_stop_preserves_optimizer_schedule_budget(self):
        optimizer_steps, train_steps, save_steps = _resolve_step_schedule(
            num_train_steps=1000,
            grad_accum_steps=4,
            max_optimizer_steps=100,
            save_optimizer_steps="20,50",
            stop_optimizer_steps=50,
        )
        self.assertEqual(optimizer_steps, 100)
        self.assertEqual(train_steps, 200)
        self.assertEqual(save_steps, {20, 50})

    def test_staged_stop_cannot_exceed_optimizer_schedule_budget(self):
        with self.assertRaisesRegex(ValueError, "optimizer schedule budget"):
            _resolve_step_schedule(1000, 4, 100, "", stop_optimizer_steps=101)

    def test_resume_position_uses_checkpoint_step(self):
        self.assertEqual(_resume_position(250, 100), (2, 50))

    def test_requested_checkpoint_only_fires_after_optimizer_update(self):
        requested = {20}
        self.assertIsNone(_requested_checkpoint_step(79, 4, requested))
        self.assertEqual(_requested_checkpoint_step(80, 4, requested), 20)
        self.assertIsNone(_requested_checkpoint_step(81, 4, requested))


class TrainingMetricsTest(unittest.TestCase):
    def test_reconcile_keeps_latest_unique_records_through_resume_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "train_metrics.jsonl"
            rows = [
                {"step": 100, "loss": 1.0},
                {"step": 200, "loss": 0.9},
                {"step": 200, "loss": 0.8},
                {"step": 300, "loss": 0.7},
            ]
            metrics_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows) + "truncated",
                encoding="utf-8",
            )

            summary = _reconcile_metrics_file(str(metrics_path), resume_step=200)

            reconciled = [
                json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(reconciled, [rows[0], rows[2]])
            self.assertEqual(
                summary,
                {"lines": 5, "kept": 2, "duplicates": 1, "discarded": 2},
            )

    def test_elapsed_training_offset_ignores_malformed_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "train_metrics.jsonl"
            metrics_path.write_text(
                '{"elapsed_training_seconds": 10.5}\n'
                'not-json\n'
                '{"elapsed_training_seconds": 25.0}\n',
                encoding="utf-8",
            )
            self.assertEqual(_elapsed_training_offset(str(metrics_path)), 25.0)


class TrainingFinalizationTest(unittest.TestCase):
    @mock.patch("train.run_generation")
    @mock.patch("train.save_checkpoint")
    def test_finalization_saves_and_runs_generation(self, save_mock, generation_mock):
        state = SimpleNamespace(step=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(output_dir=tmpdir, hf_repo_id=None)

            _finalize_training(
                state=state,
                encoder="encoder",
                eval_dataset="eval",
                tokenizer="tokenizer",
                config=config,
                generator="generator",
                local_batch_size=4,
                global_step=10000,
            )

        self.assertEqual(state.step, 10000)
        save_mock.assert_called_once_with(state, tmpdir, 10000, hf_repo_id=None)
        generation_mock.assert_called_once_with(
            state=state,
            encoder="encoder",
            eval_dataset="eval",
            tokenizer="tokenizer",
            config=config,
            generator="generator",
            local_batch_size=4,
        )

    @mock.patch("train.run_generation")
    @mock.patch("train.save_checkpoint")
    def test_finalization_reuses_requested_terminal_checkpoint(
        self, save_mock, generation_mock,
    ):
        state = SimpleNamespace(step=10000)
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "checkpoint_10000").write_bytes(b"checkpoint")
            config = SimpleNamespace(output_dir=tmpdir, hf_repo_id=None)
            _finalize_training(
                state=state,
                encoder=None,
                eval_dataset=None,
                tokenizer=None,
                config=config,
                generator=None,
                local_batch_size=4,
                global_step=10000,
            )

        save_mock.assert_not_called()
        generation_mock.assert_called_once()

    @mock.patch("train.run_generation")
    @mock.patch("train.save_checkpoint")
    def test_finalization_marks_training_before_skipping_pipeline_eval(
        self, save_mock, generation_mock,
    ):
        state = SimpleNamespace(step=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                output_dir=tmpdir, hf_repo_id=None, final_eval=False,
                grad_accum_steps=4,
            )
            _finalize_training(
                state=state,
                encoder=None,
                eval_dataset=None,
                tokenizer=None,
                config=config,
                generator=None,
                local_batch_size=4,
                global_step=20000,
                training_complete_payload={
                    "status": "complete",
                    "completed_optimizer_step": 5000,
                },
            )
            marker = json.loads(
                (Path(tmpdir) / "training_complete.json").read_text(encoding="utf-8")
            )

        self.assertEqual(marker["status"], "complete")
        self.assertEqual(marker["completed_optimizer_step"], 5000)
        self.assertEqual(marker["checkpoint"], str(Path(tmpdir) / "checkpoint_5000"))
        save_mock.assert_called_once()
        generation_mock.assert_not_called()


class TrainingCheckpointTest(unittest.TestCase):
    def test_warmstart_loads_compatible_weights_without_training_state(self):
        class OldModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = torch.nn.Linear(2, 2)
                self.obsolete = torch.nn.Linear(2, 2)

        class NewModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = torch.nn.Linear(2, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            old_model = OldModel()
            with torch.no_grad():
                old_model.core.weight.fill_(3.0)
                old_model.core.bias.fill_(4.0)
            old_state = TrainState(
                model=old_model,
                optimizer=torch.optim.AdamW(old_model.parameters(), lr=1e-3),
                ema_params1=TrainState.init_ema(old_model),
                step=10000,
                epoch=0.5,
            )
            save_checkpoint(old_state, tmpdir, step=10000)

            new_model = NewModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=2e-3)
            new_state = TrainState(
                model=new_model,
                optimizer=new_optimizer,
                ema_params1=TrainState.init_ema(new_model),
            )
            new_state, report = load_warmstart_checkpoint(
                str(Path(tmpdir) / "checkpoint_10000"), new_state
            )

            torch.testing.assert_close(
                new_state.model.core.weight,
                torch.full_like(new_state.model.core.weight, 3.0),
            )
            torch.testing.assert_close(
                new_state.ema_params1["core.bias"],
                torch.full_like(new_state.ema_params1["core.bias"], 4.0),
            )
            self.assertEqual(
                report["unexpected_keys"],
                ["obsolete.bias", "obsolete.weight"],
            )
            self.assertEqual(new_state.step, 0)
            self.assertEqual(new_state.epoch, 0.0)
            self.assertEqual(new_state.optimizer.state_dict()["state"], {})
            self.assertEqual(new_state.optimizer.param_groups[0]["lr"], 2e-3)

    def test_checkpoint_round_trip_preserves_fractional_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            state = TrainState(
                model=model,
                optimizer=optimizer,
                ema_params1=TrainState.init_ema(model),
                step=25,
                epoch=0.25,
            )
            save_checkpoint(state, tmpdir, step=25)

            restored_model = torch.nn.Linear(2, 2)
            restored_state = TrainState(
                model=restored_model,
                optimizer=torch.optim.AdamW(restored_model.parameters(), lr=1e-3),
                ema_params1=TrainState.init_ema(restored_model),
            )
            restored_state, restored_step = load_checkpoint(
                str(Path(tmpdir) / "checkpoint_25"), restored_state
            )

            self.assertEqual(restored_step, 25)
            self.assertEqual(restored_state.epoch, 0.25)

    def test_architecture_mismatch_fails_without_partial_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(2, 2)
            state = TrainState(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                ema_params1=TrainState.init_ema(model),
                step=10,
                epoch=0.1,
            )
            save_checkpoint(state, tmpdir, step=10)
            checkpoint_path = Path(tmpdir) / "checkpoint_10"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint["params"]["layers.0.coupling.q_proj.weight"] = (
                checkpoint["params"].pop("weight")
            )
            torch.save(checkpoint, checkpoint_path)

            restored_model = torch.nn.Linear(2, 2)
            restored_state = TrainState(
                model=restored_model,
                optimizer=torch.optim.AdamW(restored_model.parameters(), lr=1e-3),
                ema_params1=TrainState.init_ema(restored_model),
            )
            with self.assertRaisesRegex(
                ValueError, "incompatible.*old WONN checkpoints.*not.*partially loaded"
            ):
                load_checkpoint(str(checkpoint_path), restored_state)



if __name__ == "__main__":
    unittest.main()

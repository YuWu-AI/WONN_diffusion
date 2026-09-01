import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase5_pipeline_checks import (
    checkpoint_is_usable,
    diagnostic_is_complete,
    evaluation_is_complete,
    finalize_evaluation,
    write_pipeline_status,
)


class Phase5MechanismPipelineCheckTest(unittest.TestCase):
    def test_pipeline_is_not_bound_to_the_retired_worktree(self):
        pipeline = (
            REPO_ROOT / "scripts/run_phase5_mechanism_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("phase5-wmt14", pipeline)
        self.assertIn('$repo_root/.venv/bin/python', pipeline)

    def test_checkpoint_must_be_a_nonempty_file_not_a_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_20000"
            checkpoint.write_bytes(b"checkpoint")
            self.assertTrue(checkpoint_is_usable(checkpoint))

            empty = root / "checkpoint_30000"
            empty.touch()
            self.assertFalse(checkpoint_is_usable(empty))
            self.assertFalse(checkpoint_is_usable(root))

    def test_evaluation_completion_requires_matching_marker_metrics_and_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "ode-test"
            run.mkdir(parents=True)
            marker = {
                "status": "complete",
                "model": "S-Base",
                "step": 20000,
                "num_samples": 2,
            }
            (root / "evaluation_complete.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            (run / "metrics.jsonl").write_text(
                json.dumps({"step": 20000, "num_samples": 2}) + "\n",
                encoding="utf-8",
            )
            generated = run / "all_generated_test_20000.jsonl"
            generated.write_text(
                "\n".join(
                    json.dumps(
                        {"source": "src", "reference": "ref", "generated": "gen"}
                    )
                    for _ in range(2)
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(finalize_evaluation(root, "S-Base", 20000, 2))
            self.assertTrue(
                evaluation_is_complete(root, "S-Base", 20000, 2)
            )
            self.assertFalse(
                evaluation_is_complete(root, "S-Token", 20000, 2)
            )
            generated.write_text(
                json.dumps(
                    {"source": "src", "reference": "ref", "generated": "gen"}
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                evaluation_is_complete(root, "S-Base", 20000, 2)
            )

    def test_evaluation_completion_requires_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "ode-test"
            run.mkdir(parents=True)
            (run / "all_generated_test_20000.jsonl").write_text(
                json.dumps(
                    {"source": "src", "reference": "ref", "generated": "gen"}
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertFalse(finalize_evaluation(root, "S-Base", 20000, 1))

    def test_diagnostic_completion_requires_requested_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.json"
            path.write_text(
                json.dumps({"status": "complete", "models": {"S-Base": {}}}),
                encoding="utf-8",
            )
            self.assertTrue(diagnostic_is_complete(path, "S-Base"))
            self.assertFalse(diagnostic_is_complete(path, "S-Token"))

    def test_pipeline_status_is_written_atomically_with_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pipeline_status.json"
            write_pipeline_status(
                path,
                "failed",
                "experiment123",
                "runner456",
                "S-Base evaluation",
                789,
                13,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["experiment_commit"], "experiment123")
            self.assertEqual(payload["runner_commit"], "runner456")
            self.assertEqual(payload["stage"], "S-Base evaluation")
            self.assertEqual(payload["pid"], 789)
            self.assertEqual(payload["exit_code"], 13)


if __name__ == "__main__":
    unittest.main()

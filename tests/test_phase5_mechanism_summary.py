import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_phase5_mechanism import RUNS, summarize


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class Phase5MechanismSummaryTest(unittest.TestCase):
    def test_summarize_reads_all_runs_steps_splits_and_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            output = Path(temporary) / "analysis"
            for label, directory in RUNS.items():
                run = root / directory
                _write_json(
                    run / "training_complete.json",
                    {"completed_optimizer_step": 50000},
                )
                metrics = []
                for step in (5000, 10000, 20000, 30000, 40000, 50000):
                    metrics.append({
                        "step": step,
                        "loss": 1.0,
                        "l2_loss": 0.5,
                        "ce_loss": 2.0,
                    })
                    diagnostic = {
                        "models": {
                            label: {
                                "time_bins": [{"t": 0.0, "latent_mse": 0.1}],
                                "conditioning": {
                                    "correct": {"bleu": 2.0, "chrf2": 10.0},
                                    "zero": {"bleu": 1.0, "chrf2": 5.0},
                                },
                                "decoder_probe": {
                                    "clean_x0": {"ce_loss": 0.2},
                                    "metadata": "ignored",
                                },
                            }
                        }
                    }
                    for split in ("train", "heldout"):
                        _write_json(
                            run / f"diagnostics/checkpoint_{step}/{split}/diagnostics.json",
                            diagnostic,
                        )
                (run / "train_metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
                (run / "train_metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in metrics),
                    encoding="utf-8",
                )
                for step in (20000, 40000, 50000):
                    generated = run / (
                        f"evaluations/checkpoint_{step}/ode/all_generated_test_{step}.jsonl"
                    )
                    generated.parent.mkdir(parents=True, exist_ok=True)
                    generated.write_text(
                        json.dumps({"generated": "hello", "reference": "hello"}) + "\n",
                        encoding="utf-8",
                    )

            with mock.patch(
                "summarize_phase5_mechanism._metric_bundle",
                return_value={"bleu": 100.0, "chrf2": 100.0},
            ):
                result = summarize(root, output)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["tables"]["training"]), 18)
            self.assertEqual(len(result["tables"]["evaluation"]), 9)
            self.assertEqual(len(result["tables"]["diagnostics"]), 108)
            self.assertTrue((output / "mechanism_summary.json").is_file())
            self.assertTrue((output / "evaluation.csv").is_file())


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cloud_pipeline_checks import (
    checkpoint_is_usable,
    evaluation_is_complete,
    finalize_evaluation,
    write_pipeline_status,
)


class CloudPipelineChecksTest(unittest.TestCase):
    def test_checkpoint_requires_a_nonempty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertFalse(checkpoint_is_usable(root / "missing"))
            (root / "directory").mkdir()
            self.assertFalse(checkpoint_is_usable(root / "directory"))
            (root / "empty").touch()
            self.assertFalse(checkpoint_is_usable(root / "empty"))
            (root / "checkpoint").write_bytes(b"state")
            self.assertTrue(checkpoint_is_usable(root / "checkpoint"))

    def test_evaluation_completion_requires_marker_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = root / "ode-test-cond"
            sample_dir.mkdir(parents=True)
            rows = [
                {"source": "a", "reference": "b", "generated": "c"},
                {"source": "d", "reference": "e", "generated": "f"},
            ]
            (sample_dir / "all_generated_0_10.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (sample_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 10, "num_samples": 2}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(evaluation_is_complete(root, "ELF", 10, 2))
            self.assertTrue(finalize_evaluation(root, "ELF", 10, 2))
            self.assertTrue(evaluation_is_complete(root, "ELF", 10, 2))
            self.assertFalse(evaluation_is_complete(root, "WONN", 10, 2))

    def test_pipeline_status_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline_status.json"
            write_pipeline_status(path, "failed", "abc", "abc", "training", 123, 7)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["exit_code"], 7)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()

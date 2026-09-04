import re
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]*`")


def _project_markdown_files():
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "PROJECT_HANDOFF.md",
        REPO_ROOT / "CLAUDE.md",
        REPO_ROOT / "references" / "README.md",
        REPO_ROOT / "scripts" / "README.md",
        REPO_ROOT / "src" / "configs" / "training_configs" / "README.md",
    ]
    return roots + sorted((REPO_ROOT / "docs").rglob("*.md"))


class RepositoryHygieneTest(unittest.TestCase):
    def test_agents_rules_are_single_source(self):
        agents = REPO_ROOT / "AGENTS.md"
        self.assertTrue(agents.is_symlink())
        self.assertEqual(agents.readlink(), Path("CLAUDE.md"))

    def test_project_markdown_relative_links_resolve(self):
        missing = []
        for document in _project_markdown_files():
            self.assertTrue(document.is_file(), document)
            markdown = document.read_text(encoding="utf-8")
            prose = INLINE_CODE.sub("", FENCED_CODE.sub("", markdown))
            for raw_target in MARKDOWN_LINK.findall(prose):
                target = raw_target.strip().split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (document.parent / target).resolve()
                if not resolved.exists():
                    missing.append(f"{document.relative_to(REPO_ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_current_cloud_entrypoints_are_unambiguous(self):
        current = (
            "src/configs/training_configs/train_de-en-ELF-B-E0.yml",
            "src/configs/training_configs/train_de-en-WONN-L12K768T3-W0.yml",
            "src/configs/training_configs/train_de-en-WONN-L6K768T3-W1.yml",
            "src/configs/training_configs/train_de-en-WONN-L9K768T3-W2.yml",
            "scripts/run_cloud_matrix_pipeline.sh",
            "scripts/analyze_cloud_matrix.py",
        )
        for relative_path in current:
            self.assertTrue((REPO_ROOT / relative_path).is_file(), relative_path)

        active_docs = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "RESEARCH_PLAN.md",
            REPO_ROOT / "docs" / "CLOUD_PHASE1_RUNBOOK.md",
            REPO_ROOT / "docs" / "PROJECT_STRUCTURE.md",
        ]
        for document in active_docs:
            text = document.read_text(encoding="utf-8")
            self.assertNotIn("phase5-cloud", text, document)
            self.assertNotIn("cloud-60k", text, document)
            self.assertNotIn("run_cloud_pair_pipeline", text, document)


if __name__ == "__main__":
    unittest.main()

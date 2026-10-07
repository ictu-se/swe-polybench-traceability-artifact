"""Regression checks for the design failures identified in the original study."""
import unittest
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
from unittest.mock import Mock, patch

from analyze import gold, metrics, require_complete_models
from repository_benchmark import snapshot
from run_models import make_prompt


class EvaluationTests(unittest.TestCase):
    def test_new_files_cannot_be_counted_as_retrievable(self):
        patch = "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+pass\n"
        target = gold({"patch": patch, "test_patch": ""}, {"existing.py"})
        self.assertEqual(target[0]["status"], "added")
        self.assertFalse(target[0]["exists"])
        self.assertEqual(metrics(["existing.py"], target)["n_gold"], 0)

    def test_rename_uses_old_path(self):
        patch = "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
        target = gold({"patch": patch, "test_patch": ""}, {"old.py"})
        self.assertEqual(target[0]["path"], "old.py")
        self.assertEqual(metrics(["old.py"], target)["recall_1"], 1)

    def test_basename_and_suffix_are_not_exact_matches(self):
        target = [{"path": "pkg/a.py", "kind": "code", "exists": True, "status": "modified"}]
        self.assertEqual(metrics(["a.py"], target)["recall_1"], 0)
        self.assertEqual(metrics(["other/pkg/a.py"], target)["recall_1"], 0)

    def test_fixed_precision_budget_and_shared_joint_ranking(self):
        target = [{"path": "a.py", "kind": "code", "exists": True, "status": "modified"},
                  {"path": "test_a.py", "kind": "test", "exists": True, "status": "modified"}]
        result = metrics(["test_a.py", "a.py"], target)
        self.assertEqual(result["recall_1"], .5)
        self.assertEqual(result["code_recall_1"], 0)
        self.assertEqual(result["test_recall_1"], 1)
        self.assertEqual(result["precision_5"], .4)

    def test_duplicates_do_not_inflate_precision(self):
        target = [{"path": "a.py", "kind": "code", "exists": True, "status": "modified"}]
        self.assertEqual(metrics(["a.py", "a.py"], target)["precision_3"], 1/3)

    def test_deleted_file_remains_a_valid_navigation_target(self):
        p = "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-pass\n"
        target = gold({"patch": p, "test_patch": ""}, {"old.py"})
        self.assertEqual(target[0]["status"], "deleted")
        self.assertEqual(metrics(["old.py"], target)["recall_1"], 1)

    def test_repository_universe_is_not_filtered_by_solution_paths(self):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for name in ("src/relevant.py", "src/distractor.py", "docs/guide.txt"):
                body = b"repository content"
                info = tarfile.TarInfo("project-commit/" + name)
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
        response = Mock(content=raw.getvalue())
        task = {"instance_id": "example", "repo": "org/project", "base_commit": "abc",
                "patch": "secret future path", "test_patch": "secret new test"}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "repository_benchmark.requests.get", return_value=response):
            snapshot(task, Path(tmp))
            with gzip.open(Path(tmp) / "corpus/example.json.gz", "rt") as f:
                paths = {r["path"] for r in json.load(f)["files"]}
        self.assertEqual(paths, {"src/relevant.py", "src/distractor.py", "docs/guide.txt"})

    def test_prompt_excludes_reference_changes_and_hints(self):
        task = {"problem_statement": "Add the requested feature", "patch": "PATCH_SENTINEL",
                "test_patch": "TEST_SENTINEL", "hints_text": "HINT_SENTINEL"}
        retrieved = {"candidates": [{"id": 1, "path": "src/a.py", "excerpt": "SOURCE_SENTINEL"}]}
        paths = make_prompt(task, retrieved, "paths")
        content = make_prompt(task, retrieved, "content")
        for value in (paths, content):
            self.assertIn("[1] src/a.py", value)
            for forbidden in ("PATCH_SENTINEL", "TEST_SENTINEL", "HINT_SENTINEL"):
                self.assertNotIn(forbidden, value)
        self.assertNotIn("SOURCE_SENTINEL", paths)
        self.assertIn("SOURCE_SENTINEL", content)

    def test_incomplete_or_duplicate_model_runs_are_rejected(self):
        tasks = [{"instance_id": "task1", "repo": "org/project"}]
        records = [{"model": model, "task_id": "task1", "condition": condition, "seed": seed}
                   for model in ("qwen3-coder:30b", "devstral-small-2:24b")
                   for condition in ("paths", "content") for seed in (11, 29, 47)]
        require_complete_models(tasks, records)
        with self.assertRaises(ValueError):
            require_complete_models(tasks, records[:-1])
        with self.assertRaises(ValueError):
            require_complete_models(tasks, records + [records[0]])


if __name__ == "__main__":
    unittest.main()

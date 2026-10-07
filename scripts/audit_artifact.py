"""Check archived data integrity and the complete experimental matrix."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze import require_complete_models
from repository_benchmark import ROOT, load_rows, save_json
from run_models import make_prompt, SYSTEM, SCHEMA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    task_file = ROOT / "data/tasks.jsonl"
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    assert hashlib.sha256(task_file.read_bytes()).hexdigest() == provenance["feature_task_sha256"]
    tasks = load_rows(task_file)
    by_id = {t["instance_id"]: t for t in tasks}
    assert len(tasks) == len(by_id) == 184
    retrieval = {}
    for tid, task in by_id.items():
        row = json.loads((args.results / "retrieval" / (tid + ".json")).read_text())
        assert row["base_commit"] == task["base_commit"] and row["repo"] == task["repo"]
        universe = set(row["universe"])
        assert len(universe) == len(row["universe"]) == row["file_count"]
        candidates = [c["path"] for c in row["candidates"]]
        assert candidates == row["rankings"]["hybrid_rrf"][:40]
        assert [c["id"] for c in row["candidates"]] == list(range(1, len(candidates) + 1))
        for paths in row["rankings"].values():
            assert len(paths) == len(set(paths)) and set(paths).issubset(universe)
        assert len(row["archive_sha256"]) == 64
        retrieval[tid] = row
    records = [r for p in args.results.glob("model_*.jsonl") for r in load_rows(p)]
    require_complete_models(tasks, records)
    environments = {}
    for model in {r["model"] for r in records}:
        slug = model.replace(":", "_")
        environment = json.loads((args.results / "environment" / (slug + ".json")).read_text())
        manifest = (ROOT / "data/model_manifests" / (slug + ".json")).read_bytes()
        assert hashlib.sha256(manifest).hexdigest() == environment["model"]["digest"]
        environments[model] = environment
    for row in records:
        environment = environments[row["model"]]
        assert row["digest"] == environment["model"]["digest"]
        task = by_id[row["task_id"]]
        assert row["repo"] == task["repo"] and row["language"] == task["language"]
        inputs = retrieval[row["task_id"]]
        assert row["candidate_paths"] == [c["path"] for c in inputs["candidates"]]
        request = row["request"]
        assert request["model"] == row["model"]
        assert request["system"] == SYSTEM and request["format"] == SCHEMA
        assert request["prompt"] == make_prompt(task, inputs, row["condition"])
        assert request["options"] == {"temperature": .2, "top_p": .9, "top_k": 40,
                                      "seed": row["seed"], "num_ctx": 16384, "num_predict": 160}
        if row["valid"]:
            response = row["response"]
            ids = json.loads(response["response"])["ranking"]
            assert response["model"] == row["model"]
            assert len(ids) <= 10 and len(ids) == len(set(ids))
            assert all(type(i) is int and 1 <= i <= len(inputs["candidates"]) for i in ids)
            assert row["ranking"] == [inputs["candidates"][i-1]["path"] for i in ids]
        else:
            assert row["ranking"] == []
    summary = {"passed": True, "tasks": len(tasks), "retrieval_records": len(retrieval),
               "model_records": len(records), "checks": ["dataset checksum", "snapshot identity",
               "universe and candidate membership", "matrix completeness", "model and manifest digests",
               "exact prompts and settings", "response-to-ranking correspondence"]}
    save_json(args.results / "artifact_audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

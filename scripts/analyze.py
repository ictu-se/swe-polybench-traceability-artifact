"""Exact-path evaluation, target-availability audit and paired uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from repository_benchmark import ROOT, load_rows, save_json
from run_models import repeat_panel

KS = (1, 3, 5, 10)
TEST = re.compile(r"(^|/)(tests?|__tests__|specs?|__snapshots__)(/|$)|"
                  r"(^|/)(test_[^/]*|[^/]*[Tt]est(s)?\.java)|"
                  r"[^/]+[._](test|spec)\.[^/]+$")


def patch_targets(text):
    result = []
    for block in re.split(r"(?m)^diff --git ", text or "")[1:]:
        old = re.search(r"(?m)^--- (.*)$", block)
        new = re.search(r"(?m)^\+\+\+ (.*)$", block)
        if old and new:
            a, b = old[1].strip(), new[1].strip()
            a = a[2:] if a.startswith("a/") else a
            b = b[2:] if b.startswith("b/") else b
        else:
            rename_a = re.search(r"(?m)^rename from (.*)$", block)
            rename_b = re.search(r"(?m)^rename to (.*)$", block)
            if rename_a and rename_b:
                a, b = rename_a[1], rename_b[1]
            else:
                match = re.match(r"a/(.*?) b/(.*?)\n", block)
                if not match:
                    raise ValueError("Unsupported diff header")
                a, b = match.groups()
                if re.search(r"(?m)^new file mode ", block):
                    a = "/dev/null"
                if re.search(r"(?m)^deleted file mode ", block):
                    b = "/dev/null"
        status = "added" if a == "/dev/null" else "deleted" if b == "/dev/null" else "renamed" if a != b else "modified"
        result.append({"old": a, "new": b, "status": status})
    return result


def gold(task, universe):
    targets = {}
    for field in ("patch", "test_patch"):
        for change in patch_targets(task.get(field, "")):
            path = change["new"] if change["status"] == "added" else change["old"]
            kind = "test" if field == "test_patch" or TEST.search(path) else "code"
            # A test-patch occurrence takes precedence for duplicated file targets.
            if path not in targets or kind == "test":
                targets[path] = {**change, "path": path, "kind": kind,
                                 "exists": path in universe}
    return list(targets.values())


def metrics(prediction, targets):
    pred = list(dict.fromkeys(prediction))
    g = {t["path"] for t in targets if t["exists"] and t["status"] != "added"}
    code = {t["path"] for t in targets if t["path"] in g and t["kind"] == "code"}
    tests = g - code
    out = {"n_gold": len(g), "n_code": len(code), "n_test": len(tests),
           "n_returned_10": min(10, len(pred))}
    for k in KS:
        s = set(pred[:k])
        hit = len(g & s)
        out[f"recall_{k}"] = hit / len(g) if g else np.nan
        out[f"precision_{k}"] = hit / k
        out[f"returned_precision_{k}"] = hit / len(s) if s else 0.0
        for label, group in (("code", code), ("test", tests)):
            out[f"{label}_recall_{k}"] = len(s & group) / len(group) if group else np.nan
    out["mrr_10"] = next((1/i for i, p in enumerate(pred[:10], 1) if p in g), 0.0) if g else np.nan
    return out


def interval(values, groups=None, seed=20261007, n=10000):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    if not len(values):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if groups is None:
        means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    else:
        groups = np.asarray(groups)[valid]
        unique = sorted(set(groups))
        sums = np.array([values[groups == g].sum() for g in unique])
        counts = np.array([(groups == g).sum() for g in unique])
        selected = rng.integers(0, len(unique), (n, len(unique)))
        means = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def legacy_audit():
    """Audit the frozen historical inputs without rerunning unidentified models."""
    tasks = load_rows(ROOT / "data/legacy_tasks.jsonl")
    pools = json.loads((ROOT / "data/legacy_patch_pools.json").read_text())
    coverages = [len(set(t["gold_all_files"]) & set(t["candidate_paths"])) /
                 len(set(t["gold_all_files"])) for t in tasks]
    return {
        "tasks": len(tasks),
        "candidate_paths_all_patch_derived": all(
            set(t["candidate_paths"]).issubset(pools[t["repo"]]) for t in tasks),
        "code_gold_over_5": sum(len(t["gold_code_files"]) > 5 for t in tasks),
        "test_gold_over_5": sum(len(t["gold_test_files"]) > 5 for t in tasks),
        "mean_gold_candidate_coverage": float(np.mean(coverages)),
    }


def require_complete_models(tasks, records):
    expected_models = {"qwen3-coder:30b", "devstral-small-2:24b"}
    if {r["model"] for r in records} != expected_models:
        raise ValueError("Missing or unexpected model family in the fixed matrix")
    task_ids = {t["instance_id"] for t in tasks}
    panel = set(repeat_panel(tasks))
    expected = {(tid, condition, seed) for seed in (11, 29, 47)
                for tid in (task_ids if seed == 11 else panel)
                for condition in ("paths", "content")}
    for model in sorted(expected_models):
        actual_rows = [r for r in records if r["model"] == model]
        actual = {(r["task_id"], r["condition"], r["seed"]) for r in actual_rows}
        if actual != expected or len(actual_rows) != len(expected):
            raise ValueError(f"Incomplete or duplicate model matrix for {model}: "
                             f"missing={len(expected-actual)}, extra={len(actual-expected)}, "
                             f"rows={len(actual_rows)}, expected={len(expected)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    tasks = load_rows(ROOT / "data/tasks.jsonl")
    legacy_by_id = {t["task_id"]: t for t in load_rows(ROOT / "data/legacy_tasks.jsonl")}
    target_rows, rows, coverage = [], [], []
    targets_by_task = {}
    for task in tasks:
        tid = task["instance_id"]
        file = args.results / "retrieval" / (tid + ".json")
        if not file.exists():
            if args.allow_partial:
                continue
            raise SystemExit("Missing repository retrieval: " + tid)
        r = json.loads(file.read_text())
        universe = set(r["universe"])
        targets = gold(task, universe)
        targets_by_task[tid] = targets
        common = {"task_id": tid, "repo": task["repo"], "language": task["language"]}
        target_rows.extend({**common, **t} for t in targets)
        existing = {t["path"] for t in targets if t["exists"] and t["status"] != "added"}
        candidates = {c["path"] for c in r["candidates"]}
        legacy_candidates = set(legacy_by_id[tid]["candidate_paths"])
        coverage.append({**common, "repository_files": r["file_count"], "existing_gold": len(existing),
                         "existing_code": sum(t["path"] in existing and t["kind"] == "code" for t in targets),
                         "existing_tests": sum(t["path"] in existing and t["kind"] == "test" for t in targets),
                         "added_gold": sum(t["status"] == "added" for t in targets),
                         "unexplained_absent": sum(not t["exists"] and t["status"] != "added" for t in targets),
                         "candidate_ceiling_40": len(existing & candidates) / len(existing) if existing else np.nan,
                         "candidate_oracle_10": min(10, len(existing & candidates)) / len(existing) if existing else np.nan,
                         "full_universe_oracle_10": min(10, len(existing)) / len(existing) if existing else np.nan,
                         "legacy_candidate_count": len(legacy_candidates),
                         "legacy_candidates_absent": len(legacy_candidates - universe),
                         "legacy_absent_fraction": len(legacy_candidates - universe) / len(legacy_candidates) if legacy_candidates else np.nan,
                         "legacy_existing_gold_coverage": len(existing & legacy_candidates) / len(existing) if existing else np.nan,
                         **{method + "_coverage_40": len(existing & set(paths[:40])) / len(existing) if existing else np.nan
                            for method, paths in r["rankings"].items()},
                         "hints_nonempty": bool((task.get("hints_text") or "").strip()),
                         "hints_mention_gold_path": any(t["path"] in (task.get("hints_text") or "") for t in targets),
                         "issue_characters": len(task["problem_statement"]),
                         "binary_files": r["binary_count"], "truncated_files": r["truncated_count"]})
        for method, prediction in r["rankings"].items():
            rows.append({**common, "method": method, "condition": "retrieval", "seed": 11,
                         "valid": True, "elapsed_seconds": r["retrieval_seconds"].get(method, np.nan),
                         **metrics(prediction, targets)})
    model_records = [r for path in sorted(args.results.glob("model_*.jsonl"))
                     for r in load_rows(path)]
    if not args.allow_partial:
        require_complete_models(tasks, model_records)
    for r in model_records:
        if r["task_id"] not in targets_by_task:
            continue
        rows.append({k: r[k] for k in ("task_id", "repo", "language", "condition", "seed", "valid", "elapsed_seconds")}
                    | {"method": r["model"] + "/" + r["condition"],
                       **metrics(r["ranking"], targets_by_task[r["task_id"]])})
    out = args.results / "analysis"
    out.mkdir(exist_ok=True)
    save_json(out / "legacy_audit.json", legacy_audit())
    frame = pd.DataFrame(rows)
    if frame.duplicated(["task_id", "method", "seed"]).any():
        raise ValueError("Duplicate method-task-seed records")
    frame.to_csv(out / "task_metrics.csv", index=False)
    pd.DataFrame(target_rows).to_csv(out / "target_availability.csv", index=False)
    cov = pd.DataFrame(coverage)
    cov.to_csv(out / "candidate_coverage.csv", index=False)
    summaries = []
    main_rows = frame[frame.seed == 11]
    for method, group in main_rows.groupby("method"):
        summary = {"method": method, "n_tasks": len(group), "valid_rate": group.valid.mean(),
                   "mean_returned_10": group.n_returned_10.mean(),
                   "median_seconds": group.elapsed_seconds.dropna().median() if group.elapsed_seconds.notna().any() else np.nan}
        for col in [c for c in frame if c.startswith(("recall_", "precision_", "returned_precision_", "code_recall_", "test_recall_", "mrr_"))]:
            summary[col] = group[col].mean()
        lo, hi = interval(group.recall_10, group.repo)
        summary.update(recall_10_cluster_low=lo, recall_10_cluster_high=hi)
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    contrasts = []
    pairs = [("content_bm25", "path_bm25"), ("dense_minilm", "content_bm25"), ("hybrid_rrf", "content_bm25")]
    for model in ("qwen3-coder:30b", "devstral-small-2:24b"):
        pairs.extend([(model+"/content", model+"/paths"), (model+"/content", "hybrid_rrf")])
    for a, b in pairs:
        left = main_rows[main_rows.method == a][["task_id", "repo", "recall_10"]]
        right = main_rows[main_rows.method == b][["task_id", "recall_10"]]
        merged = left.merge(right, on="task_id", suffixes=("_a", "_b")).dropna()
        if merged.empty:
            continue
        d = merged.recall_10_a - merged.recall_10_b
        lo, hi = interval(d, merged.repo)
        tlo, thi = interval(d)
        contrasts.append({"a": a, "b": b, "n": len(d), "difference": d.mean(),
                          "cluster_low": lo, "cluster_high": hi, "task_low": tlo, "task_high": thi,
                          "wins": int((d>0).sum()), "ties": int((d==0).sum()), "losses": int((d<0).sum())})
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    model_rows = frame[frame.condition.isin(["paths", "content"])]
    repeat = model_rows.groupby(["method", "task_id"]).filter(lambda x: len(x) == 3)
    repeats = []
    for method, group in repeat.groupby("method"):
        by_task = group.groupby("task_id").recall_10
        repeats.append({"method": method, "n_tasks": group.task_id.nunique(),
                        "n_invalid": int((~group.valid).sum()),
                        "valid_rate": float(group.valid.mean()),
                        "mean_within_task_sd": by_task.std().mean(),
                        "mean_within_task_range": (by_task.max()-by_task.min()).mean(),
                        "seed_mean_min": group.groupby("seed").recall_10.mean().min(),
                        "seed_mean_max": group.groupby("seed").recall_10.mean().max()})
    pd.DataFrame(repeats).to_csv(out / "repeat_stability.csv", index=False)
    main_rows.groupby(["method", "language"])[["recall_10", "precision_10", "mrr_10"]].agg(["mean", "count"]).to_csv(out / "by_language.csv")
    stats = {"tasks": len(coverage), "repositories": cov.repo.nunique(), "targets": len(target_rows),
             "added_targets": sum(t["status"] == "added" for t in target_rows),
             "renamed_targets": sum(t["status"] == "renamed" for t in target_rows),
             "unexplained_absent": int(cov.unexplained_absent.sum()),
             "tasks_no_existing_gold": int((cov.existing_gold == 0).sum()),
             "tasks_no_existing_code": int((cov.existing_code == 0).sum()),
             "tasks_no_existing_test": int((cov.existing_tests == 0).sum()),
             "tasks_with_added": int((cov.added_gold > 0).sum()),
             "repository_files_median": float(cov.repository_files.median()),
             "repository_files_min": int(cov.repository_files.min()), "repository_files_max": int(cov.repository_files.max()),
             "candidate_ceiling_40": float(cov.candidate_ceiling_40.mean()),
             "candidate_oracle_10": float(cov.candidate_oracle_10.mean()),
             "full_universe_oracle_10": float(cov.full_universe_oracle_10.mean()),
             "tasks_existing_over_10": int((cov.existing_gold > 10).sum()),
             "issues_truncated_for_models": int((cov.issue_characters > 8000).sum()),
             "legacy_candidate_occurrences": int(cov.legacy_candidate_count.sum()),
             "legacy_candidate_occurrences_absent": int(cov.legacy_candidates_absent.sum()),
             "tasks_with_legacy_absent_candidates": int((cov.legacy_candidates_absent > 0).sum()),
             "mean_legacy_absent_fraction": float(cov.legacy_absent_fraction.mean()),
             "legacy_existing_gold_coverage": float(cov.legacy_existing_gold_coverage.mean()),
             "hints_nonempty": int(cov.hints_nonempty.sum()), "hints_mention_gold_path": int(cov.hints_mention_gold_path.sum()),
             "model_rows": len(model_rows), "complete_repeat_rows": len(repeat)}
    stats.update(invalid_model_rows=int((~model_rows.valid).sum()),
                 invalid_primary_model_rows=int((~model_rows[model_rows.seed == 11].valid).sum()),
                 invalid_extra_seed_rows=int((~model_rows[model_rows.seed != 11].valid).sum()))
    save_json(out / "summary.json", stats)
    print(json.dumps(stats, indent=2))
    print(summary[["method", "n_tasks", "recall_10", "precision_10", "valid_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()

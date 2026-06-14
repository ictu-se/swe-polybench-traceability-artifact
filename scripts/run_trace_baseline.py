import argparse
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_path(path):
    path = str(path or "").strip().strip("'\"`")
    path = path.replace("\\", "/")
    path = re.sub(r"^[ab]/", "", path)
    return path


def looks_like_test_path(path):
    lowered = path.lower()
    name = Path(path).name.lower()
    return (
        "/test/" in lowered
        or "/tests/" in lowered
        or lowered.startswith("test/")
        or lowered.startswith("tests/")
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("test.java")
        or name.endswith(".test.js")
        or name.endswith(".spec.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
    )


def tokens(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def score_path(problem_tokens, path):
    path_token_set = tokens(path.replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " "))
    if not path_token_set:
        return 0.0
    return len(problem_tokens & path_token_set) / len(path_token_set)


def rank(problem, candidates, want_tests):
    problem_tokens = tokens(problem)
    ranked = []
    for path in candidates:
        path = normalize_path(path)
        if not path:
            continue
        is_test = looks_like_test_path(path)
        score = score_path(problem_tokens, path)
        if want_tests and is_test:
            score += 0.25
        if not want_tests and not is_test:
            score += 0.25
        ranked.append((score, -len(path), path))
    ranked.sort(reverse=True)
    out = []
    for _, _, path in ranked:
        if path not in out:
            out.append(path)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(PROJECT / "data/swe_polybench_trace_tasks.jsonl"))
    parser.add_argument("--conditions", default="issue_only,issue_plus_inventory,issue_plus_hints")
    parser.add_argument("--out", default=str(PROJECT / "results/lexical_path_ranker_outputs.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    count = 0
    for task in load_jsonl(args.tasks):
        if args.limit and count >= args.limit:
            break
        problem = task.get("problem_statement", "")
        candidates = task.get("candidate_paths", [])
        code_files = rank(problem, candidates, want_tests=False)[:10]
        test_files = rank(problem, candidates, want_tests=True)[:10]
        payload = {
            "related_code_files": code_files,
            "related_test_files": test_files,
            "rationale": "Lexical path ranking over candidate path inventory.",
            "confidence": 0.35,
        }
        for condition in conditions:
            append_jsonl(out_path, {
                "task_id": task["task_id"],
                "repo": task["repo"],
                "language": task["language"],
                "task_category": task["task_category"],
                "condition": condition,
                "model": "lexical_path_ranker",
                "returncode": 0,
                "elapsed_sec": 0.0,
                "stdout": json.dumps(payload, ensure_ascii=False),
                "stderr": "",
            })
        count += 1
    print(f"wrote {count * len(conditions)} baseline outputs to {out_path}")


if __name__ == "__main__":
    main()

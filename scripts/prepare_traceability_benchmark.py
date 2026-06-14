import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[0]
DEFAULT_SOURCE = WORKSPACE / "datasets/11_from_source_code_to_user_goals__benchmarks/SWE-PolyBench_500/test.jsonl"

CONDITIONS = ("issue_only", "issue_plus_inventory", "issue_plus_hints")


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_path(path):
    path = str(path or "").strip().strip("'\"`")
    path = path.replace("\\", "/")
    path = re.sub(r"^[ab]/", "", path)
    path = re.sub(r"/+", "/", path)
    return "" if path in {"", "/dev/null", "dev/null"} else path


def patch_paths(patch_text):
    paths = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch_text or "", re.MULTILINE):
        for raw in match.groups():
            path = normalize_path(raw)
            if path and path not in paths:
                paths.append(path)
    return paths


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


def token_set(text):
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def path_tokens(path):
    path = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(path))
    return token_set(path.replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " "))


def lexical_rank(problem, paths, language=None, limit=80):
    issue_tokens = token_set(problem)
    ranked = []
    for path in paths:
        toks = path_tokens(path)
        overlap = len(issue_tokens & toks)
        score = overlap / max(len(toks), 1)
        score += 0.02 if language and language.lower() in path.lower() else 0.0
        score += 0.01 if looks_like_test_path(path) else 0.0
        ranked.append((score, -len(path), path))
    ranked.sort(reverse=True)
    return [path for _, _, path in ranked[:limit]]


def compact_text(text, max_chars):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"


def build_prompt(row, condition):
    inventory = "\n".join(f"- {path}" for path in row.get("candidate_paths", []))
    hints = compact_text(row.get("hints_text", ""), 1200)
    parts = [
        "You are performing issue-to-code-and-test traceability before seeing the patch.",
        "Return strict JSON only, with no markdown and no extra keys.",
        "",
        f"Repository: {row['repo']}",
        f"Language: {row['language']}",
        f"Task category: {row['task_category']}",
        "",
        "Issue or feature request:",
        compact_text(row["problem_statement"], 2600),
    ]
    if condition in {"issue_plus_inventory", "issue_plus_hints"}:
        parts.extend([
            "",
            "Candidate repository paths:",
            inventory or "(no candidate paths available)",
        ])
    if condition == "issue_plus_hints":
        parts.extend([
            "",
            "Additional public discussion hints:",
            hints or "(none)",
        ])
    parts.extend([
        "",
        "Predict files that are likely to need source-code changes and tests that should be changed or added.",
        "Return at most 5 related_code_files and at most 5 related_test_files.",
        "Keep the rationale under 25 words.",
        "Use paths from the candidate list when it is present. If uncertain, still provide ranked best guesses.",
        "",
        'Required JSON schema: {"related_code_files": ["path1"], "related_test_files": ["path2"], "rationale": "short evidence summary", "confidence": 0.0}',
    ])
    return "\n".join(parts)


def flatten_modified_nodes(value):
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    out = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                label = item.get("name") or item.get("full_name") or item.get("node") or ""
                if label:
                    out.append(str(label))
            elif item:
                out.append(str(item))
    return out


def select_rows(rows, category, max_per_language, seed):
    filtered = [row for row in rows if not category or row.get("task_category") == category]
    filtered.sort(key=lambda row: (row.get("language", ""), row.get("created_at", ""), row.get("instance_id", "")))
    if not max_per_language:
        return filtered
    rng = random.Random(seed)
    by_language = defaultdict(list)
    for row in filtered:
        by_language[row.get("language", "unknown")].append(row)
    selected = []
    for language, group in sorted(by_language.items()):
        if len(group) > max_per_language:
            group = rng.sample(group, max_per_language)
            group.sort(key=lambda row: row.get("instance_id", ""))
        selected.extend(group)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--category", default="Feature", help="Use empty string for all task categories.")
    parser.add_argument("--max-per-language", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    raw_rows = list(load_jsonl(args.source))
    repo_inventory = defaultdict(set)
    for row in raw_rows:
        for path in patch_paths(row.get("patch", "")) + patch_paths(row.get("test_patch", "")):
            repo_inventory[row["repo"]].add(path)

    selected = select_rows(raw_rows, args.category or "", args.max_per_language, args.seed)
    task_rows = []
    for row in selected:
        patch_file_paths = patch_paths(row.get("patch", ""))
        test_patch_file_paths = patch_paths(row.get("test_patch", ""))
        code_files = [path for path in patch_file_paths if not looks_like_test_path(path)]
        test_files = []
        for path in test_patch_file_paths + [p for p in patch_file_paths if looks_like_test_path(p)]:
            if path not in test_files:
                test_files.append(path)
        candidate_paths = lexical_rank(
            row.get("problem_statement", ""),
            sorted(repo_inventory[row["repo"]]),
            row.get("language", ""),
            args.candidate_limit,
        )
        task_rows.append({
            "task_id": row["instance_id"],
            "repo": row["repo"],
            "pull_number": row.get("pull_number", ""),
            "base_commit": row.get("base_commit", ""),
            "language": row.get("language", ""),
            "task_category": row.get("task_category", ""),
            "created_at": row.get("created_at", ""),
            "problem_statement": row.get("problem_statement", ""),
            "hints_text": row.get("hints_text", ""),
            "test_command": row.get("test_command", ""),
            "gold_code_files": code_files,
            "gold_test_files": test_files,
            "gold_all_files": sorted(set(code_files + test_files)),
            "modified_nodes": flatten_modified_nodes(row.get("modified_nodes")),
            "candidate_paths": candidate_paths,
            "candidate_path_count": len(repo_inventory[row["repo"]]),
        })

    packs = []
    for row in task_rows:
        for condition in CONDITIONS:
            packs.append({
                "task_id": row["task_id"],
                "repo": row["repo"],
                "language": row["language"],
                "task_category": row["task_category"],
                "condition": condition,
                "prompt": build_prompt(row, condition),
                "candidate_paths": row["candidate_paths"],
            })

    data_dir = PROJECT / "data"
    prompts_dir = PROJECT / "prompts"
    write_jsonl(data_dir / "swe_polybench_trace_tasks.jsonl", task_rows)
    write_jsonl(data_dir / "swe_polybench_feature_subset.jsonl", task_rows)
    write_jsonl(prompts_dir / "trace_prompt_packs.jsonl", packs)

    csv_path = data_dir / "swe_polybench_feature_subset.csv"
    csv_fields = [
        "task_id", "repo", "language", "task_category", "created_at",
        "gold_code_files", "gold_test_files", "gold_all_files",
        "candidate_path_count", "test_command",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in task_rows:
            csv_row = {key: row.get(key, "") for key in csv_fields}
            for key in ("gold_code_files", "gold_test_files", "gold_all_files"):
                csv_row[key] = json.dumps(csv_row[key], ensure_ascii=False)
            writer.writerow(csv_row)

    counts = Counter((row["language"], row["task_category"]) for row in task_rows)
    print(f"source rows: {len(raw_rows)}")
    print(f"selected tasks: {len(task_rows)}")
    print(f"prompt packs: {len(packs)}")
    for (language, category), count in sorted(counts.items()):
        print(f"{language:12s} {category:12s} {count}")
    print(f"wrote {data_dir / 'swe_polybench_trace_tasks.jsonl'}")
    print(f"wrote {prompts_dir / 'trace_prompt_packs.jsonl'}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_path(value):
    value = str(value or "").strip().strip("`'\"")
    value = value.split("::", 1)[0]
    value = value.replace("\\", "/")
    value = re.sub(r"^[ab]/", "", value)
    value = re.sub(r"/+", "/", value)
    return value


def list_field(parsed, key):
    if not isinstance(parsed, dict):
        return []
    value = parsed.get(key, [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("file") or item.get("path") or item.get("name") or ""
        norm = normalize_path(item)
        if norm and norm not in out:
            out.append(norm)
    return out


def path_hit(pred, gold, basename=False):
    pred = normalize_path(pred)
    gold = normalize_path(gold)
    if not pred or not gold:
        return False
    if pred == gold or pred.endswith("/" + gold) or gold.endswith("/" + pred):
        return True
    return basename and Path(pred).name == Path(gold).name


def recall_at(preds, golds, k, basename=False):
    if not golds:
        return ""
    top = preds[:k]
    hits = sum(1 for gold in golds if any(path_hit(pred, gold, basename=basename) for pred in top))
    return hits / len(golds)


def precision_at(preds, golds, k, basename=False):
    top = preds[:k]
    if not top:
        return 0.0
    hits = sum(1 for pred in top if any(path_hit(pred, gold, basename=basename) for gold in golds))
    return hits / len(top)


def f1(precision, recall):
    if precision == "" or recall == "":
        return ""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean(values):
    nums = [float(value) for value in values if value != ""]
    return sum(nums) / len(nums) if nums else ""


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    metrics = [
        "parse_ok", "code_recall_at_5", "code_recall_at_10", "code_precision_at_5",
        "test_recall_at_5", "test_recall_at_10", "test_precision_at_5",
        "all_recall_at_10", "basename_all_recall_at_10", "elapsed_sec",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for group_key, items in sorted(groups.items()):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary["n"] = len(items)
        for metric in metrics:
            summary[metric] = mean([item.get(metric, "") for item in items])
        out.append(summary)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs_jsonl")
    parser.add_argument("--tasks", default=str(PROJECT / "data/swe_polybench_trace_tasks.jsonl"))
    parser.add_argument("--out-dir", default=str(PROJECT / "results"))
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in load_jsonl(args.tasks)}
    rows = []
    for record in load_jsonl(args.outputs_jsonl):
        task = tasks.get(record["task_id"], {})
        parsed = extract_json(record.get("stdout", ""))
        code_preds = list_field(parsed, "related_code_files")
        test_preds = list_field(parsed, "related_test_files")
        all_preds = []
        for path in code_preds + test_preds:
            if path not in all_preds:
                all_preds.append(path)
        code_gold = task.get("gold_code_files", [])
        test_gold = task.get("gold_test_files", [])
        all_gold = task.get("gold_all_files", [])
        code_r5 = recall_at(code_preds, code_gold, 5)
        code_p5 = precision_at(code_preds, code_gold, 5)
        test_r5 = recall_at(test_preds, test_gold, 5)
        test_p5 = precision_at(test_preds, test_gold, 5)
        rows.append({
            "task_id": record.get("task_id", ""),
            "repo": record.get("repo", task.get("repo", "")),
            "language": record.get("language", task.get("language", "")),
            "task_category": record.get("task_category", task.get("task_category", "")),
            "model": record.get("model", ""),
            "condition": record.get("condition", ""),
            "returncode": record.get("returncode", ""),
            "elapsed_sec": record.get("elapsed_sec", ""),
            "parse_ok": int(parsed is not None),
            "code_recall_at_5": code_r5,
            "code_recall_at_10": recall_at(code_preds, code_gold, 10),
            "code_precision_at_5": code_p5,
            "code_f1_at_5": f1(code_p5, code_r5),
            "test_recall_at_5": test_r5,
            "test_recall_at_10": recall_at(test_preds, test_gold, 10),
            "test_precision_at_5": test_p5,
            "test_f1_at_5": f1(test_p5, test_r5),
            "all_recall_at_10": recall_at(all_preds, all_gold, 10),
            "all_precision_at_10": precision_at(all_preds, all_gold, 10),
            "basename_all_recall_at_10": recall_at(all_preds, all_gold, 10, basename=True),
            "predicted_code_files": len(code_preds),
            "predicted_test_files": len(test_preds),
            "gold_code_files": len(code_gold),
            "gold_test_files": len(test_gold),
            "candidate_path_count": task.get("candidate_path_count", ""),
        })

    out_dir = Path(args.out_dir)
    stem = Path(args.outputs_jsonl).stem
    metric_path = out_dir / f"{stem}_metrics.csv"
    write_csv(metric_path, rows)
    summaries = {
        "summary_by_model_condition.csv": aggregate(rows, ["model", "condition"]),
        "summary_by_language.csv": aggregate(rows, ["model", "condition", "language"]),
        "summary_by_task_category.csv": aggregate(rows, ["model", "condition", "task_category"]),
    }
    for name, summary_rows in summaries.items():
        write_csv(out_dir / name, summary_rows)

    md_path = out_dir / "TRACE_EXPERIMENT_SUMMARY.md"
    top = summaries["summary_by_model_condition.csv"]
    with md_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# SWE-PolyBench Traceability Experiment Summary\n\n")
        handle.write(f"- Prediction file: `{args.outputs_jsonl}`\n")
        handle.write(f"- Metric rows: {len(rows)}\n")
        handle.write(f"- Detailed metrics: `{metric_path}`\n\n")
        handle.write("| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in top:
            handle.write(
                f"| {row.get('model', '')} | {row.get('condition', '')} | {row.get('n', '')} | "
                f"{row.get('code_recall_at_10', ''):.3f} | {row.get('test_recall_at_10', ''):.3f} | "
                f"{row.get('all_recall_at_10', ''):.3f} | {row.get('basename_all_recall_at_10', ''):.3f} | "
                f"{row.get('parse_ok', ''):.3f} |\n"
            )
    print(f"wrote {len(rows)} metric rows to {metric_path}")
    print(f"wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()

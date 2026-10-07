"""Retain repository-level license notices for the archived source excerpts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from repository_benchmark import ROOT, load_rows, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    dest = args.results / "attribution"
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for task in load_rows(ROOT / "data/tasks.jsonl"):
        source = args.cache / "corpus" / (task["instance_id"] + ".json.gz")
        with gzip.open(source, "rt", encoding="utf-8") as f:
            corpus = json.load(f)
        notices = []
        for file in corpus["files"]:
            path = Path(file["path"])
            if len(path.parts) <= 2 and re.match(
                    r"^(LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)([._-].*)?$", path.name, re.I):
                content = file["text"]
                if not content:
                    continue
                digest = hashlib.sha256(content.encode()).hexdigest()
                (dest / (digest + ".txt")).write_text(content)
                notices.append({"source_path": file["path"], "text_sha256": digest})
        records.append({"task_id": task["instance_id"], "repo": task["repo"],
                        "base_commit": task["base_commit"], "notices": notices})
    save_json(dest / "manifest.json", records)
    missing = [r["task_id"] for r in records if not r["notices"]]
    if missing:
        raise SystemExit("No repository-level notice found for: " + ", ".join(missing))
    print(f"Retained notices for {len(records)} snapshots.")


if __name__ == "__main__":
    main()

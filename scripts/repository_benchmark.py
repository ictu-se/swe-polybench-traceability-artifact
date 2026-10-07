"""Rebuild file retrieval from immutable pre-change repository snapshots.

Candidate construction sees repository, base commit and issue text only. Patches
are consumed separately, after rankings have been saved, by the analysis script.
Downloaded archives and embedding caches are disposable, not release artifacts.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tarfile
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*|[0-9]+")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def tokens(text):
    return TOKEN.findall(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower())


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def snapshot(task, cache):
    """Never use patch paths to construct or filter the repository universe."""
    name = task["instance_id"]
    dest = cache / "corpus" / (name + ".json.gz")
    if dest.exists():
        return name
    url = f"https://codeload.github.com/{task['repo']}/tar.gz/{task['base_commit']}"
    for attempt in range(4):
        try:
            res = requests.get(url, timeout=(30, 240))
            res.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    files = []
    with tarfile.open(fileobj=io.BytesIO(res.content), mode="r:gz") as archive:
        for member in archive:
            if not (member.isfile() or member.issym()):
                continue
            path = member.name.split("/", 1)[-1]
            if not path or path.startswith("/") or ".." in Path(path).parts:
                continue
            body = b""
            if member.isfile():
                with archive.extractfile(member) as handle:
                    body = handle.read(1_000_001)
            binary = b"\x00" in body[:8192]
            text = "" if binary else body[:1_000_000].decode("utf-8", errors="replace")
            files.append({"path": path, "text": text, "size": member.size,
                          "binary": binary, "truncated": len(body) > 1_000_000,
                          "symlink": member.issym()})
    files.sort(key=lambda f: f["path"])
    if not files:
        raise ValueError(f"Empty snapshot: {name}")
    data = {"task_id": name, "repo": task["repo"], "base_commit": task["base_commit"],
            "archive_url": url, "archive_sha256": hashlib.sha256(res.content).hexdigest(),
            "files": files}
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    temp.replace(dest)
    return name


def bm25(query, documents):
    """Robertson BM25 with nonnegative IDF; full text up to the declared byte cap."""
    q = set(tokens(query))
    lengths, counts = [], []
    df = Counter()
    for doc in documents:
        words = tokens(doc)
        lengths.append(len(words))
        c = Counter(w for w in words if w in q)
        counts.append(c)
        df.update(c.keys())
    average = max(float(np.mean(lengths)), 1.0)
    n = len(documents)
    scores = np.zeros(n)
    for i, c in enumerate(counts):
        norm = 1.2 * (1 - .75 + .75 * lengths[i] / average)
        scores[i] = sum(math.log(1 + (n - df[w] + .5) / (df[w] + .5)) *
                        f * 2.2 / (f + norm) for w, f in c.items())
    return scores


def ranking(scores, paths):
    return sorted(range(len(paths)), key=lambda i: (-float(scores[i]), paths[i]))


def embedding_text(file):
    # Same query-independent representation for every file, including non-gold files.
    # The encoder truncates this at 256 wordpieces; the limit is recorded explicitly.
    return file["path"] + "\n" + file["text"][:16000]


def vectors(texts, model, database):
    keys = [hashlib.sha256(s.encode()).hexdigest() for s in texts]
    found = {}
    missing = {}
    for key, value in zip(keys, texts):
        row = database.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
        if row:
            found[key] = np.frombuffer(row[0], dtype=np.float32)
        else:
            missing[key] = value
    if missing:
        todo = list(missing)
        encoded = model.encode([missing[k] for k in todo], batch_size=64,
                               normalize_embeddings=True, show_progress_bar=False)
        for key, value in zip(todo, encoded):
            value = np.asarray(value, dtype=np.float32)
            found[key] = value
            database.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (key, value.tobytes()))
        database.commit()
    return np.stack([found[k] for k in keys])


def excerpt(text, query, budget=600):
    """Select lines by query overlap, not by patch or target-file membership."""
    lines = text.splitlines()
    if not lines:
        return ""
    q = set(tokens(query))
    anchors = sorted(range(len(lines)), key=lambda i: (-len(q & set(tokens(lines[i]))), i))[:4]
    selected = sorted({j for i in anchors for j in range(max(0, i-1), min(len(lines), i+2))})
    return "\n".join(f"{i+1}: {lines[i]}" for i in selected)[:budget]


def retrieve(task, cache, output, model, database):
    tid = task["instance_id"]
    dest = output / "retrieval" / (tid + ".json")
    if dest.exists():
        return
    with gzip.open(cache / "corpus" / (tid + ".json.gz"), "rt", encoding="utf-8") as handle:
        corpus = json.load(handle)
    files = corpus.pop("files")
    paths = [f["path"] for f in files]
    query = task["problem_statement"]
    start = time.monotonic()
    path_scores = bm25(query, paths)
    path_order = ranking(path_scores, paths)
    path_seconds = time.monotonic() - start
    start = time.monotonic()
    content_scores = bm25(query, [f["path"] + "\n" + f["text"] for f in files])
    content_order = ranking(content_scores, paths)
    content_seconds = time.monotonic() - start
    start = time.monotonic()
    embs = vectors([embedding_text(f) for f in files], model, database)
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    dense_order = ranking(embs @ query_emb, paths)
    dense_seconds = time.monotonic() - start
    fused = np.zeros(len(paths))
    for order in (content_order, dense_order):
        for rank, idx in enumerate(order, 1):
            fused[idx] += 1 / (60 + rank)
    hybrid_order = ranking(fused, paths)
    # The prompt selector is independent of all gold targets and has a fixed budget.
    candidates = [{"id": n+1, "path": paths[i], "excerpt": excerpt(files[i]["text"], query)}
                  for n, i in enumerate(hybrid_order[:40])]
    record = {**corpus, "language": task["language"], "file_count": len(paths),
              "binary_count": sum(f["binary"] for f in files),
              "truncated_count": sum(f["truncated"] for f in files),
              "universe": paths, "candidates": candidates,
              "rankings": {"path_bm25": [paths[i] for i in path_order[:100]],
                           "content_bm25": [paths[i] for i in content_order[:100]],
                           "dense_minilm": [paths[i] for i in dense_order[:100]],
                           "hybrid_rrf": [paths[i] for i in hybrid_order[:100]]},
              "retrieval_seconds": {"path_bm25": path_seconds, "content_bm25": content_seconds,
                                    "dense_minilm": dense_seconds},
              "embedding_model": EMBEDDING_MODEL, "embedding_max_tokens": model.max_seq_length}
    save_json(dest, record)
    print(f"retrieved {tid}: {len(paths)} files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.jsonl")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = load_rows(args.tasks)
    # Project a strict allowlist before any retrieval function receives a task.
    tasks = [{k: t[k] for k in ("instance_id", "repo", "base_commit", "problem_statement", "language")}
             for t in tasks]
    if not args.retrieve_only:
        errors = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(snapshot, t, args.cache): t["instance_id"] for t in tasks}
            for future in as_completed(futures):
                try:
                    print("snapshot " + future.result(), flush=True)
                except Exception as exc:
                    errors[futures[future]] = str(exc)
                    print("DOWNLOAD ERROR", futures[future], str(exc), flush=True)
        save_json(args.output / "download_errors.json", errors)
        if errors:
            raise SystemExit("Snapshots incomplete; rerun to resume. No silent task removal.")
    if args.download_only:
        return
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)
    model.max_seq_length = 256
    database = sqlite3.connect(args.cache / "minilm_embeddings.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)")
    pending = list(tasks)
    while pending:
        ready = [t for t in pending if (args.cache / "corpus" / (t["instance_id"] + ".json.gz")).exists()]
        for task in ready:
            retrieve(task, args.cache, args.output, model, database)
            pending.remove(task)
        if pending and not ready:
            print(f"Waiting for {len(pending)} snapshots", flush=True)
            time.sleep(15)
    database.close()


if __name__ == "__main__":
    main()

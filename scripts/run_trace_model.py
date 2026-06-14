import argparse
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "You are a software traceability researcher. "
    "Return strict JSON only. Do not include markdown."
)


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_terminal_text(text):
    text = text or ""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def run_ollama_api(model, prompt, timeout, host):
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
    if model.startswith("qwen3"):
        full_prompt = "/no_think\n" + full_prompt
    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 900,
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {
        "stdout": clean_terminal_text(body.get("response", "")),
        "stderr": "",
        "returncode": 0,
        "elapsed_sec": round(time.time() - started, 3),
        "eval_count": body.get("eval_count", ""),
        "prompt_eval_count": body.get("prompt_eval_count", ""),
    }


def run_ollama_cli(model, prompt, timeout):
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
    if model.startswith("qwen3"):
        full_prompt = "/no_think\n" + full_prompt
    started = time.time()
    proc = subprocess.run(
        ["ollama", "run", model],
        input=full_prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "stdout": clean_terminal_text(proc.stdout),
        "stderr": clean_terminal_text(proc.stderr),
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "eval_count": "",
        "prompt_eval_count": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", default=str(PROJECT / "prompts/trace_prompt_packs.jsonl"))
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--conditions", default="", help="Comma-separated condition filter.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--fallback-cli", action="store_true")
    args = parser.parse_args()

    packs = load_jsonl(args.packs)
    if args.conditions:
        keep = {item.strip() for item in args.conditions.split(",") if item.strip()}
        packs = [row for row in packs if row["condition"] in keep]
    if args.max_tasks:
        seen = []
        selected = []
        for row in packs:
            if row["task_id"] not in seen:
                seen.append(row["task_id"])
            if len(seen) <= args.max_tasks:
                selected.append(row)
        packs = selected

    safe_model = args.model.replace(":", "_").replace("/", "_")
    out_path = Path(args.out) if args.out else PROJECT / "results" / f"trace_outputs_{safe_model}.jsonl"
    completed = set()
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            completed.add((row["task_id"], row["condition"], row.get("model", "")))

    for index, pack in enumerate(packs, start=1):
        key = (pack["task_id"], pack["condition"], args.model)
        if key in completed:
            continue
        percent = index * 100 / len(packs)
        print(f"[{index}/{len(packs)} | {percent:6.2f}%] {args.model} {pack['condition']} {pack['task_id']}", flush=True)
        try:
            try:
                result = run_ollama_api(args.model, pack["prompt"], args.timeout, args.ollama_host)
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as api_error:
                if not args.fallback_cli:
                    result = {
                        "stdout": "",
                        "stderr": f"api_error: {api_error}",
                        "returncode": -1,
                        "elapsed_sec": args.timeout,
                        "eval_count": "",
                        "prompt_eval_count": "",
                    }
                else:
                    result = run_ollama_cli(args.model, pack["prompt"], args.timeout)
                    result["stderr"] = (result.get("stderr", "") + f"\nAPI fallback reason: {api_error}").strip()
        except subprocess.TimeoutExpired as exc:
            result = {
                "stdout": clean_terminal_text(exc.stdout or ""),
                "stderr": f"timeout after {args.timeout}s",
                "returncode": -1,
                "elapsed_sec": args.timeout,
                "eval_count": "",
                "prompt_eval_count": "",
            }
        append_jsonl(out_path, {
            "task_id": pack["task_id"],
            "repo": pack["repo"],
            "language": pack["language"],
            "task_category": pack["task_category"],
            "condition": pack["condition"],
            "model": args.model,
            **result,
        })
    print(f"Wrote outputs to {out_path}")


if __name__ == "__main__":
    main()

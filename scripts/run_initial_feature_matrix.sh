#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/prepare_traceability_benchmark.py --category Feature --candidate-limit 80
python3 scripts/run_trace_baseline.py
python3 scripts/score_trace_predictions.py results/lexical_path_ranker_outputs.jsonl

MODELS=(
  "qwen2.5-coder:1.5b"
  "qwen2.5-coder:3b"
  "llama3.2:3b"
  "gemma3:4b"
  "deepseek-coder:6.7b"
  "qwen2.5-coder:7b"
)

OUTPUTS=("results/lexical_path_ranker_outputs.jsonl")

for model in "${MODELS[@]}"; do
  safe="${model//[:\/]/_}"
  out="results/trace_outputs_${safe}.jsonl"
  python3 scripts/run_trace_model.py \
    --model "$model" \
    --conditions issue_only,issue_plus_inventory,issue_plus_hints \
    --resume \
    --timeout 300 \
    --out "$out"
  python3 scripts/score_trace_predictions.py "$out" --out-dir "results/${safe}"
  OUTPUTS+=("$out")
  cat "${OUTPUTS[@]}" > results/trace_outputs_feature_matrix_all.jsonl
  python3 scripts/score_trace_predictions.py results/trace_outputs_feature_matrix_all.jsonl --out-dir results/feature_matrix_combined
done

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/prepare_traceability_benchmark.py --category Feature --candidate-limit 80

MODELS=(
  "phi3:mini"
  "qwen2.5:3b"
  "qwen3:4b"
  "qwen2.5-coder:14b"
  "qwen2.5-coder:32b"
)

BASE_OUTPUTS=(
  "results/lexical_path_ranker_outputs.jsonl"
  "results/trace_outputs_qwen2.5-coder_1.5b.jsonl"
  "results/trace_outputs_qwen2.5-coder_3b.jsonl"
  "results/trace_outputs_llama3.2_3b.jsonl"
  "results/trace_outputs_gemma3_4b.jsonl"
  "results/trace_outputs_deepseek-coder_6.7b.jsonl"
  "results/trace_outputs_qwen2.5-coder_7b.jsonl"
)

NEW_OUTPUTS=()

for model in "${MODELS[@]}"; do
  safe="${model//[:\/]/_}"
  out="results/trace_outputs_${safe}.jsonl"
  python3 scripts/run_trace_model.py \
    --model "$model" \
    --conditions issue_only,issue_plus_inventory,issue_plus_hints \
    --resume \
    --timeout 480 \
    --out "$out"
  python3 scripts/score_trace_predictions.py "$out" --out-dir "results/${safe}"
  NEW_OUTPUTS+=("$out")
  existing=()
  for f in "${BASE_OUTPUTS[@]}" "${NEW_OUTPUTS[@]}"; do
    if [[ -s "$f" ]]; then
      existing+=("$f")
    fi
  done
  cat "${existing[@]}" > results/trace_outputs_feature_matrix_all_extended.jsonl
  python3 scripts/score_trace_predictions.py \
    results/trace_outputs_feature_matrix_all_extended.jsonl \
    --out-dir results/feature_matrix_extended
done

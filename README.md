# SWE-PolyBench Issue-Code-Test Traceability Artifact

Replication package for the study:

> Can Local Code Models Recover Issue-to-Code-and-Test Trace Links? Evidence from a SWE-PolyBench Feature Study

This repository contains only the code, generated task data, prompt packs, raw model outputs, metric files, and summary tables needed to reproduce the analysis. It intentionally excludes manuscript source files, compiled PDFs, submission material, local terminal logs, caches, and workspace notes.

## Contents

- `scripts/prepare_traceability_benchmark.py`: builds traceability tasks and prompt packs from SWE-PolyBench 500.
- `scripts/run_trace_baseline.py`: runs the lexical path-ranker baseline.
- `scripts/run_trace_model.py`: runs local Ollama models with resumable JSONL output.
- `scripts/score_trace_predictions.py`: parses outputs and computes code/test/all recall, precision, basename recall, parse rate, and grouped summaries.
- `scripts/run_initial_feature_matrix.sh`: original six-model run script.
- `scripts/run_extended_feature_models.sh`: extended model run script.
- `data/`: generated Feature-subset task files and gold file traces.
- `prompts/trace_prompt_packs.jsonl`: archived prompts for all 184 tasks and three conditions.
- `results/trace_outputs_feature_matrix_all_extended.jsonl`: combined raw outputs for the completed matrix.
- `results/feature_matrix_extended/`: metric rows and summary tables used in the manuscript.
- `results/*/`: per-model metric summaries for the completed local model runs.

## Environment

Python 3.9+ is sufficient to reproduce the analysis from stored outputs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The analysis scripts use only the Python standard library. Ollama is needed only for optional local model reruns.

## Quick Reproduction Without Running LLMs

The archived raw outputs are included, so the main metric tables can be regenerated without downloading models:

```bash
python3 scripts/score_trace_predictions.py \
  results/trace_outputs_feature_matrix_all_extended.jsonl \
  --out-dir reproduced_results/feature_matrix_extended
```

The regenerated key files are:

- `reproduced_results/feature_matrix_extended/trace_outputs_feature_matrix_all_extended_metrics.csv`
- `reproduced_results/feature_matrix_extended/summary_by_model_condition.csv`
- `reproduced_results/feature_matrix_extended/summary_by_language.csv`
- `reproduced_results/feature_matrix_extended/summary_by_task_category.csv`
- `reproduced_results/feature_matrix_extended/TRACE_EXPERIMENT_SUMMARY.md`

## Optional Full Rerun With Local Models

The reported LLM runs used local Ollama models. Install Ollama and pull the models you want to rerun, for example:

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen2.5-coder:32b
ollama pull deepseek-coder:6.7b
ollama pull gemma3:4b
ollama pull llama3.2:3b
ollama pull phi3:mini
ollama pull qwen2.5:3b
ollama pull qwen3:4b
```

Then run:

```bash
bash scripts/run_extended_feature_models.sh
```

The runner resumes partially completed JSONL output files and recomputes summaries after each model.

## Regenerating Task and Prompt Files

The generated `data/` and `prompts/` files are included. To rebuild them from the original SWE-PolyBench 500 JSONL, provide the dataset path explicitly if it is not in the same workspace layout as the original run:

```bash
python3 scripts/prepare_traceability_benchmark.py \
  --source /path/to/SWE-PolyBench_500/test.jsonl \
  --category Feature \
  --candidate-limit 80
```

## Reported Matrix

The completed matrix contains:

- 184 SWE-PolyBench Feature instances.
- 18 repositories.
- 44 Java, 50 JavaScript, 38 Python, and 52 TypeScript tasks.
- Three prompt conditions: `issue_only`, `issue_plus_inventory`, and `issue_plus_hints`.
- A lexical path-ranker baseline and eleven local model runs.
- 6,624 scored prediction rows.

## Notes on Data

Gold traces are derived from changed files in SWE-PolyBench reference patches and test patches. The package stores generated trace tasks, prompt packs, and model outputs, but it does not vendor the full SWE-PolyBench repository checkouts. Reproducing the archived analysis from stored outputs does not require repository clones.

## License

Code and documentation in this artifact are released under the MIT License. SWE-PolyBench-derived task data and any third-party path metadata remain subject to their original dataset and repository licenses.

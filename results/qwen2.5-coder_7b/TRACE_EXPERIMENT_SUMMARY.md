# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_qwen2.5-coder_7b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/qwen2.5-coder_7b/trace_outputs_qwen2.5-coder_7b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-coder:7b | issue_only | 184 | 0.150 | 0.039 | 0.099 | 0.166 | 1.000 |
| qwen2.5-coder:7b | issue_plus_hints | 184 | 0.413 | 0.485 | 0.417 | 0.459 | 1.000 |
| qwen2.5-coder:7b | issue_plus_inventory | 184 | 0.392 | 0.475 | 0.404 | 0.453 | 1.000 |

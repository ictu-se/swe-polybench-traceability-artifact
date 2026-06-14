# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_qwen2.5-coder_14b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/qwen2.5-coder_14b/trace_outputs_qwen2.5-coder_14b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-coder:14b | issue_only | 184 | 0.187 | 0.057 | 0.125 | 0.229 | 1.000 |
| qwen2.5-coder:14b | issue_plus_hints | 184 | 0.471 | 0.547 | 0.467 | 0.543 | 1.000 |
| qwen2.5-coder:14b | issue_plus_inventory | 184 | 0.473 | 0.549 | 0.469 | 0.544 | 1.000 |

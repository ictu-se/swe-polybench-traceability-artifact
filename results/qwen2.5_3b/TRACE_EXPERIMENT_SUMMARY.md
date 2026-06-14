# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_qwen2.5_3b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/qwen2.5_3b/trace_outputs_qwen2.5_3b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:3b | issue_only | 184 | 0.093 | 0.008 | 0.056 | 0.130 | 1.000 |
| qwen2.5:3b | issue_plus_hints | 184 | 0.339 | 0.413 | 0.362 | 0.431 | 1.000 |
| qwen2.5:3b | issue_plus_inventory | 184 | 0.333 | 0.416 | 0.369 | 0.443 | 1.000 |

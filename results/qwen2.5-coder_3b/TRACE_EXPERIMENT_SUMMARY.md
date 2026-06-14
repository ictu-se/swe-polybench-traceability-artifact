# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_qwen2.5-coder_3b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/qwen2.5-coder_3b/trace_outputs_qwen2.5-coder_3b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-coder:3b | issue_only | 184 | 0.122 | 0.008 | 0.071 | 0.137 | 1.000 |
| qwen2.5-coder:3b | issue_plus_hints | 184 | 0.334 | 0.429 | 0.363 | 0.430 | 1.000 |
| qwen2.5-coder:3b | issue_plus_inventory | 184 | 0.320 | 0.406 | 0.337 | 0.399 | 1.000 |

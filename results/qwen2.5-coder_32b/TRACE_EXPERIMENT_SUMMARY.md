# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_qwen2.5-coder_32b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/qwen2.5-coder_32b/trace_outputs_qwen2.5-coder_32b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-coder:32b | issue_only | 184 | 0.202 | 0.061 | 0.133 | 0.224 | 1.000 |
| qwen2.5-coder:32b | issue_plus_hints | 184 | 0.472 | 0.565 | 0.477 | 0.528 | 1.000 |
| qwen2.5-coder:32b | issue_plus_inventory | 184 | 0.464 | 0.568 | 0.470 | 0.522 | 1.000 |

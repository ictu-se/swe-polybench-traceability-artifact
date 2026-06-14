# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_qwen2.5-coder_1.5b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/qwen2.5-coder_1.5b/trace_outputs_qwen2.5-coder_1.5b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-coder:1.5b | issue_only | 184 | 0.108 | 0.012 | 0.066 | 0.136 | 1.000 |
| qwen2.5-coder:1.5b | issue_plus_hints | 184 | 0.300 | 0.241 | 0.328 | 0.382 | 1.000 |
| qwen2.5-coder:1.5b | issue_plus_inventory | 184 | 0.305 | 0.220 | 0.326 | 0.395 | 1.000 |

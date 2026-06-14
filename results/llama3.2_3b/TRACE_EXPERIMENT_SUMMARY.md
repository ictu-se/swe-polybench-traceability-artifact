# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_llama3.2_3b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/llama3.2_3b/trace_outputs_llama3.2_3b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama3.2:3b | issue_only | 184 | 0.107 | 0.019 | 0.068 | 0.142 | 1.000 |
| llama3.2:3b | issue_plus_hints | 184 | 0.361 | 0.353 | 0.333 | 0.387 | 1.000 |
| llama3.2:3b | issue_plus_inventory | 184 | 0.369 | 0.350 | 0.338 | 0.409 | 1.000 |

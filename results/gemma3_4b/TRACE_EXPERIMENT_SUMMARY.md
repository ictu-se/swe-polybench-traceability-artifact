# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_gemma3_4b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/gemma3_4b/trace_outputs_gemma3_4b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma3:4b | issue_only | 184 | 0.084 | 0.010 | 0.052 | 0.110 | 1.000 |
| gemma3:4b | issue_plus_hints | 184 | 0.332 | 0.443 | 0.359 | 0.403 | 1.000 |
| gemma3:4b | issue_plus_inventory | 184 | 0.354 | 0.432 | 0.379 | 0.431 | 1.000 |

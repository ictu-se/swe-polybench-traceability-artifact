# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_deepseek-coder_6.7b.jsonl`
- Metric rows: 552
- Detailed metrics: `results/deepseek-coder_6.7b/trace_outputs_deepseek-coder_6.7b_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-coder:6.7b | issue_only | 184 | 0.091 | 0.005 | 0.054 | 0.099 | 1.000 |
| deepseek-coder:6.7b | issue_plus_hints | 184 | 0.321 | 0.267 | 0.291 | 0.331 | 1.000 |
| deepseek-coder:6.7b | issue_plus_inventory | 184 | 0.318 | 0.308 | 0.296 | 0.336 | 1.000 |

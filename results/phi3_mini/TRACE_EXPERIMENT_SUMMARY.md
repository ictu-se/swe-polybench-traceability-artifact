# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_phi3_mini.jsonl`
- Metric rows: 552
- Detailed metrics: `results/phi3_mini/trace_outputs_phi3_mini_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| phi3:mini | issue_only | 184 | 0.023 | 0.003 | 0.017 | 0.065 | 0.995 |
| phi3:mini | issue_plus_hints | 184 | 0.196 | 0.076 | 0.163 | 0.213 | 1.000 |
| phi3:mini | issue_plus_inventory | 184 | 0.204 | 0.110 | 0.175 | 0.227 | 1.000 |

# SWE-PolyBench Traceability Experiment Summary

- Prediction file: `results/trace_outputs_feature_matrix_all_extended.jsonl`
- Metric rows: 6624
- Detailed metrics: `results/feature_matrix_extended/trace_outputs_feature_matrix_all_extended_metrics.csv`

| Model | Condition | n | Code R@10 | Test R@10 | All R@10 | Basename All R@10 | Parse OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-coder:6.7b | issue_only | 184 | 0.091 | 0.005 | 0.054 | 0.099 | 1.000 |
| deepseek-coder:6.7b | issue_plus_hints | 184 | 0.321 | 0.267 | 0.291 | 0.331 | 1.000 |
| deepseek-coder:6.7b | issue_plus_inventory | 184 | 0.318 | 0.308 | 0.296 | 0.336 | 1.000 |
| gemma3:4b | issue_only | 184 | 0.084 | 0.010 | 0.052 | 0.110 | 1.000 |
| gemma3:4b | issue_plus_hints | 184 | 0.332 | 0.443 | 0.359 | 0.403 | 1.000 |
| gemma3:4b | issue_plus_inventory | 184 | 0.354 | 0.432 | 0.379 | 0.431 | 1.000 |
| lexical_path_ranker | issue_only | 184 | 0.563 | 0.617 | 0.388 | 0.410 | 1.000 |
| lexical_path_ranker | issue_plus_hints | 184 | 0.563 | 0.617 | 0.388 | 0.410 | 1.000 |
| lexical_path_ranker | issue_plus_inventory | 184 | 0.563 | 0.617 | 0.388 | 0.410 | 1.000 |
| llama3.2:3b | issue_only | 184 | 0.107 | 0.019 | 0.068 | 0.142 | 1.000 |
| llama3.2:3b | issue_plus_hints | 184 | 0.361 | 0.353 | 0.333 | 0.387 | 1.000 |
| llama3.2:3b | issue_plus_inventory | 184 | 0.369 | 0.350 | 0.338 | 0.409 | 1.000 |
| phi3:mini | issue_only | 184 | 0.023 | 0.003 | 0.017 | 0.065 | 0.995 |
| phi3:mini | issue_plus_hints | 184 | 0.196 | 0.076 | 0.163 | 0.213 | 1.000 |
| phi3:mini | issue_plus_inventory | 184 | 0.204 | 0.110 | 0.175 | 0.227 | 1.000 |
| qwen2.5-coder:1.5b | issue_only | 184 | 0.108 | 0.012 | 0.066 | 0.136 | 1.000 |
| qwen2.5-coder:1.5b | issue_plus_hints | 184 | 0.300 | 0.241 | 0.328 | 0.382 | 1.000 |
| qwen2.5-coder:1.5b | issue_plus_inventory | 184 | 0.305 | 0.220 | 0.326 | 0.395 | 1.000 |
| qwen2.5-coder:14b | issue_only | 184 | 0.187 | 0.057 | 0.125 | 0.229 | 1.000 |
| qwen2.5-coder:14b | issue_plus_hints | 184 | 0.471 | 0.547 | 0.467 | 0.543 | 1.000 |
| qwen2.5-coder:14b | issue_plus_inventory | 184 | 0.473 | 0.549 | 0.469 | 0.544 | 1.000 |
| qwen2.5-coder:32b | issue_only | 184 | 0.202 | 0.061 | 0.133 | 0.224 | 1.000 |
| qwen2.5-coder:32b | issue_plus_hints | 184 | 0.472 | 0.565 | 0.477 | 0.528 | 1.000 |
| qwen2.5-coder:32b | issue_plus_inventory | 184 | 0.464 | 0.568 | 0.470 | 0.522 | 1.000 |
| qwen2.5-coder:3b | issue_only | 184 | 0.122 | 0.008 | 0.071 | 0.137 | 1.000 |
| qwen2.5-coder:3b | issue_plus_hints | 184 | 0.334 | 0.429 | 0.363 | 0.430 | 1.000 |
| qwen2.5-coder:3b | issue_plus_inventory | 184 | 0.320 | 0.406 | 0.337 | 0.399 | 1.000 |
| qwen2.5-coder:7b | issue_only | 184 | 0.150 | 0.039 | 0.099 | 0.166 | 1.000 |
| qwen2.5-coder:7b | issue_plus_hints | 184 | 0.413 | 0.485 | 0.417 | 0.459 | 1.000 |
| qwen2.5-coder:7b | issue_plus_inventory | 184 | 0.392 | 0.475 | 0.404 | 0.453 | 1.000 |
| qwen2.5:3b | issue_only | 184 | 0.093 | 0.008 | 0.056 | 0.130 | 1.000 |
| qwen2.5:3b | issue_plus_hints | 184 | 0.339 | 0.413 | 0.362 | 0.431 | 1.000 |
| qwen2.5:3b | issue_plus_inventory | 184 | 0.333 | 0.416 | 0.369 | 0.443 | 1.000 |
| qwen3:4b | issue_only | 184 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3:4b | issue_plus_hints | 184 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3:4b | issue_plus_inventory | 184 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

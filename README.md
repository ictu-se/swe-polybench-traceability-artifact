# Repository-grounded issue-to-code-and-test retrieval

This package reconstructs the pre-change repositories of 184 SWE-PolyBench Feature
tasks and compares retrieval and local-model reranking under one exact-path budget.
The previous patch-derived candidate experiment is historical evidence only; it
does not measure retrieval from a pre-change repository. Its original implementation
is preserved in Git history at commit `83014bf75660231c3225524d82ffe35e097dd6a2`.

## Recorded run

The 7 October 2026 run contains 184 repository retrieval records and 1,016 model
outputs (508 per family). All 736 primary-seed responses satisfy the ranking
protocol. Two of the 280 additional-seed responses contain duplicate identifiers;
they are retained and scored as empty rankings. The artifact audit verifies the
complete matrix, exact prompts/settings, candidate membership, dataset checksum,
and model-manifest identity. Nine evaluation tests cover the scoring contract.

Primary R@10 is 0.157 (path BM25), 0.174 (content BM25), 0.175 (dense), and 0.267
(hybrid). Qwen3-Coder scores 0.404/0.390 and Devstral scores 0.401/0.382 for
paths/content prompts. Both excerpt-contrast repository-bootstrap intervals
include zero; these comparisons do not establish equivalence. Both prompt
conditions already use a content-informed hybrid shortlist.

## Installation

Use Python 3.9 or later and the pinned dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/test_evaluation.py
```

On systems without Apple Metal, embedding inference automatically uses the CPU.
The hardware and exact embedding revision used for the reported run are recorded
under `results/environment/`. Model generation additionally requires Ollama and
the two model artifacts identified there.

## Reproduce tables and figures from archived outputs

```sh
python scripts/audit_artifact.py
python scripts/analyze.py
python scripts/plot_results.py
```

The analysis exports task metrics, target availability, candidate coverage, model
summaries, language summaries, paired bootstrap contrasts, and repeat stability
to `results/analysis/`. Figure generation reads those tables and writes four
empirical figures under `results/figures/`. No model calls or repository downloads
are needed for this route. CSV tables contain the values used in the article.

## Reconstruct the repository experiment

```sh
python scripts/repository_benchmark.py --cache .cache --workers 4
python scripts/collect_attribution.py --cache .cache
```

This downloads immutable repository archives at each task's `base_commit`, indexes
every regular-file and symbolic-link path, and produces the four baseline rankings.
Archive checksums, full candidate universes, top-100 rankings and top-40 reranking
excerpts are preserved under `results/retrieval/`. Cached archives are converted
to compressed text corpora; caches and third-party repository trees are not release
artifacts. Existing results are resumed. To rebuild rankings, use a fresh output
directory through `--output`; to verify downloads independently, use a fresh cache.

Repository downloads can be several gigabytes. Candidate construction never reads
`patch`, `test_patch`, hints, changed-node metadata, or test execution metadata.
Those fields are used only by the later target/availability audit. Retrieval uses
the full issue text; prompts cap issues at 8,000 characters. The MiniLM encoder
has a separate 256-wordpiece truncation limit, including for its query.

## Rerun the model matrix

Install the exact artifacts recorded in `results/environment/`; tags alone are
mutable and are insufficient to reproduce the weight version. An example pull is:

```sh
ollama pull qwen3-coder:30b
ollama pull devstral-small-2:24b
```

Compare the resulting digests with the archived environment before rerunning.
The archived OCI manifests in `data/model_manifests/` list immutable model,
template and configuration blob digests. Their SHA-256 values match the captured
Ollama model digests. The runner checks this identity before making generation
requests, including when starting a fresh output directory.
Use `--output results_rerun` consistently for reconstruction and model generation
to preserve archived outputs. The runner resumes existing task/condition/seed
triples and rejects a changed digest. Then pass `--results results_rerun` to analysis
and plotting. Omitting these options operates on the archived `results` directory.

```sh
python scripts/run_models.py --model qwen3-coder:30b
python scripts/run_models.py --model devstral-small-2:24b
python scripts/audit_artifact.py
python scripts/analyze.py
python scripts/plot_results.py
```

`--host` selects a dedicated local Ollama endpoint if another experiment uses the
default service. Run the two model families sequentially to avoid weight swapping.
Seed 11 covers all 184 tasks and both paths/content conditions. Seeds 29 and 47
cover a predetermined hash-selected panel of up to two tasks per repository. The
panel is stored in `data/repeat_panel.json`. All requests, responses, model digests,
timestamps, validity outcomes and elapsed times are archived. The generated
Ollama Modelfile is omitted from environment metadata because it embeds a local
blob path; model digests, templates, parameters and architecture fields remain. Do not compare these
elapsed times as a controlled hardware benchmark: early requests shared a service,
and later requests used a dedicated local service on the same machine. Host
suspension also produced gaps between wall-clock timestamps; recorded durations
use a monotonic clock and are not an end-to-end wall-clock throughput measure.

## Evaluation contract

- A single ordered top-ten list is used for all methods. Report R/P at 1, 3, 5, 10.
- Exact relative paths are required; basenames and suffixes do not receive credit.
- Fixed-budget precision divides by k even when fewer items are returned.
- Code/test recall uses the same joint ranking; no separate-list concatenation.
- Added files are audited separately and excluded from existing-file recall.
- Renames use the old path, and deleted files remain eligible pre-change targets.
- Invalid model outputs receive an empty ranking; no best-of-seed selection.
- Empty gold sets have undefined recall and are counted explicitly.
- Bootstrap contrasts are paired by task, with repository-cluster and task variants.

The complete choices are in `protocol.json`. This is a static retrieval and reranking
experiment, not a reproduction of an end-to-end coding agent. The dense baseline
is a short-context general semantic encoder, not a trained code-search frontier model.
Pretraining exposure to the public benchmark is unknown.

## Provenance and licensing

`data/provenance.json` records the original dataset revision and checksums.
`data/tasks.jsonl` contains the archived Feature records. `data/legacy_tasks.jsonl`
and `data/legacy_patch_pools.json` preserve only the data required to reproduce the
candidate-provenance and output-budget audit of the older design.

Code is distributed under the repository's MIT license. The upstream dataset card
labels the benchmark MIT; its attribution and license are included in `NOTICE`.
Benchmark data and
third-party repository content retain their original licenses and attribution.
Retrieved excerpts are attributable through repository URL and immutable commit
in each retrieval record. Manuscript sources, manuscript PDFs, templates, review
comments, cover letters, private files, local notes, logs and caches are excluded.

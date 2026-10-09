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
python scripts/unpack_outputs.py
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

Large output streams may be stored as lossless gzip archives to stay within
repository file-size limits. `scripts/unpack_outputs.py` restores them using the
compressed and uncompressed SHA-256 values in `data/compressed_outputs.json`.
It verifies existing outputs and refuses conflicting files; no responses are
dropped or shortened. Run it once after cloning and before analysis or resumption.

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

The complete choices are in `protocol.json`. The original comparison is static retrieval and reranking. The extension below
adds controlled navigation, specialized encoders, temporal transfer and a bounded
repair probe. It does not reproduce complete published software-engineering agents.
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
comments, cover letters, private files, local notes, host/build logs and caches are excluded.


## October 2026 extensions

The frozen extension protocol adds paired tool-assisted navigation, CodeRankEmbed
controls, full-population Qwen seed repetition, and an external recent-issue cohort.
`data/extension_protocol.json` and `data/external_protocol.json` state the sample
selection, action/input limits, model revisions and interpretation boundaries.
The extensions are exploratory, designed after inspecting the original experiment.
They were fixed before their own model outcomes. Invalid outputs remain in the
score denominator and are never retried based on quality.

The feature navigation matrix contains 508 episodes: two scopes for seed 11 on all
184 tasks and seeds 29/47 on the previously selected 35-task panel. Restricted scope
exposes only the initial hybrid top 40; repository scope exposes the reconstructed
universe. Both start with the same top 10 and permit four search/read actions.
All responses and tool observations are archived. The semantic matrix contains
three rankings per task over the same hybrid top 100: MiniLM 256, CodeRankEmbed 256,
and CodeRankEmbed 2048. This isolates conditional ranking, not full-universe
CodeRankEmbed retrieval. Equal token limits use each encoder's own tokenizer.

The Qwen repeat extension contains 1,104 task/condition/seed records. Its 508 original
records are retained unchanged at the parsed-record level, with 596 additional
calls. Devstral retains its original 35-task repeat panel. Task means across seeds,
not individual responses, are the units for the extended paired contrast.

The external sample contains 24 date-verified 2026 issues from 20 additional repositories with Python as their primary language. It is not a second Feature-labeled population. Source data, public
GitHub metadata and fixed hash-order selection are pinned. The dataset's split
name is 2026_03, but actual PR creation dates span March–May 2026. Verified issue
creation dates are recorded separately. The benchmark-supplied issue body is not
an independently reconstructed creation-time snapshot. Release-relative dates
reduce one exposure risk but do not certify clean model training.

### Reproduce extension analysis without model calls

```sh
python scripts/unpack_outputs.py
python scripts/test_navigation.py
python scripts/test_repair_grading.py
python scripts/test_extension_analysis.py
python scripts/audit_extension.py
python scripts/analyze_extension.py
python scripts/plot_extension.py
```

Analysis writes separate feature/external summaries, task-level metrics, target
availability, paired task and repository-bootstrap intervals, full-seed summaries,
search coverage outside the original top 40, and repair eligibility/outcomes.
All generated figures use these empirical measurements. `--allow-partial` is for
monitoring only; partial summaries must never be used as final results.

### Reconstruct the external sample and repositories

```sh
python scripts/select_external.py --cache .cache --output .cache/external_tasks_reproduced.jsonl
python scripts/repository_benchmark.py --tasks data/external_tasks.jsonl --cache .cache_external --output results/extension/external --workers 3
python scripts/collect_attribution.py --tasks data/external_tasks.jsonl --cache .cache_external --results results/extension/external
```

The selection command downloads the immutable upstream split, verifies its hash,
and reproduces the exact archived task checksum using the metadata snapshot. It
requires no GitHub credentials. Public issue titles may later change, so live
re-querying does not replace the archived selection evidence. Dataset materials
are CC-BY 4.0; source repositories retain their own licenses and notices.

### Rerun representation and navigation controls

Install the pinned encoder, then use the appropriate reconstructed corpus cache:

```sh
python scripts/download_code_encoder.py --output .cache/coderank
python scripts/semantic_rerank.py --cache .cache --embedding-cache .cache/semantic.sqlite --code-model-dir .cache/coderank
python scripts/navigation_agent.py --cache .cache --host http://127.0.0.1:11435
python scripts/run_models.py --model qwen3-coder:30b --repeat-all --retrieval results/retrieval --output results/extension/repeats --host http://127.0.0.1:11435
```

CodeRankEmbed uses its pinned Python model implementation (`trust_remote_code`),
whose source/configuration hashes are archived with weight hashes. Embeddings run
in FP32, batches of four, on Apple Metal when available or CPU otherwise. Run one
semantic process per embedding-cache database. Downloaded model weights and
corpus/embedding caches are excluded from this repository.

For the external cohort pass `--tasks data/external_tasks.jsonl`,
`--retrieval results/extension/external/retrieval`, and the external output path to
semantic/navigation runners. For navigation additionally pass `--seeds 11` and use
`.cache_external`. For each static model, use `--tasks data/external_tasks.jsonl
--seeds 11 --output results/extension/external`. Run model families sequentially to
avoid swapping weights. The navigation default endpoint is a dedicated local
Ollama service on 11435; start it with `OLLAMA_HOST=127.0.0.1:11435 ollama serve`.

Existing records are resumed. Preserve published evidence by copying the package
into a separate experiment directory or by using fresh `--output` paths. The
full-repeat artifact audit additionally verifies retention of the 508 original
records; a completely fresh rerun is a new execution and will have new timestamps.

### Executable repair probe

Docker with Linux AMD64 emulation is required on ARM hosts. The twelve task IDs
are fixed in the external protocol. Each image is pulled and its digest recorded.
Base and gold controls run before any generated patch is scored:

```sh
python scripts/repair_probe.py preflight --logs .cache/repair_logs
python scripts/audit_repair_inputs.py
python scripts/repair_probe.py agent --logs .cache/repair_logs --host http://127.0.0.1:11435
```

The commands above resume archived records. To independently rerun reference
controls using the exact archived image digests and full test-ID mapping, write
to a fresh directory:

```sh
python scripts/replay_repair_controls.py --output .cache/control_replay
```

For a fresh repair-agent execution, use a separate copy of this package and move
its `results/extension/repair/*_agent.json` records out of that directory before
running agent mode. Retain the preflight records: agent mode uses their immutable
image digests and frozen test-ID mappings. Do not delete or alter the published
control evidence. Fresh control collection through `repair_probe.py preflight`
resolves the upstream task image tags at execution time; the replay command above
is the route for checking the exact archived images.

The archived input audit covers all twelve pinned images before repair inference:
base commit, working-tree status, refs/reflogs, unreachable Git objects, known
reference-patch files, network mode and host bind mounts. It is a specific input
state check, not an exhaustive proof about every file inside an image.

Only script-owned disposable containers are removed. Containers have no network,
no host mounts, two CPUs, 4 GiB memory, dropped capabilities and bounded execution
time. Checksummed reference-control and generated-patch verification output is archived
with structured test statuses, permitting independent replay of the parser. Host
and build logs remain excluded. Reference patches and evaluator test-selection metadata stay outside model
prompts; the issue and tests already present in the base repository remain
available to the agent. Verification occurs in a new
container after restoring reference test paths. Every selected task remains in
the report; failed environment controls are separated from model failures, with
no replacement. Test success is a bounded acceptance result, not proof of full
semantic correctness. This is not the official SWE-agent or mini-swe-agent harness.

All extension timings are diagnostic: embedding, reconstruction, generation and
container work shared one host. They are unsuitable for controlled throughput or
energy comparisons. Code, configurations, selected public data, results and
figure-generation procedures are included; manuscript sources/PDFs, submission
files, private review comments, local notes, caches and model weights are excluded.

A pre-generation grading correction handles upstream pytest identifiers truncated
at parameter whitespace. Exact IDs take precedence; an archived alias expands to all matching full IDs
observed in the gold control. This mapping is frozen before repair generation.
Every mapped member must appear and meet the phase-specific status requirement;
missing and skipped members fail. Tox controls motivated expansion beyond the
initial unique-only correction, which rejected collapsed parameterized families.
The correction was identified from Meltano base/gold controls before any generated
patch and changes no pass/fail status requirement. Original control records remain
under `results/extension/repair/control_initial/`; controls are regraded before the
agent matrix without rerunning containers or selecting on model quality.

The pgmpy baseline tests require four public example-network assets. The initial
network-isolated controls failed on those downloads even for the reference patch.
Before any repair generation, the assets were pinned by upstream commit/hash and
cached identically in base, gold, agent and evaluator containers. The fixture
manifest is `data/repair_fixtures.json`; the runner downloads these public assets
outside the isolated container and verifies their checksums. It does not modify
repository code or tests. The original failed control record is retained, and only
that infrastructure-affected control pair is rerun. No task is replaced.

After rerunning the repair panel, archive only the verification output referenced
by the result records:

```sh
python scripts/collect_repair_outputs.py --logs .cache/repair_logs
```

The collector verifies each output hash and reconstructs the recorded test statuses
before lossless compression. A manifest links original controls, amended controls
and generated-patch evaluations to their outputs; identical outputs are stored once.
No unrelated host or runtime logs are included.

### Post-hoc query/document window attribution

After the complete feature-encoder comparison, an exploratory analysis crosses
the 256/2048 query limits with the 256/2048 document limits. It recombines cached
vectors without further model calls and verifies that both original conditions
remain unchanged. This is explicitly post-hoc, not a prespecified primary result
or a replacement chosen by performance. The decision is recorded in
`data/cross_window_protocol.json`.

```sh
python scripts/derive_cross_window.py --cache .cache --embedding-cache .cache/semantic.sqlite
python scripts/derive_cross_window.py --cache .cache_external --embedding-cache .cache/semantic.sqlite --tasks data/external_tasks.jsonl --semantic results/extension/external/semantic --output results/extension/external/cross_window
python scripts/analyze_cross_window.py
```

Run both semantic populations first, using the same embedding database, or pass
their respective databases to the two derivation commands. The analysis exports
all four conditions and paired simple-effect/interaction intervals separately
for each population. Archived cross-window results can be analyzed directly
without reconstructing embeddings.

Navigation-pair diagnostics compare canonical structured first responses under
identical first requests and seed values. Some responses differ despite that
match; the archived analysis reports these pairs without dropping them. Scope
order is fixed. The comparison characterizes the configured systems and does
not isolate runtime or order effects from all decoding variability.

### Completed extension findings

All frozen matrices are complete and pass `scripts/audit_extension.py`: 508 feature
navigation episodes, 48 external navigation episodes, 208 three-condition encoder
records, 1,104 full-population Qwen responses, 96 external static responses, and
12 evaluated repair attempts. The original 508 Qwen records are unchanged.

| Primary R@10 | Feature tasks (184) | External issues (24) |
| --- | ---: | ---: |
| MiniLM, 256, hybrid top100 | 0.175 | 0.323 |
| CodeRankEmbed, 256, same pool | 0.304 | 0.325 |
| CodeRankEmbed, 2048, same pool | 0.206 | 0.233 |
| Navigation, restricted top40 | 0.315 | 0.599 |
| Navigation, repository | 0.445 | 0.657 |

The feature scope difference is +0.130, with repository-bootstrap interval
[0.089, 0.160]. The external difference is +0.058, with interval [-0.094, 0.197];
that smaller cohort does not resolve the same advantage. Invalid navigation
outputs remain scored as empty lists: 57/508 feature and 6/48 external episodes.
Full-population three-seed Qwen means are 0.403 (paths) and 0.389 (excerpts), with
an excerpt contrast interval [-0.033, 0.006]. No seed is selected for performance.

All twelve repair environments pass base/gold controls. None of the twelve model
attempts meets the full verification criterion. Eight emit empty diffs; four emit
nonempty diffs. One patch passes its fail-to-pass test but fails three pass-to-pass
tests, and another evaluation lacks all expected statuses. Missing tests fail the
criterion. The 138 model calls use the frozen 12-action budget; these outcomes
characterize this configured probe, not the ability of unrestricted repair agents.
`repair.csv` also records patch size and mapped fail-to-pass/pass-to-pass status
counts. The navigator and repair agent run independently, so these measurements
do not identify a causal localization-to-repair effect.

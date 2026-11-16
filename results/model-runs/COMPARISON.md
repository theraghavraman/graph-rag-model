# Real-model Graph-RAG versus RAG comparison

Completed on 16 November 2026. All three systems were run on the unchanged original 120 questions and the unchanged 60 paraphrased questions with **two actual local chat models**, for **1,080 scored answers**. The vector baseline uses actual Nomic embeddings. No prompt or question was revised after seeing results.

- **llama-3.2-3b-instruct:** questions: graph-llm has the most exact answers (69/120); questions_paraphrased: graph-llm has the most exact answers (32/60).
- **qwen2.5-coder-1.5b-instruct-mlx:** questions: graph-llm has the most exact answers (79/120); questions_paraphrased: graph-llm has the most exact answers (43/60).

The following tables also show question-type results and cases where a document baseline beats the graph. These local runs establish a measured comparison for the supplied implementation; they do not establish a universal Graph-RAG advantage.

## llama-3.2-3b-instruct

Exact match requires the complete predicted answer set to equal the gold set, using the supplied case-insensitive scorer and catalog-name normalization. F1 is the mean of per-question set F1 scores. All abstentions score zero.

| Set | System | Exact answers | Macro F1 | Abstentions | Negative false positives | Median seconds |
|---|---|---:|---:|---:|---:|---:|
| Original | graph-llm | 69/120 (57.5%) | 0.583 | 28/120 (23.3%) | 1/30 (3.3%) | 1.286 |
| Original | bm25-llm | 40/120 (33.3%) | 0.467 | 0/120 (0.0%) | 16/30 (53.3%) | 7.690 |
| Original | vector-llm | 40/120 (33.3%) | 0.458 | 0/120 (0.0%) | 16/30 (53.3%) | 8.353 |
| Paraphrased | graph-llm | 32/60 (53.3%) | 0.544 | 13/60 (21.7%) | 3/15 (20.0%) | 1.862 |
| Paraphrased | bm25-llm | 17/60 (28.3%) | 0.387 | 0/60 (0.0%) | 7/15 (46.7%) | 8.787 |
| Paraphrased | vector-llm | 21/60 (35.0%) | 0.431 | 0/60 (0.0%) | 7/15 (46.7%) | 4.300 |

### Results by question type

| Set | System | Lookup exact | One-hop exact | Multi-hop exact | Negative exact | Planner correct |
|---|---|---:|---:|---:|---:|---:|
| Original | graph-llm | 18/30 (60.0%) | 17/30 (56.7%) | 10/30 (33.3%) | 24/30 (80.0%) | 57/120 (47.5%) |
| Original | bm25-llm | 20/30 (66.7%) | 6/30 (20.0%) | 0/30 (0.0%) | 14/30 (46.7%) | — |
| Original | vector-llm | 20/30 (66.7%) | 6/30 (20.0%) | 0/30 (0.0%) | 14/30 (46.7%) | — |
| Paraphrased | graph-llm | 11/15 (73.3%) | 6/15 (40.0%) | 7/15 (46.7%) | 8/15 (53.3%) | 29/60 (48.3%) |
| Paraphrased | bm25-llm | 9/15 (60.0%) | 0/15 (0.0%) | 0/15 (0.0%) | 8/15 (53.3%) | — |
| Paraphrased | vector-llm | 11/15 (73.3%) | 2/15 (13.3%) | 0/15 (0.0%) | 8/15 (53.3%) | — |

### Paired outcomes

Each pair uses exactly the same questions; a win means one system was exact and the other was not. Both-wrong cases include partial answers and abstentions.

| Set | Baseline | Graph only exact | Baseline only exact | Both exact | Neither exact |
|---|---|---:|---:|---:|---:|
| Original | bm25-llm | 43 | 14 | 26 | 37 |
| Original | vector-llm | 42 | 13 | 27 | 38 |
| Paraphrased | bm25-llm | 19 | 4 | 13 | 24 |
| Paraphrased | vector-llm | 17 | 6 | 15 | 22 |

### Examples where the baseline wins

- **bm25-llm, questions, lookup-007**: Who owns the "Returns Rate" dashboard? Gold: `['Hannah Schmidt']`. Graph: abstained. Baseline: `['Hannah Schmidt']`. Graph explanation: I could not work out what to look up: the model found no matching entity. Closest catalog entries: rpt_returns_rate.
- **vector-llm, questions, lookup-007**: Who owns the "Returns Rate" dashboard? Gold: `['Hannah Schmidt']`. Graph: abstained. Baseline: `['Hannah Schmidt']`. Graph explanation: I could not work out what to look up: the model found no matching entity. Closest catalog entries: rpt_returns_rate.

### Examples where the graph wins

- **Versus bm25-llm, questions, one_hop-001**: Which pipelines read column destination_city of the core.fct_fulfilment table directly? Gold and graph: `['pl_mart_shipping_sla']`. Baseline: `['pl_mart_shipping_sla', 'pl_core_fct_fulfilment']`.
- **Versus vector-llm, questions, one_hop-001**: Which pipelines read column destination_city of the core.fct_fulfilment table directly? Gold and graph: `['pl_mart_shipping_sla']`. Baseline: `['pl_core_fct_fulfilment', 'pl_core_fct_shipments']`.

## qwen2.5-coder-1.5b-instruct-mlx

Exact match requires the complete predicted answer set to equal the gold set, using the supplied case-insensitive scorer and catalog-name normalization. F1 is the mean of per-question set F1 scores. All abstentions score zero.

| Set | System | Exact answers | Macro F1 | Abstentions | Negative false positives | Median seconds |
|---|---|---:|---:|---:|---:|---:|
| Original | graph-llm | 79/120 (65.8%) | 0.690 | 8/120 (6.7%) | 2/30 (6.7%) | 0.693 |
| Original | bm25-llm | 23/120 (19.2%) | 0.300 | 0/120 (0.0%) | 30/30 (100.0%) | 3.004 |
| Original | vector-llm | 23/120 (19.2%) | 0.264 | 0/120 (0.0%) | 30/30 (100.0%) | 2.357 |
| Paraphrased | graph-llm | 43/60 (71.7%) | 0.741 | 2/60 (3.3%) | 2/15 (13.3%) | 0.944 |
| Paraphrased | bm25-llm | 12/60 (20.0%) | 0.253 | 0/60 (0.0%) | 15/15 (100.0%) | 3.415 |
| Paraphrased | vector-llm | 15/60 (25.0%) | 0.301 | 0/60 (0.0%) | 15/15 (100.0%) | 2.190 |

### Results by question type

| Set | System | Lookup exact | One-hop exact | Multi-hop exact | Negative exact | Planner correct |
|---|---|---:|---:|---:|---:|---:|
| Original | graph-llm | 24/30 (80.0%) | 9/30 (30.0%) | 20/30 (66.7%) | 26/30 (86.7%) | 58/120 (48.3%) |
| Original | bm25-llm | 19/30 (63.3%) | 4/30 (13.3%) | 0/30 (0.0%) | 0/30 (0.0%) | — |
| Original | vector-llm | 18/30 (60.0%) | 5/30 (16.7%) | 0/30 (0.0%) | 0/30 (0.0%) | — |
| Paraphrased | graph-llm | 11/15 (73.3%) | 8/15 (53.3%) | 11/15 (73.3%) | 13/15 (86.7%) | 33/60 (55.0%) |
| Paraphrased | bm25-llm | 11/15 (73.3%) | 1/15 (6.7%) | 0/15 (0.0%) | 0/15 (0.0%) | — |
| Paraphrased | vector-llm | 10/15 (66.7%) | 5/15 (33.3%) | 0/15 (0.0%) | 0/15 (0.0%) | — |

### Paired outcomes

Each pair uses exactly the same questions; a win means one system was exact and the other was not. Both-wrong cases include partial answers and abstentions.

| Set | Baseline | Graph only exact | Baseline only exact | Both exact | Neither exact |
|---|---|---:|---:|---:|---:|
| Original | bm25-llm | 63 | 7 | 16 | 34 |
| Original | vector-llm | 62 | 6 | 17 | 35 |
| Paraphrased | bm25-llm | 35 | 4 | 8 | 13 |
| Paraphrased | vector-llm | 33 | 5 | 10 | 12 |

### Examples where the baseline wins

- **bm25-llm, questions, lookup-023**: Which tables come from the crm system? Gold: `['crm.returns', 'crm.reviews', 'crm.subscriptions', 'crm.suppliers', 'crm.tickets']`. Graph: abstained. Baseline: `['crm.returns', 'crm.suppliers', 'crm.subscriptions', 'crm.reviews', 'crm.tickets']`. Graph explanation: I could not work out what to look up: the model found no matching entity. Closest catalog entries: crm.
- **vector-llm, questions, lookup-023**: Which tables come from the crm system? Gold: `['crm.returns', 'crm.reviews', 'crm.subscriptions', 'crm.suppliers', 'crm.tickets']`. Graph: abstained. Baseline: `['crm.returns', 'crm.suppliers', 'crm.subscriptions', 'crm.reviews', 'crm.tickets']`. Graph explanation: I could not work out what to look up: the model found no matching entity. Closest catalog entries: crm.

### Examples where the graph wins

- **Versus bm25-llm, questions, one_hop-001**: Which pipelines read column destination_city of the core.fct_fulfilment table directly? Gold and graph: `['pl_mart_shipping_sla']`. Baseline: `['pl_core_fct_fulfilment', 'pl_mart_shipping_sla']`.
- **Versus vector-llm, questions, one_hop-001**: Which pipelines read column destination_city of the core.fct_fulfilment table directly? Gold and graph: `['pl_mart_shipping_sla']`. Baseline: `['pl_core_fct_fulfilment']`.

## BM25 versus vector with the same reader

Equal aggregate accuracy can hide different question-level outcomes. These counts compare the two document systems directly.

| Model | Set | BM25 only exact | Vector only exact | Both exact | Neither exact |
|---|---|---:|---:|---:|---:|
| llama-3.2-3b-instruct | questions | 10 | 10 | 30 | 70 |
| llama-3.2-3b-instruct | questions_paraphrased | 3 | 7 | 14 | 36 |
| qwen2.5-coder-1.5b-instruct-mlx | questions | 6 | 6 | 17 | 91 |
| qwen2.5-coder-1.5b-instruct-mlx | questions_paraphrased | 2 | 5 | 10 | 43 |

## Planner failure diagnosis

Plan correctness is a strict match to the gold structured query after entity resolution. A different valid query can still produce the right answer set on this catalog; this diagnostic is separate from answer accuracy.

| Model | Set | Gold-matching plan | Other valid plan | Invalid plan after retry | Model declined entity | Other abstention |
|---|---|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | questions | 57 | 35 | 16 | 10 | 2 |
| llama-3.2-3b-instruct | questions_paraphrased | 29 | 18 | 10 | 3 | 0 |
| qwen2.5-coder-1.5b-instruct-mlx | questions | 58 | 54 | 0 | 8 | 0 |
| qwen2.5-coder-1.5b-instruct-mlx | questions_paraphrased | 33 | 25 | 0 | 2 | 0 |

## Retrieval answer-name coverage

Share of gold answer names appearing literally anywhere in the retrieved documents, averaged across questions with nonempty gold answers. This is a lexical coverage diagnostic, not a mathematical ceiling: aliases, normalization and correct guesses can affect measured reader recall. k=8 matches the reader runs. No document is cut off by the 2,500-character limit (the longest supplied document is 1,123 characters).

| Set | Retriever | Group | k=8 answer-name coverage | k=8 all answers present | k=20 answer-name coverage |
|---|---|---|---:|---:|---:|
| Original | bm25 | lookup | 98.0% | 90.0% | 98.0% |
| Original | bm25 | one_hop | 100.0% | 100.0% | 100.0% |
| Original | bm25 | multi_hop | 26.0% | 0.0% | 43.3% |
| Original | vector | lookup | 92.7% | 90.0% | 98.0% |
| Original | vector | one_hop | 73.3% | 66.7% | 83.9% |
| Original | vector | multi_hop | 21.4% | 0.0% | 39.0% |
| Paraphrased | bm25 | lookup | 93.3% | 93.3% | 93.3% |
| Paraphrased | bm25 | one_hop | 100.0% | 100.0% | 100.0% |
| Paraphrased | bm25 | multi_hop | 29.5% | 0.0% | 55.4% |
| Paraphrased | vector | lookup | 93.3% | 93.3% | 93.3% |
| Paraphrased | vector | one_hop | 64.4% | 60.0% | 84.4% |
| Paraphrased | vector | multi_hop | 22.9% | 0.0% | 42.1% |

## Method and interpretation

- **Graph system:** the model selects a validated structured query; the graph computes the scored answer set. Optional model rephrasing was disabled. The comparison measures model planning plus exact graph execution versus retrieval plus model reading, not unrestricted generated-prose quality.
- **No ground truth supplied to models:** requests contain the fixed prompts, the question and either candidate entity names or retrieved documents. Gold queries and answer sets are used only for scoring after the prediction.
- **Shared reader settings:** temperature 0, top 8 documents, 2,500 characters per document, 500 output tokens. Planner: 200 output tokens, up to two attempts for invalid plans. These are the supplied implementation's defaults. A truncated or invalid response remains a failure; none was edited by hand.
- **Models:** Llama 3.2 3B Instruct (4-bit MLX), Qwen2.5 Coder 1.5B Instruct (4-bit MLX), and Nomic Embed Text v1.5 (Q4_K_M, 768 dimensions), served by the existing LM Studio endpoint. Model inventory and loaded context settings are saved in the manifest. Requests and responses identify the actual served model.
- **Embedding inputs:** unchanged `title + newline + text` for documents and the bare question for queries, matching the supplied code. No task-specific embedding prefixes, reranker, query expansion or multi-step retrieval were added. Results describe this implemented vector baseline, not every possible vector RAG configuration.
- **Timing:** questions and systems ran sequentially. Vector query/document embeddings were precomputed once using the real model, then reused exactly. Reported vector per-question latency excludes that preprocessing and is not an end-to-end vector latency comparison. Model loading, cache warming and execution order can also affect timing. Token usage is endpoint-reported; no hosted API cost was incurred.
- **Scope:** one run per cell, two small local quantized chat models, one embedding model, one synthetic catalog and two templated question sets. The paraphrased set samples different entities with another seed; it is not a paired rewrite of the same 120 questions, so cross-set differences cannot be attributed only to wording. It was previously inspected by the repository author and is not an untouched independent test set. These measurements do not establish results for hosted models, production catalogs or other RAG designs.
- **Negative questions:** an empty answer is correct only when it is not an abstention. Returning empty on every question would score 25% exact overall because one quarter of each set is negative. Group-level results and planner accuracy make this visible.

## Reproduce and audit

```sh
pip install -r requirements-benchmark.txt
pip install -e ".[dev]"
# Start a compatible server with these model IDs available, then:
python scripts/benchmark_models.py \
  --model llama-3.2-3b-instruct \
  --model qwen2.5-coder-1.5b-instruct-mlx \
  --embed-model text-embedding-nomic-embed-text-v1.5 \
  --base-url http://localhost:1234/v1 \
  --out results/new-model-runs
python scripts/summarize_benchmark.py --runs results/new-model-runs
pytest -q
```

The supplied database was rebuilt with the generator's defaults; all three supplied JSONL files reproduced byte for byte and were retained unchanged. The runner binds resumable checkpoints to data, source and configuration hashes. A completed directory is resumed without replacing predictions; use a fresh output directory for a new measurement.

Validation passed for **12 complete cells, 1080 scored answers, 1139 successful live chat requests and 458 real embedding vectors**. Scores were recomputed from predictions; graph plans were rechecked against the gold queries; trace model IDs, usage totals, unique question IDs and input hashes were checked. The original test suite passed: 77 tests.

Files: `manifest.json` contains provenance; each model directory contains standard result JSON, incremental prediction checkpoints and raw request/response traces. `embedding-cache.json` contains actual numeric vectors and `embeddings-trace.jsonl` their request provenance. `comparison.json` contains unrounded overview metrics. The original retrieval-only result files remain in the parent directory as historical measurements.

## Response completion checks

| Model | System | Set | Output-token limit reached | Request errors | Invalid reader format |
|---|---|---|---:|---:|---:|
| llama-3.2-3b-instruct | graph-llm | questions | 0 | 0 | — |
| llama-3.2-3b-instruct | graph-llm | questions_paraphrased | 0 | 0 | — |
| llama-3.2-3b-instruct | bm25-llm | questions | 0 | 0 | 0 |
| llama-3.2-3b-instruct | bm25-llm | questions_paraphrased | 0 | 0 | 0 |
| llama-3.2-3b-instruct | vector-llm | questions | 0 | 0 | 0 |
| llama-3.2-3b-instruct | vector-llm | questions_paraphrased | 0 | 0 | 0 |
| qwen2.5-coder-1.5b-instruct-mlx | graph-llm | questions | 0 | 0 | — |
| qwen2.5-coder-1.5b-instruct-mlx | graph-llm | questions_paraphrased | 0 | 0 | — |
| qwen2.5-coder-1.5b-instruct-mlx | bm25-llm | questions | 0 | 0 | 0 |
| qwen2.5-coder-1.5b-instruct-mlx | bm25-llm | questions_paraphrased | 0 | 0 | 0 |
| qwen2.5-coder-1.5b-instruct-mlx | vector-llm | questions | 0 | 0 | 0 |
| qwen2.5-coder-1.5b-instruct-mlx | vector-llm | questions_paraphrased | 0 | 0 | 0 |

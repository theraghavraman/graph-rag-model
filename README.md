# catalog-graph-rag

Graph-based retrieval over a data catalog. It executes structured lineage queries
exactly by traversing a graph of tables, columns, pipelines, reports and owners.
A model translates questions such as "which reports depend on this source column?"
into queries. The benchmark compares this pipeline against BM25 and vector retrieval
with real language models on the same questions.

**Status: model comparison measured locally.** All three model-backed systems have now
been run with Llama 3.2 3B Instruct and Qwen2.5 Coder 1.5B Instruct on both question sets:
1,080 scored answers, with Nomic Embed Text v1.5 for vector retrieval. The original
77 tests pass. [Full comparison, failure cases and provenance](results/model-runs/COMPARISON.md).

## The idea

Lineage questions have exact answers, so the graph computes them and a language model
does not. The model's job is narrow: turn a question into a small validated structure,
then phrase the result.

```
question ──> planner (LLM) ──> CatalogQuery (validated) ──> graph traversal ──> paths ──> answer (LLM)
              rules or model     validated, never SQL         DuckDB recursive query        template or model
```

## Quick start

```sh
pip install -e ".[dev]"
catalog-graph build        # catalog.duckdb, two question sets and the baseline documents, under data/
catalog-graph ask "Which reports depend on support.subscriptions.started_on?"
catalog-graph query lineage "Revenue Overview" --direction upstream --target system --paths
catalog-graph eval --system graph-rules
catalog-graph eval --system retrieval-bound
pytest
```

`ask` prints the structured query it ran, every result, and the evidence path:

```
query    {"kind":"lineage","entity":"support.subscriptions.started_on","target_type":"report"}
5 results for reports downstream of support.subscriptions.started_on:
  - rpt_revenue_overview (3 steps)
  ...
Evidence:
  support.subscriptions.started_on -> stg.support_subscriptions.started_on -> core.fct_subscriptions.started_on -> rpt_revenue_overview
```

When it cannot tell what is being asked, it says so and exits with status 1; it does not guess.

## Real-model results

Measured on 16 November 2026, using two local 4-bit chat models and the existing LM Studio server. The supplied prompts, questions and retrieval settings were retained unchanged. These are exact-answer rates; abstentions count as wrong.

| Chat model | Set | graph-llm | bm25-llm | vector-llm |
|---|---|---:|---:|---:|
| llama-3.2-3b-instruct | Original | 69/120 (57.5%) | 40/120 (33.3%) | 40/120 (33.3%) |
| llama-3.2-3b-instruct | Paraphrased | 32/60 (53.3%) | 17/60 (28.3%) | 21/60 (35.0%) |
| qwen2.5-coder-1.5b-instruct-mlx | Original | 79/120 (65.8%) | 23/120 (19.2%) | 23/120 (19.2%) |
| qwen2.5-coder-1.5b-instruct-mlx | Paraphrased | 43/60 (71.7%) | 12/60 (20.0%) | 15/60 (25.0%) |

The [full report](results/model-runs/COMPARISON.md) separates lookup, one-hop, multi-hop and negative questions; includes F1, planner accuracy, abstentions, false positives, paired wins and baseline-winning examples; and measures both BM25 and vector answer-name coverage. Raw model requests and responses, per-question predictions, source/data hashes and exact dependency versions are included.

The graph computes the scored answer set after model planning; optional model rephrasing was disabled. These runs compare this graph pipeline with the implemented retrieval-and-reader pipelines. They cover two small local models and a synthetic catalog; a hosted-model replication and independent questions remain future work.

Vector embeddings were computed once with the real model and cached. Reported vector per-question latency excludes this preprocessing, so it is not an end-to-end latency comparison. See the report for the full configuration and limitations.

```sh
pip install -r requirements-benchmark.txt
pip install -e .
python scripts/benchmark_models.py \
  --model llama-3.2-3b-instruct \
  --model qwen2.5-coder-1.5b-instruct-mlx \
  --embed-model text-embedding-nomic-embed-text-v1.5 \
  --base-url http://localhost:1234/v1 \
  --out results/new-model-runs
python scripts/summarize_benchmark.py --runs results/new-model-runs
```

Use a fresh output directory for a new measurement. A matching existing directory resumes saved predictions.

## Earlier retrieval and rule measurements

The following historical measurements were run on 16 November 2026 without a language model.
They are retained separately from the real-model runs above.

### 1. Keyword retrieval has limited multi-hop answer-name coverage

For each question, what share of the correct answer names appear anywhere in the top-k
documents BM25 retrieves? This is literal answer-name coverage in retrieved context;
alias normalization and correct guesses can affect a reader's measured recall, so it is
a diagnostic rather than a mathematical ceiling. The figure in brackets is the share of
questions where every answer name was retrieved.

| Question set | lookup, k=5 | one hop, k=5 | multi-hop, k=5 | multi-hop, k=10 | multi-hop, k=20 |
|---|---:|---:|---:|---:|---:|
| Original wording (120) | 97% | 100% | 17% (0%) | 29% (3%) | 43% (23%) |
| Held-out wording (60) | 93% | 93% | 21% (0%) | 30% (0%) | 55% (33%) |

The documents contain every lineage fact (a test checks this), so low coverage is not
caused by omitted facts. The answer to "which reports depend on this source
column" sits three or four documents away from anything the question mentions. Given the
right query, the graph returns the complete answer set for the same questions, because it
follows the chain.
The historical table is BM25 only. Both retrievers are now measured at k=5, 8, 10 and 20 in the full model report.

### 2. Rules do not generalise, which is the case for a model planner

| Graph system with the rule planner | planned correctly | exact answers | wrong answers | abstained |
|---|---:|---:|---:|---:|
| Original wording (120) | 100% | 100% | 0 | 0% |
| Held-out wording (60) | 0% | 0% | 0 | 100% |

The rules were written against the original phrasings and score perfectly on them, which
says nothing: that is fitting to my own templates. On 60 questions worded differently
they recognised none and abstained every time. They never gave a wrong answer, and an
abstention is scored as incorrect, including on questions whose true answer is "nothing".

### 3. Entity linking

The linker found the right entity first for 100% of the original questions and
100% of the held-out ones. The held-out figure was 88% until a unit test on a
separate hand-built graph exposed a bug (typos in dotted names were skipped whenever the
table name still matched). The fix was not derived from the held-out set, but the set had
already been run once, so treat 100% as optimistic. Both sets mention entities in fairly
regular ways; real questions will be harder.

### Model-backed runs

The three systems formerly listed as unmeasured are covered by the real-model results
above. The original `catalog-graph eval` commands still run a single system and question
set; `scripts/benchmark_models.py` adds the full matrix, raw traces and resumable progress.

## What is in the catalog

`catalog-graph build` generates a synthetic warehouse, the same every time for a given seed:

| | |
|---|---|
| Source systems | 8 |
| Tables | 135 (40 raw, 40 staging, 30 core, 25 mart) |
| Columns | about 1,100 |
| Pipelines | 95, one per derived table |
| Reports | 40 |
| Owners | 12 |

Every derived column records the exact columns it is computed from. Some columns are
deliberate dead ends, so "does anything use X?" has true negative answers. All names
and people are invented.

## The question set

`data/questions.jsonl` holds 120 questions, 30 per group. Each has the structured query
that answers it and the exact answer set, both computed from the graph.

| Group | Example |
|---|---|
| lookup | Who owns the stg.erp_leads table? |
| one_hop | Which pipelines read column destination_city of the core.fct_fulfilment table directly? |
| multi_hop | Which dashboards ultimately use wms.employees.job_title? |
| negative | Which dashboards would break if the brand column in shop.products were removed? (answer: none) |

Eight of the negatives are near misses: the column flows through several tables but
never reaches a report.

`data/questions_paraphrased.jsonl` holds 60 more with looser wording ("blast radius",
"who needs a heads-up"), looser entity mentions ("total amount in billing orders") and
the occasional typo. It is a held-out set: the rules and the linker were written against
the original wording. Both sets are templated and written by one author, so the second is
a harder test, not an independent one.

## The baseline will be given the same facts

`data/documents.jsonl` restates the whole catalog as 278 text documents. Pipeline
documents spell out every column mapping and report documents list every column they
show; a test checks that no lineage fact is missing. If vector retrieval loses on
multi-hop questions, it will be because the answer needs several documents chained
together, not because facts were withheld.

## Layout

```
src/catalog_graph/
  model.py        nodes and edges
  generator.py    synthetic catalog
  store.py        DuckDB schema, exact-name resolution, traversal
  query.py        the only query shapes that will be executed
  questions.py    question set with ground truth
  paraphrases.py  held-out question set with different wording
  documents.py    text documents for the baseline
  linker.py       entity linking: exact, composed, fuzzy
  planner.py      question -> query: rules, or a model with validation and one retry
  llm.py          client for any OpenAI-compatible endpoint (standard library only)
  retrieval.py    BM25 and embedding retrieval
  systems.py      the graph system and the document-reading baseline
  answer.py       answer text with evidence; abstention message
  evaluate.py     metrics, answer-name coverage, linking accuracy, result files
  cli.py
tests/            hand-checked answers on a tiny graph, properties of the generated one,
                  scripted-model tests, and a local OpenAI-compatible server for the client
results/          every measured run, per question
docs/DESIGN.md    decisions, limits and what comes next
```

## Limits

- **Local-model scope.** The headline comparison has been run with two small, quantized
  local chat models. Hosted models and larger models have not been measured.
- **Synthetic catalog.** Models are wired to plausibly related inputs, but it is not a
  real warehouse and results on it will not transfer directly to one.
- **Templated questions.** Both sets come from templates by one author. Hand-written
  questions from other people are needed before the numbers mean much.
- **One vector configuration.** Dense retrieval is now measured with Nomic Embed Text
  v1.5 and the supplied inputs, without task prefixes, reranking or query expansion.
  This does not represent every vector RAG configuration.
- **Column-level lineage only.** Row filters and conditional logic are not modelled.
- **Planner failures remain.** The real-model runs expose invalid plans, abstentions and
  incorrect valid queries. Prompt improvements should be developed separately and tested
  on new, unseen questions.

MIT licence.

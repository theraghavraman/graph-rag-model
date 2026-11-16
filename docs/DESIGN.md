# Design notes

## Decisions

**The graph computes the answer; the model does not.**
"Which reports depend on this column?" has one correct answer. Asking a model to read
retrieved text and reason its way there adds a way to be wrong. The model is limited to
translating the question into a `CatalogQuery` and wording the result.

**The model never writes SQL.**
`CatalogQuery` has three kinds (owner, members, lineage) and a handful of typed fields.
Unknown fields are rejected, hop counts are bounded, and entity names reach the database
only as bound parameters. A test passes a name containing SQL and checks nothing happens.

**Column-to-column edges, with the pipeline on the edge.**
The first version routed lineage through pipeline nodes (column -> pipeline -> column).
That made every input of a pipeline appear to feed every output, so a path existed from
`returned_on` to `order_item_id` only because one job touched both. Lineage edges now run
column -> column and carry `via`, the pipeline that performs the step. Column lineage is
exact, and pipeline questions are answered from the edges that were traversed.

**DuckDB, not a graph database.**
Two tables and a recursive query handle about 1,400 nodes and 2,600 edges in
milliseconds, with no server to run. The 120 gold queries execute in under a second in
total. A graph database would earn its place at a much larger scale or with heavier
path queries.

**Every result carries its evidence path.**
Reachability runs in SQL; one shortest path per result is rebuilt from the hop counts and
returned with it. An answer can then show "A -> B -> C -> report", which is what a
person checks before trusting it.

**Hops count processing steps.**
One pipeline run from a column to the column it produces is one hop, and a column shown on
a report is one hop. A pipeline as the starting point has already taken its step.

**Ground truth comes from the generator, not from a model.**
The catalog is generated, so every answer is known. Questions store the query that
answers them, and a test re-executes all 120 and compares.

**A planner may abstain, and abstaining is scored as wrong.**
Both planners return "no query" when they cannot decide, and the system then says it could
not work out what to look up. In scoring, an abstention never counts as correct, even when
the true answer is the empty set. An earlier version of the scoring gave the rule planner
100% on held-out negative questions for recognising none of them; that was a scoring bug,
not a result.

**The model chooses among candidates; it does not name entities freely.**
The linker proposes up to six catalog entities and the model must copy one exactly. A
plan naming anything else is rejected, even if it exists in the catalog. This removes
invented table names as a failure mode and keeps the model's task to "what is being
asked", which is what it is good at.

**One corrective retry, then abstain.**
An invalid plan is sent back once with the validation error. Two attempts bound the cost
and latency of a question; a third rarely helps a model that failed twice.

**A rule planner exists to be beaten.**
It costs nothing and answers instantly. If a model planner does no better on held-out
wording, the model has not earned its place. The rules scored 100% on the wording they
were written for and 0% on 60 held-out questions, so the bar is low but real.

**Retrieval answer-name coverage is measured separately from the reader.**
Answer-name coverage checks what the retriever puts in front of the reader. For multi-hop
questions BM25 surfaces 17% of the answer names at k=5 and 43% at k=20 in the original
historical run. Literal-name matching is a diagnostic, not a strict mathematical ceiling:
alias normalization and correct guesses can affect measured reader recall. Both BM25
and vector coverage are now measured alongside actual model-reader results.

**The baseline gets help where it is cheap to give.**
Reader output is mapped onto catalog names before scoring, so "Churn Watchlist" counts as
`rpt_churn_watchlist`. The comparison should be about retrieval and reasoning, not naming.

**The client is about a hundred lines of standard library.**
Any OpenAI-compatible endpoint works: Ollama, LM Studio, vLLM or a hosted API. No SDK to
pin, and it is tested against a local server that speaks the same wire format.

## Limits

- **Synthetic data.** Wiring is guided by subject words (customer_churn reads customer and
  subscription models), but the result is still artificial. In a real catalog, names carry
  more signal, which may help vector retrieval more than it will here. A public dbt
  project is planned as a check.
- **Sparse reach.** About three quarters of source columns reach no report. That is
  plausible, but it means the multi-hop group is drawn from the remaining quarter.
- **Linking is lexical.** "total amount in billing orders" and a one-letter typo resolve;
  "the field with the customer's mail address" does not.
- **One owner per object, one pipeline per table.** Real catalogs are messier.
- **Question wording is templated.** Variation is limited to a few phrasings per type.
  Hand-written paraphrases should be added before the evaluation is taken seriously.

## Next

**Replicate with a hosted model.** All three systems have now been measured on both sets
with two actual local chat models, and both retrievers' answer-name coverage has been
measured. The [complete comparison](../results/model-runs/COMPARISON.md) includes failures,
paired wins and raw model traces. A hosted-model run remains outstanding; no hosted API
credential was configured for this execution.

**Improve planning on a separate development set.** Preserve the reported unchanged-prompt
runs. Test any revised prompt on new questions rather than tuning to the measured sets.

**Collect hand-written questions** from two or three other people, with the catalog in
front of them, and score all systems on those.

**A public dbt project** parsed with sqlglot as a realism check.

**Embedding pass in the linker** for mentions that share no words with the entity name.

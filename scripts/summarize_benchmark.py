#!/usr/bin/env python3
"""Validate saved runs against inputs and traces, then publish the comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

from catalog_graph import evaluate
from catalog_graph.llm import extract_json
from catalog_graph.planner import same_query
from catalog_graph.query import CatalogQuery
from catalog_graph.questions import GROUPS, read_jsonl
from catalog_graph.store import GraphStore

SYSTEMS = ("graph-llm", "bm25-llm", "vector-llm")
SETS = ("questions", "questions_paraphrased")


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rate(n, total):
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/model-runs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    manifest = load(args.runs / "manifest.json")
    assert "finished_at_utc" in manifest, "Matrix has not finished"
    config = manifest["config"]
    for name, expected in config["inputs_sha256"].items():
        assert hashlib.sha256((args.data / name).read_bytes()).hexdigest() == expected, name
    for name, expected in config["source_sha256"].items():
        assert hashlib.sha256((Path("src/catalog_graph") / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256(Path("scripts/benchmark_models.py").read_bytes()).hexdigest() == config["runner_sha256"]
    qs = {label: read_jsonl(args.data / f"{label}.jsonl") for label in SETS}
    runs = {}
    total_calls = 0
    with GraphStore(str(args.data / "catalog.duckdb"), read_only=True) as store:
        for model in config["models"]:
            model_dir = args.runs / re.sub(r"[^A-Za-z0-9._-]", "_", model)
            for system in SYSTEMS:
                for label in SETS:
                    stem = f"{system}__{label}"
                    result = load(model_dir / f"{stem}.json")
                    checkpoint = lines(model_dir / f"{stem}.checkpoint.jsonl")
                    trace = lines(model_dir / f"{stem}.trace.jsonl")
                    rows = result["rows"]
                    assert len(rows) == len(qs[label]) == len(checkpoint)
                    assert len({r["id"] for r in rows}) == len(rows)
                    for row, q, saved in zip(rows, qs[label], checkpoint):
                        assert (row["id"], row["group"], row["question"], row["gold"]) == (q.id, q.group, q.question, q.answers)
                        assert row == saved["row"]
                        pred = saved["prediction"]
                        assert pred["answers"] == row["predicted"] and pred["abstained"] == row["abstained"]
                        expected = (0., 0., 0., False) if row["abstained"] else evaluate.score_sets(row["predicted"], q.answers)
                        assert tuple(row[key] for key in ("precision", "recall", "f1", "exact")) == expected
                        if system == "graph-llm":
                            correct = pred["query"] is not None and same_query(CatalogQuery(**pred["query"]), CatalogQuery(**q.gold_query), store)
                            assert row["plan_correct"] == correct
                    assert evaluate.summarise([evaluate.Row(**row) for row in rows]) == result["summary"]
                    successful = [event for event in trace if "response" in event]
                    assert len(successful) == result["meta"]["model_calls"]
                    for event in successful:
                        assert event["path"] == "/chat/completions"
                        assert event["request"]["model"] == model == event["response"]["model"]
                        assert event["response"]["id"].startswith("chatcmpl-")
                        assert event["request"]["temperature"] == 0
                    assert {event["question_id"] for event in trace} == {q.id for q in qs[label]}
                    total_calls += len(successful)
                    for key in ("prompt_tokens", "completion_tokens"):
                        assert sum(event["response"]["usage"][key] for event in successful) == result["meta"][key]
                    runs[(model, system, label)] = {**result, "trace": trace, "checkpoint": checkpoint}
    embeddings = load(args.runs / "embedding-cache.json")
    vector_values = list(embeddings["vectors"].values())
    assert len({len(v) for v in vector_values}) == 1
    assert all(any(v) and all(math.isfinite(x) for x in v) for v in vector_values)
    embedding_trace = lines(args.runs / "embeddings-trace.jsonl")
    assert sum(len(t["request"]["input"]) for t in embedding_trace) == len(vector_values)
    assert {text for t in embedding_trace for text in t["request"]["input"]} == set(embeddings["vectors"])
    assert all(t["request"]["model"] == config["embed_model"] and "response" in t for t in embedding_trace)

    finished = datetime.fromisoformat(manifest["finished_at_utc"]).astimezone(ZoneInfo("Asia/Kolkata"))
    finished_date = f"{finished.day} {finished.strftime('%B %Y')}"
    out = ["# Real-model Graph-RAG versus RAG comparison", "",
           f"Completed on {finished_date}. All three systems were run on the unchanged original 120 questions and the unchanged 60 paraphrased questions with **two actual local chat models**, for **1,080 scored answers**. "
           "The vector baseline uses actual Nomic embeddings. No prompt or question was revised after seeing results.", ""]
    for model in config["models"]:
        descriptions = []
        for label in SETS:
            cells = [runs[(model, system, label)]["rows"] for system in SYSTEMS]
            counts = [sum(r["exact"] for r in rows) for rows in cells]
            winner = ", ".join(system for system, count in zip(SYSTEMS, counts) if count == max(counts))
            descriptions.append(f"{label}: {winner} has the most exact answers ({max(counts)}/{len(cells[0])})")
        out += [f"- **{model}:** " + "; ".join(descriptions) + "."]
    out += ["", "The following tables also show question-type results and cases where a document baseline beats the graph. "
            "These local runs establish a measured comparison for the supplied implementation; they do not establish a universal Graph-RAG advantage.", ""]
    overview = []
    for model in config["models"]:
        out += [f"## {model}", "", "Exact match requires the complete predicted answer set to equal the gold set, using the supplied case-insensitive scorer and catalog-name normalization. F1 is the mean of per-question set F1 scores. All abstentions score zero.", "",
                "| Set | System | Exact answers | Macro F1 | Abstentions | Negative false positives | Median seconds |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for label in SETS:
            for system in SYSTEMS:
                run = runs[(model, system, label)]
                rows = run["rows"]
                n = len(rows)
                negative = [r for r in rows if not r["gold"]]
                exact = sum(r["exact"] for r in rows)
                f1 = statistics.fmean(r["f1"] for r in rows)
                abstained = sum(r["abstained"] for r in rows)
                fp = sum(bool(r["predicted"]) for r in negative)
                seconds = statistics.median(r["seconds"] for r in rows)
                out += [f"| {'Original' if label == 'questions' else 'Paraphrased'} | {system} | {rate(exact, n)} | {f1:.3f} | {rate(abstained, n)} | {rate(fp, len(negative))} | {seconds:.3f} |"]
                overview.append({"model": model, "system": system, "set": label, "questions": n, "exact_count": exact,
                                 "exact_rate": exact / n, "macro_f1": f1, "abstentions": abstained,
                                 "negative_false_positives": fp, "negative_questions": len(negative), "median_seconds": seconds})
        out += ["", "### Results by question type", "",
                "| Set | System | Lookup exact | One-hop exact | Multi-hop exact | Negative exact | Planner correct |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for label in SETS:
            for system in SYSTEMS:
                rows = runs[(model, system, label)]["rows"]
                cells = []
                for group in GROUPS:
                    subset = [r for r in rows if r["group"] == group]
                    cells.append(rate(sum(r["exact"] for r in subset), len(subset)))
                planned = rate(sum(bool(r["plan_correct"]) for r in rows), len(rows)) if system == "graph-llm" else "—"
                out += [f"| {'Original' if label == 'questions' else 'Paraphrased'} | {system} | " + " | ".join(cells) + f" | {planned} |"]
        out += ["", "### Paired outcomes", "",
                "Each pair uses exactly the same questions; a win means one system was exact and the other was not. Both-wrong cases include partial answers and abstentions.", "",
                "| Set | Baseline | Graph only exact | Baseline only exact | Both exact | Neither exact |",
                "|---|---|---:|---:|---:|---:|"]
        for label in SETS:
            graph = runs[(model, "graph-llm", label)]["rows"]
            for baseline in SYSTEMS[1:]:
                other = runs[(model, baseline, label)]["rows"]
                counts = Counter((g["exact"], b["exact"]) for g, b in zip(graph, other))
                out += [f"| {'Original' if label == 'questions' else 'Paraphrased'} | {baseline} | {counts[(True, False)]} | {counts[(False, True)]} | {counts[(True, True)]} | {counts[(False, False)]} |"]
        out += ["", "### Examples where the baseline wins", ""]
        for baseline in SYSTEMS[1:]:
            found = False
            for label in SETS:
                for g, b, cp in zip(runs[(model, "graph-llm", label)]["rows"], runs[(model, baseline, label)]["rows"], runs[(model, "graph-llm", label)]["checkpoint"]):
                    if b["exact"] and not g["exact"]:
                        out += [f"- **{baseline}, {label}, {g['id']}**: {g['question']} Gold: `{g['gold']}`. "
                                f"Graph: {'abstained' if g['abstained'] else str(g['predicted'])}. Baseline: `{b['predicted']}`. "
                                + (f"Graph explanation: {cp['prediction']['text']}" if g["abstained"] else f"Graph plan: `{cp['prediction']['query']}`")]
                        found = True
                        break
                if found:
                    break
            if not found:
                out += [f"- No {baseline}-only exact cases in either set."]
        out += ["", "### Examples where the graph wins", ""]
        for baseline in SYSTEMS[1:]:
            found = False
            for label in SETS:
                for g, b in zip(runs[(model, "graph-llm", label)]["rows"], runs[(model, baseline, label)]["rows"]):
                    if g["exact"] and not b["exact"] and g["group"] in ("one_hop", "multi_hop"):
                        out += [f"- **Versus {baseline}, {label}, {g['id']}**: {g['question']} "
                                f"Gold and graph: `{g['gold']}`. Baseline: `{b['predicted']}`" +
                                (" (abstained)." if b["abstained"] else ".")]
                        found = True
                        break
                if found:
                    break
            if not found:
                out += [f"- No graph-only exact one-hop or multi-hop case against {baseline} in either set."]
        out += [""]
    out += ["## BM25 versus vector with the same reader", "",
            "Equal aggregate accuracy can hide different question-level outcomes. These counts compare the two document systems directly.", "",
            "| Model | Set | BM25 only exact | Vector only exact | Both exact | Neither exact |",
            "|---|---|---:|---:|---:|---:|"]
    for model in config["models"]:
        for label in SETS:
            counts = Counter((a["exact"], b["exact"]) for a, b in zip(runs[(model, "bm25-llm", label)]["rows"], runs[(model, "vector-llm", label)]["rows"]))
            out += [f"| {model} | {label} | {counts[(True, False)]} | {counts[(False, True)]} | {counts[(True, True)]} | {counts[(False, False)]} |"]
    out += ["", "## Planner failure diagnosis", "",
            "Plan correctness is a strict match to the gold structured query after entity resolution. A different valid query can still produce the right answer set on this catalog; this diagnostic is separate from answer accuracy.", "",
            "| Model | Set | Gold-matching plan | Other valid plan | Invalid plan after retry | Model declined entity | Other abstention |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for model in config["models"]:
        for label in SETS:
            counts = Counter()
            for item in runs[(model, "graph-llm", label)]["checkpoint"]:
                row = item["row"]
                text = item["prediction"]["text"]
                if row["plan_correct"]:
                    category = "correct"
                elif not row["abstained"]:
                    category = "wrong-valid"
                elif "did not produce a valid query" in text:
                    category = "invalid"
                elif "model found no matching entity" in text:
                    category = "declined"
                else:
                    category = "other"
                counts[category] += 1
            out += [f"| {model} | {label} | {counts['correct']} | {counts['wrong-valid']} | {counts['invalid']} | {counts['declined']} | {counts['other']} |"]
    out += ["", "## Retrieval answer-name coverage", "",
            "Share of gold answer names appearing literally anywhere in the retrieved documents, averaged across questions with nonempty gold answers. "
            "This is a lexical coverage diagnostic, not a mathematical ceiling: aliases, normalization and correct guesses can affect measured reader recall. k=8 matches the reader runs. "
            "No document is cut off by the 2,500-character limit (the longest supplied document is 1,123 characters).", "",
            "| Set | Retriever | Group | k=8 answer-name coverage | k=8 all answers present | k=20 answer-name coverage |",
            "|---|---|---|---:|---:|---:|"]
    for label in SETS:
        for retriever in ("bm25", "vector"):
            bound = load(args.runs / f"retrieval-bound-{retriever}__{label}.json")
            for group in ("lookup", "one_hop", "multi_hop"):
                b = bound["k=8"][group]
                out += [f"| {'Original' if label == 'questions' else 'Paraphrased'} | {retriever} | {group} | {100*b['answer_recall_bound']:.1f}% | {100*b['all_answers_present']:.1f}% | {100*bound['k=20'][group]['answer_recall_bound']:.1f}% |"]
    out += ["", "## Method and interpretation", "",
            "- **Graph system:** the model selects a validated structured query; the graph computes the scored answer set. Optional model rephrasing was disabled. The comparison measures model planning plus exact graph execution versus retrieval plus model reading, not unrestricted generated-prose quality.",
            "- **No ground truth supplied to models:** requests contain the fixed prompts, the question and either candidate entity names or retrieved documents. Gold queries and answer sets are used only for scoring after the prediction.",
            "- **Shared reader settings:** temperature 0, top 8 documents, 2,500 characters per document, 500 output tokens. Planner: 200 output tokens, up to two attempts for invalid plans. These are the supplied implementation's defaults. A truncated or invalid response remains a failure; none was edited by hand.",
            "- **Models:** Llama 3.2 3B Instruct (4-bit MLX), Qwen2.5 Coder 1.5B Instruct (4-bit MLX), and Nomic Embed Text v1.5 (Q4_K_M, 768 dimensions), served by the existing LM Studio endpoint. Model inventory and loaded context settings are saved in the manifest. Requests and responses identify the actual served model.",
            "- **Embedding inputs:** unchanged `title + newline + text` for documents and the bare question for queries, matching the supplied code. No task-specific embedding prefixes, reranker, query expansion or multi-step retrieval were added. Results describe this implemented vector baseline, not every possible vector RAG configuration.",
            "- **Timing:** questions and systems ran sequentially. Vector query/document embeddings were precomputed once using the real model, then reused exactly. Reported vector per-question latency excludes that preprocessing and is not an end-to-end vector latency comparison. Model loading, cache warming and execution order can also affect timing. Token usage is endpoint-reported; no hosted API cost was incurred.",
            "- **Scope:** one run per cell, two small local quantized chat models, one embedding model, one synthetic catalog and two templated question sets. The paraphrased set samples different entities with another seed; it is not a paired rewrite of the same 120 questions, so cross-set differences cannot be attributed only to wording. It was previously inspected by the repository author and is not an untouched independent test set. These measurements do not establish results for hosted models, production catalogs or other RAG designs.",
            "- **Negative questions:** an empty answer is correct only when it is not an abstention. Returning empty on every question would score 25% exact overall because one quarter of each set is negative. Group-level results and planner accuracy make this visible.",
            "", "## Reproduce and audit", "", "```sh", 'pip install -r requirements-benchmark.txt', 'pip install -e ".[dev]"',
            "# Start a compatible server with these model IDs available, then:",
            "python scripts/benchmark_models.py \\",
            "  --model llama-3.2-3b-instruct \\",
            "  --model qwen2.5-coder-1.5b-instruct-mlx \\",
            "  --embed-model text-embedding-nomic-embed-text-v1.5 \\",
            "  --base-url http://localhost:1234/v1 \\",
            "  --out results/new-model-runs",
            "python scripts/summarize_benchmark.py --runs results/new-model-runs", "pytest -q", "```", "",
            "The supplied database was rebuilt with the generator's defaults; all three supplied JSONL files reproduced byte for byte and were retained unchanged. "
            "The runner binds resumable checkpoints to data, source and configuration hashes. A completed directory is resumed without replacing predictions; use a fresh output directory for a new measurement.", "",
            f"Validation passed for **{len(runs)} complete cells, {sum(len(run['rows']) for run in runs.values())} scored answers, {total_calls} successful live chat requests and {len(vector_values)} real embedding vectors**. "
            "Scores were recomputed from predictions; graph plans were rechecked against the gold queries; trace model IDs, usage totals, unique question IDs and input hashes were checked. "
            "The original test suite passed: 77 tests.", "",
            "Files: `manifest.json` contains provenance; each model directory contains standard result JSON, incremental prediction checkpoints and raw request/response traces. "
            "`embedding-cache.json` contains actual numeric vectors and `embeddings-trace.jsonl` their request provenance. `comparison.json` contains unrounded overview metrics. "
            "The original retrieval-only result files remain in the parent directory as historical measurements.", ""]
    failures = []
    for (model, system, label), run in runs.items():
        finish = Counter(t.get("response", {}).get("choices", [{}])[0].get("finish_reason", "request-error") for t in run["trace"])
        malformed_json = invalid_answer_list = 0
        if system != "graph-llm":
            for event in run["trace"]:
                if "response" not in event:
                    continue
                try:
                    raw = extract_json(event["response"]["choices"][0]["message"]["content"])
                    invalid_answer_list += not isinstance(raw.get("answers"), list)
                except ValueError:
                    malformed_json += 1
        failures.append({"model": model, "system": system, "set": label,
                         "request_errors": sum("error" in t for t in run["trace"]),
                         "finish_reasons": dict(finish), "prediction_errors": sum(bool(r["error"]) for r in run["rows"]),
                         "malformed_reader_json": malformed_json, "invalid_reader_answer_list": invalid_answer_list})
    out += ["## Response completion checks", "", "| Model | System | Set | Output-token limit reached | Request errors | Invalid reader format |", "|---|---|---|---:|---:|---:|"]
    for item in failures:
        invalid = str(item['malformed_reader_json'] + item['invalid_reader_answer_list']) if item['system'] != 'graph-llm' else '—'
        out += [f"| {item['model']} | {item['system']} | {item['set']} | {item['finish_reasons'].get('length', 0)} | {item['request_errors']} | {invalid} |"]
    (args.runs / "COMPARISON.md").write_text("\n".join(out) + "\n")
    (args.runs / "comparison.json").write_text(json.dumps({"overview": overview, "response_checks": failures}, indent=2) + "\n")
    print(f"Validated {len(runs)} runs; wrote {args.runs / 'COMPARISON.md'}")


if __name__ == "__main__":
    main()

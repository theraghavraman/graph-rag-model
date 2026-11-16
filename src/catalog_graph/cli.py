"""Command line: build the catalog, inspect it, and run structured queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from . import answer, documents, evaluate, paraphrases, questions
from .linker import EntityLinker
from .llm import LLMClient
from .planner import LLMPlanner, RulePlanner
from .retrieval import BM25Retriever, EmbeddingRetriever, load_documents
from .systems import DocumentSystem, GraphSystem
from .generator import GeneratorConfig, generate
from .query import CatalogQuery
from .store import AmbiguousEntity, EntityNotFound, GraphStore


def _build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "catalog.duckdb"
    if db.exists():
        db.unlink()
    catalog = generate(GeneratorConfig(seed=args.seed))
    with GraphStore(str(db)) as store:
        store.load(catalog)
        if not store.is_acyclic():
            print("error: generated lineage contains a cycle", file=sys.stderr)
            return 1
        qs = questions.build_questions(catalog, store, per_group=args.per_group, seed=args.question_seed)
        held_out = paraphrases.build_paraphrased(catalog, store, per_group=max(1, args.per_group // 2))
        counts = store.counts()
    questions.write_jsonl(qs, out / "questions.jsonl")
    questions.write_jsonl(held_out, out / "questions_paraphrased.jsonl")
    docs = documents.build_documents(catalog)
    documents.write_jsonl(docs, out / "documents.jsonl")
    print(f"catalog   {db}")
    print("          " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print(f"questions {out / 'questions.jsonl'}  ({', '.join(f'{sum(1 for q in qs if q.group == g)} {g}' for g in questions.GROUPS)})")
    print(f"          {out / 'questions_paraphrased.jsonl'}  ({len(held_out)} questions with different wording, held out)")
    print(f"documents {out / 'documents.jsonl'}  ({len(docs)} text documents for the vector baseline)")
    return 0


def _print_result(result, show_paths: bool) -> None:
    if result.empty:
        print("(no results)")
        return
    for h in result.hits:
        print(f"{h.hops:>3}  {h.type:<9} {h.qualified_name}")
        if show_paths:
            print("       " + " -> ".join(h.path))
    print(f"{len(result.hits)} result(s)")


def _query(args: argparse.Namespace) -> int:
    try:
        if args.json:
            query = CatalogQuery.model_validate_json(args.json)
        else:
            fields = dict(kind=args.kind, entity=args.entity, entity_type=args.entity_type)
            if args.kind == "lineage":
                fields.update(direction=args.direction, target_type=args.target)
                if args.max_hops:
                    fields["max_hops"] = args.max_hops
            query = CatalogQuery(**fields)
    except ValidationError as e:
        print(f"invalid query: {e.errors()[0]['msg']}", file=sys.stderr)
        return 2
    try:
        with GraphStore(args.db, read_only=True) as store:
            result = store.execute(query)
    except (EntityNotFound, AmbiguousEntity, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.as_json:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result, args.paths)
    return 0


SYSTEMS = ("graph-rules", "graph-llm", "bm25-llm", "vector-llm")


def _system(name: str, store: GraphStore, data: Path):
    if name == "graph-rules":
        return GraphSystem(store, RulePlanner(store))
    client = LLMClient()
    if name == "graph-llm":
        return GraphSystem(store, LLMPlanner(store, client))
    docs = load_documents(data / "documents.jsonl")
    retriever = BM25Retriever(docs) if name == "bm25-llm" else EmbeddingRetriever(docs, client, cache=data / "embeddings.json")
    return DocumentSystem(store, retriever, client)


def _ask(args: argparse.Namespace) -> int:
    data = Path(args.data)
    with GraphStore(str(data / "catalog.duckdb"), read_only=True) as store:
        system = _system(args.system, store, data)
        pred = system.ask(args.question)
        if pred.query is not None:
            print("query    " + pred.query.model_dump_json(exclude_defaults=True))
        if pred.retrieved:
            print("read     " + ", ".join(pred.retrieved))
        if args.phrase and pred.query is not None and not pred.abstained:
            print(answer.phrase(LLMClient(), args.question, store.execute(pred.query)))
        else:
            print(pred.text or pred.error)
    return 1 if pred.abstained else 0


def _eval(args: argparse.Namespace) -> int:
    data = Path(args.data)
    qs = questions.read_jsonl(args.questions or data / "questions.jsonl")
    label = Path(args.questions).stem if args.questions else "questions"
    with GraphStore(str(data / "catalog.duckdb"), read_only=True) as store:
        if args.system == "linking":
            result = evaluate.linking_accuracy(EntityLinker(store), qs, store)
            print(json.dumps(result, indent=2))
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f"linking__{label}.json").write_text(json.dumps(result, indent=1))
            return 0
        if args.system == "retrieval-bound":
            result = evaluate.retrieval_bound(BM25Retriever(load_documents(data / "documents.jsonl")), qs)
            print(json.dumps(result, indent=2))
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f"retrieval-bound-bm25__{label}.json").write_text(json.dumps(result, indent=1))
            return 0
        system = _system(args.system, store, data)
        shown = {"n": 0}

        def progress(done: int, total: int) -> None:
            if done == total or done - shown["n"] >= 10:
                shown["n"] = done
                print(f"  {done}/{total}", file=sys.stderr)

        rows = evaluate.run(system, qs, store, progress)
    summary = evaluate.summarise(rows)
    client = getattr(getattr(system, "planner", None), "client", None) or getattr(system, "client", None)
    meta = {"system": args.system, "questions": label,
            "model": getattr(client, "model", None), "embed_model": getattr(client, "embed_model", None) if args.system == "vector-llm" else None}
    if client is not None:
        meta.update(model_calls=client.usage.calls, prompt_tokens=client.usage.prompt_tokens, completion_tokens=client.usage.completion_tokens)
    path = evaluate.save(rows, summary, args.out, f"{args.system}__{label}", meta)
    print(evaluate.markdown_table(summary, f"{args.system} on {label}" + (f" ({meta['model']})" if meta.get("model") else "")))
    print(f"\nsaved {path}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    with GraphStore(args.db, read_only=True) as store:
        for k, v in store.counts().items():
            print(f"{v:>6}  {k}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catalog-graph", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="generate the synthetic catalog, the question set and the baseline documents")
    b.add_argument("--out", default="data")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--question-seed", type=int, default=11)
    b.add_argument("--per-group", type=int, default=30)
    b.set_defaults(func=_build)

    q = sub.add_parser("query", help="run one structured query")
    q.add_argument("kind", nargs="?", choices=["owner", "members", "lineage"])
    q.add_argument("entity", nargs="?")
    q.add_argument("--db", default="data/catalog.duckdb")
    q.add_argument("--json", help="the whole query as JSON, as the planner would produce it")
    q.add_argument("--direction", choices=["upstream", "downstream"], default="downstream")
    q.add_argument("--target", choices=["system", "table", "column", "pipeline", "report", "owner"])
    q.add_argument("--entity-type", choices=["system", "table", "column", "pipeline", "report", "owner"])
    q.add_argument("--max-hops", type=int)
    q.add_argument("--paths", action="store_true", help="show the evidence path for every result")
    q.add_argument("--as-json", action="store_true")
    q.set_defaults(func=_query)

    s = sub.add_parser("stats", help="node and edge counts")
    s.add_argument("--db", default="data/catalog.duckdb")
    s.set_defaults(func=_stats)

    a = sub.add_parser("ask", help="answer one question in plain language")
    a.add_argument("question")
    a.add_argument("--system", choices=SYSTEMS, default="graph-rules")
    a.add_argument("--data", default="data")
    a.add_argument("--phrase", action="store_true", help="let the model word the answer (graph systems)")
    a.set_defaults(func=_ask)

    e = sub.add_parser("eval", help="score a system on a question set")
    e.add_argument("--system", choices=(*SYSTEMS, "linking", "retrieval-bound"), required=True)
    e.add_argument("--questions", help="question file (default: data/questions.jsonl)")
    e.add_argument("--data", default="data")
    e.add_argument("--out", default="results")
    e.set_defaults(func=_eval)

    args = parser.parse_args(argv)
    if args.command == "query" and not args.json and not (args.kind and args.entity):
        parser.error("query needs KIND and ENTITY, or --json")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

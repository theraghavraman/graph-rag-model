#!/usr/bin/env python3
"""Run the unchanged three systems on both supplied sets, retaining an audit trail.

Chat calls are never replayed. Embeddings are computed with the real endpoint once,
then cached verbatim to avoid swapping the chat and embedding models on small hosts.
No gold query or answer is sent to a model. Resume is bound to inputs and settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
import urllib.request

from catalog_graph import evaluate
from catalog_graph.llm import LLMClient
from catalog_graph.planner import LLMPlanner
from catalog_graph.questions import read_jsonl
from catalog_graph.retrieval import BM25Retriever, EmbeddingRetriever, load_documents
from catalog_graph.store import GraphStore
from catalog_graph.systems import DocumentSystem, GraphSystem

SYSTEMS = ("graph-llm", "bm25-llm", "vector-llm")
SETS = ("questions", "questions_paraphrased")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)
    except Exception as error:
        return {"unavailable": str(error)}


class RecordedClient(LLMClient):
    """Record actual requests/responses, excluding credentials and embedding arrays."""
    def __init__(self, trace: Path, vectors: dict[str, list[float]] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.trace, self.vectors, self.question_id = trace, vectors, None

    def _post(self, path, payload):
        started = time.perf_counter()
        event = {"at_utc": datetime.now(timezone.utc).isoformat(),
                 "question_id": self.question_id, "path": path, "request": payload}
        try:
            body = super()._post(path, payload)
            event["response"] = body if path != "/embeddings" else {
                "model": body.get("model"), "usage": body.get("usage"),
                "dimensions": [len(row["embedding"]) for row in body.get("data", [])],
                "vectors_saved_in": "embedding-cache.json",
            }
            return body
        except Exception as error:
            event["error"] = str(error)
            raise
        finally:
            event["seconds"] = time.perf_counter() - started
            with self.trace.open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def embed(self, texts, batch=64):
        if self.vectors is None:
            return super().embed(texts, batch=batch)
        # A missing exact text is an error; never substitute synthetic vectors.
        return [self.vectors[text] for text in texts]


class RecordedGraph(GraphSystem):
    def ask(self, question):
        self.last_prediction = super().ask(question)
        return self.last_prediction


class RecordedDocuments(DocumentSystem):
    def ask(self, question):
        self.last_prediction = super().ask(question)
        return self.last_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--embed-model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("CATALOG_LLM_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/model-runs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qs = {name: read_jsonl(args.data / f"{name}.jsonl") for name in SETS}
    docs = load_documents(args.data / "documents.jsonl")
    files = sorted((args.data / name for name in ("documents.jsonl", "questions.jsonl", "questions_paraphrased.jsonl", "catalog.duckdb")))
    sources = sorted((Path(__file__).resolve().parents[1] / "src/catalog_graph").glob("*.py"))
    config = {
        "base_url": args.base_url, "models": args.model, "embed_model": args.embed_model,
        "inputs_sha256": {p.name: digest(p.read_bytes()) for p in files},
        "source_sha256": {p.name: digest(p.read_bytes()) for p in sources},
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "temperature": 0.0, "planner_max_tokens": 200, "reader_max_tokens": 500,
        "retrieval_k": 8, "document_max_chars": 2500,
        "question_counts": {name: len(items) for name, items in qs.items()},
        "execution": "sequential questions and systems; cached real embeddings; no prompt tuning",
    }
    config_hash = digest(json.dumps(config, sort_keys=True).encode())
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_sha256"] != config_hash:
            raise SystemExit("Inputs or settings differ from this run; choose a new --out directory.")
    else:
        manifest = {"config_sha256": config_hash, "config": config,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in ("catalog-graph-rag", "duckdb", "pydantic")},
                    "endpoint_models_before": get_json(args.base_url + "/models"),
                    "endpoint_details_before": get_json(args.base_url.removesuffix("/v1") + "/api/v0/models")}
        write_json(manifest_path, manifest)

    # Compute vectors before chat calls: exact inputs used by EmbeddingRetriever.
    texts = list(dict.fromkeys([d["title"] + "\n" + d["text"] for d in docs] +
                              [q.question for items in qs.values() for q in items]))
    cache_path = args.out / "embedding-cache.json"
    cache_key = digest(json.dumps({"endpoint": args.base_url, "model": args.embed_model, "texts": texts}, sort_keys=True).encode())
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached["key"] != cache_key:
            raise SystemExit("Embedding cache does not match inputs/model.")
        vectors = cached["vectors"]
    else:
        embedding_client = RecordedClient(args.out / "embeddings-trace.jsonl", base_url=args.base_url,
            api_key=os.environ.get("CATALOG_LLM_API_KEY", "not-needed"), embed_model=args.embed_model, timeout=180)
        vectors = {}
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            vectors.update(zip(batch, embedding_client.embed(batch)))
            print(f"Embeddings {len(vectors)}/{len(texts)}", flush=True)
        write_json(cache_path, {"key": cache_key, "model": args.embed_model, "vectors": vectors,
                              "seconds": embedding_client.usage.seconds})
    lengths = {len(v) for v in vectors.values()}
    if len(vectors) != len(texts) or len(lengths) != 1 or 0 in lengths:
        raise SystemExit("Incomplete or inconsistent embedding cache.")

    with GraphStore(str(args.data / "catalog.duckdb"), read_only=True) as store:
        for model in args.model:
            model_dir = args.out / re.sub(r"[^A-Za-z0-9._-]", "_", model)
            model_dir.mkdir(parents=True, exist_ok=True)
            for name in SYSTEMS:
                for label, questions in qs.items():
                    stem = f"{name}__{label}"
                    checkpoint = model_dir / f"{stem}.checkpoint.jsonl"
                    completed = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
                    if [r["row"]["id"] for r in completed] != [q.id for q in questions[:len(completed)]]:
                        raise SystemExit(f"Checkpoint order differs: {checkpoint}")
                    client = RecordedClient(model_dir / f"{stem}.trace.jsonl", vectors=vectors,
                        base_url=args.base_url, api_key=os.environ.get("CATALOG_LLM_API_KEY", "not-needed"),
                        model=model, embed_model=args.embed_model, timeout=120)
                    if name == "graph-llm":
                        system = RecordedGraph(store, LLMPlanner(store, client))
                    else:
                        retriever = BM25Retriever(docs) if name == "bm25-llm" else EmbeddingRetriever(docs, client)
                        system = RecordedDocuments(store, retriever, client)
                    for index in range(len(completed), len(questions)):
                        q = questions[index]
                        client.question_id = q.id
                        before = client.usage.snapshot()
                        row = evaluate.run(system, [q], store)[0]
                        pred = system.last_prediction
                        after = client.usage.snapshot()
                        prediction = asdict(pred)
                        prediction["query"] = pred.query.model_dump() if pred.query is not None else None
                        item = {"row": asdict(row), "prediction": prediction,
                                "usage": {"model_calls": after[0] - before[0], "prompt_tokens": after[1] - before[1],
                                          "completion_tokens": after[2] - before[2]}}
                        with checkpoint.open("a") as stream:
                            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stream.flush()
                        completed.append(item)
                        if (index + 1) % 10 == 0 or index + 1 == len(questions):
                            print(f"{model} {stem} {index + 1}/{len(questions)} "
                                  f"exact={sum(x['row']['exact'] for x in completed)} "
                                  f"abstained={sum(x['row']['abstained'] for x in completed)}", flush=True)
                    rows = [evaluate.Row(**r["row"]) for r in completed]
                    meta = {"system": name, "questions": label, "model": model,
                            "embed_model": args.embed_model if name == "vector-llm" else None,
                            "config_sha256": config_hash, "temperature": 0, "k": 8,
                            "max_chars": 2500, "embeddings_cached": name == "vector-llm",
                            "trace": f"{stem}.trace.jsonl", "checkpoint": checkpoint.name}
                    meta.update({key: sum(r["usage"][key] for r in completed)
                                 for key in ("model_calls", "prompt_tokens", "completion_tokens")})
                    path = evaluate.save(rows, evaluate.summarise(rows), model_dir, stem, meta)
                    print(f"Saved {path}", flush=True)
        # Same real query vectors, no repeated embedding calls; match reader k as well.
        embed_client = RecordedClient(args.out / "embeddings-trace.jsonl", vectors=vectors, embed_model=args.embed_model)
        for label, questions in qs.items():
            for name, retriever in (("bm25", BM25Retriever(docs)), ("vector", EmbeddingRetriever(docs, embed_client))):
                bound = evaluate.retrieval_bound(retriever, questions, ks=(5, 8, 10, 20))
                write_json(args.out / f"retrieval-bound-{name}__{label}.json", bound)
    manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    endpoint_details_after=get_json(args.base_url.removesuffix("/v1") + "/api/v0/models"))
    write_json(manifest_path, manifest)
    print("All model runs and retrieval coverage measurements completed.", flush=True)


if __name__ == "__main__":
    main()

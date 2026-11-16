"""Document retrieval for the baseline: BM25 (no model) and embeddings (any OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from .llm import LLMClient

TOKEN = re.compile(r"[a-z0-9]+")


def load_documents(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: str) -> list[str]:
    """Lower-case words. Dotted and snake_case names are split, and also kept whole so exact names match strongly."""
    text = text.lower()
    whole = re.findall(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+|[a-z0-9]+(?:_[a-z0-9]+)+", text)
    return TOKEN.findall(text) + whole


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[dict]: ...


class BM25Retriever:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b = documents, k1, b
        self.terms = [Counter(tokenize(d["title"] + " " + d["text"])) for d in documents]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter = Counter()
        for t in self.terms:
            df.update(t.keys())
        n = len(documents)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = [t for t in tokenize(query) if t in self.idf]
        scores = []
        for i, terms in enumerate(self.terms):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average)
            score = sum(self.idf[t] * terms[t] * (self.k1 + 1) / (terms[t] + norm) for t in q if t in terms)
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda s: (-s[0], s[1]))
        return [{**self.documents[i], "score": round(score, 4)} for score, i in scores[:k]]


class EmbeddingRetriever:
    """Dense retrieval. Document vectors are computed once and cached on disk next to the documents."""

    def __init__(self, documents: list[dict], client: LLMClient, cache: str | Path | None = None):
        self.documents, self.client = documents, client
        cached = None
        if cache and Path(cache).exists():
            data = json.loads(Path(cache).read_text())
            if data.get("model") == client.embed_model and data.get("ids") == [d["id"] for d in documents]:
                cached = data["vectors"]
        vectors = cached or client.embed([d["title"] + "\n" + d["text"] for d in documents])
        if cache and cached is None:
            Path(cache).write_text(json.dumps({"model": client.embed_model, "ids": [d["id"] for d in documents], "vectors": vectors}))
        self.vectors = [self._unit(v) for v in vectors]

    @staticmethod
    def _unit(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self._unit(self.client.embed([query])[0])
        scores = sorted(((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self.vectors)), key=lambda s: (-s[0], s[1]))
        return [{**self.documents[i], "score": round(score, 4)} for score, i in scores[:k]]

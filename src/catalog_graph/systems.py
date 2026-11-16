"""The two systems being compared, behind one interface: question -> set of entity names.

GraphSystem      planner -> validated query -> graph traversal. The answer set comes from the graph.
DocumentSystem   retrieve text documents -> a language model reads them and lists the answer.
                 This is ordinary retrieval-augmented generation, the baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from . import answer
from .llm import LLMClient, LLMError, extract_json
from .planner import Plan, Planner
from .query import CatalogQuery
from .retrieval import Retriever
from .store import GraphStore


@dataclass
class Prediction:
    answers: list[str]                     # qualified names; empty means "nothing" or "abstained"
    abstained: bool = False
    text: str = ""
    query: Optional[CatalogQuery] = None   # graph system only
    retrieved: list[str] = field(default_factory=list)   # document system only
    seconds: float = 0.0
    error: str = ""


class GraphSystem:
    name = "graph"

    def __init__(self, store: GraphStore, planner: Planner):
        self.store, self.planner = store, planner

    def ask(self, question: str) -> Prediction:
        started = time.perf_counter()
        plan: Plan = self.planner.plan(question)
        if plan.query is None:
            return Prediction([], abstained=True, text=answer.abstain(plan), seconds=time.perf_counter() - started)
        try:
            result = self.store.execute(plan.query)
        except (LookupError, ValueError) as e:      # a valid shape that does not fit the catalog
            return Prediction([], abstained=True, text=f"I could not run that lookup: {e}", query=plan.query, seconds=time.perf_counter() - started, error=str(e))
        return Prediction(result.names, text=answer.render(result), query=plan.query, seconds=time.perf_counter() - started)


READER_PROMPT = """You answer questions about a data catalog using ONLY the documents provided.

Reply with one JSON object: {"answers": ["...", "..."]}
List every entity that answers the question, using exact names from the documents:
qualified names for tables, columns and pipelines (for example core.dim_customer.email),
the rpt_ identifier for reports, the system name for source systems, and the full name for people.
If the documents do not show any answer, reply {"answers": []}. Do not guess."""


class DocumentSystem:
    name = "documents"

    def __init__(self, store: GraphStore, retriever: Retriever, client: LLMClient, k: int = 8, max_chars: int = 2500):
        self.store, self.retriever, self.client, self.k, self.max_chars = store, retriever, client, k, max_chars

    def ask(self, question: str) -> Prediction:
        started = time.perf_counter()
        docs = self.retriever.search(question, k=self.k)
        context = "\n\n".join(f"[{i + 1}] {d['text'][:self.max_chars]}" for i, d in enumerate(docs))
        titles = [d["title"] for d in docs]
        try:
            reply = self.client.chat([{"role": "system", "content": READER_PROMPT},
                                      {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}], max_tokens=500)
            raw = extract_json(reply).get("answers", [])
        except (LLMError, ValueError) as e:
            return Prediction([], abstained=True, retrieved=titles, seconds=time.perf_counter() - started, error=str(e))
        names = self._normalise(raw if isinstance(raw, list) else [])
        return Prediction(names, text=", ".join(names) or "Nothing found.", retrieved=titles, seconds=time.perf_counter() - started)

    def _normalise(self, raw: list) -> list[str]:
        """Map what the model wrote onto catalog names where possible, so "Churn Watchlist" scores as rpt_churn_watchlist."""
        out: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                continue
            try:
                node_id, _ = self.store.resolve(item.strip())
                name = self.store.con.execute("SELECT qualified_name FROM nodes WHERE id = ?", [node_id]).fetchone()[0]
            except LookupError:
                name = item.strip()      # kept as written; it will count as a wrong answer
            if name not in out:
                out.append(name)
        return out

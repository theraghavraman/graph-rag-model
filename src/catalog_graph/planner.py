"""Planning: question -> CatalogQuery.

Two planners behind one interface:

RulePlanner   keyword rules and the entity linker. No model, instant, and a baseline:
              if rules do as well as a language model, the model has not earned its place.
LLMPlanner    a language model fills in the query structure, choosing the entity from
              the linker's candidates. Its output is validated; an invalid plan gets one
              corrective retry, and after that the planner abstains.

Neither planner can produce anything but a CatalogQuery, so neither can run arbitrary
SQL. A planner that cannot decide returns no query, and the system answers "I could not
work out what to look up" - it does not guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from pydantic import ValidationError

from .linker import Candidate, EntityLinker
from .llm import LLMClient, LLMError, extract_json
from .query import CatalogQuery
from .store import GraphStore


@dataclass
class Plan:
    query: Optional[CatalogQuery]
    reason: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    attempts: int = 1


class Planner(Protocol):
    def plan(self, question: str) -> Plan: ...


# (pattern, lineage fields, entity types the subject may have - most likely first)
_UP, _DOWN = "upstream", "downstream"
RULES: list[tuple[str, dict, tuple[str, ...]]] = [
    (r"\b(which|list)\b.*\btables\b.*\b(come from|delivered by)\b", dict(kind="members"), ("system",)),
    (r"\b(what|list)\b.*\bcolumns\b\s+(does|of)\b", dict(kind="members"), ("table",)),
    (r"\bwho should be told\b|\bowners?\b.*\baffected\b", dict(kind="lineage", direction=_DOWN, target_type="owner"), ("column", "table")),
    (r"^\s*who\s+(owns|maintains|is responsible|do i contact)\b|\bwhich person owns\b", dict(kind="owner"), ("column", "report", "pipeline", "table")),
    (r"\bsource systems?\b.*\bfeed\b|\boriginally come from\b", dict(kind="lineage", direction=_UP, target_type="system"), ("report", "column", "table")),
    (r"\bsource-system columns\b|\braw source columns\b|\boriginate from\b", dict(kind="lineage", direction=_UP, target_type="column", target_layer="raw"), ("column", "report")),
    (r"\bhave to succeed\b|\bdepend on, end to end\b", dict(kind="lineage", direction=_UP, target_type="pipeline"), ("report", "column", "table")),
    (r"\bread from\b|\binput tables of\b", dict(kind="lineage", direction=_UP, target_type="table", max_hops=1), ("pipeline",)),
    (r"\bcomputed from directly\b|\bimmediate inputs of\b", dict(kind="lineage", direction=_UP, target_type="column", max_hops=1), ("column",)),
    (r"\bpipelines read\b.*\bdirectly\b|\bas a direct input\b", dict(kind="lineage", direction=_DOWN, target_type="pipeline", max_hops=1), ("column",)),
    (r"\bcomputed directly from\b|\bderived straight from\b", dict(kind="lineage", direction=_DOWN, target_type="column", max_hops=1), ("column",)),
    (r"\b(reports|dashboards)\s+(display|show)\b", dict(kind="lineage", direction=_DOWN, target_type="report", max_hops=1), ("column",)),
    (r"\btables\b.*\b(are built|sit downstream)\b", dict(kind="lineage", direction=_DOWN, target_type="table"), ("table", "column")),
    (r"\banything downstream read\b|\bpipelines or reports use\b", dict(kind="lineage", direction=_DOWN), ("column", "table")),
    (r"\b(reports|dashboards)\b.*\b(depend|affected|ultimately use|would break)\b|\bused by any report\b", dict(kind="lineage", direction=_DOWN, target_type="report"), ("column", "table", "pipeline")),
]


class RulePlanner:
    def __init__(self, store: GraphStore, linker: EntityLinker | None = None):
        self.linker = linker or EntityLinker(store)

    def plan(self, question: str) -> Plan:
        candidates = self.linker.link(question)
        for pattern, fields, subject_types in RULES:
            if re.search(pattern, question, re.I):
                subject = next((c for t in subject_types for c in candidates if c.type == t), None)
                if subject is None:
                    return Plan(None, "understood the question but could not find the entity it is about", candidates)
                extra = {"entity_type": subject.type} if subject.type == "system" else {}
                return Plan(CatalogQuery(entity=subject.qualified_name, **fields, **extra), f"rule: {pattern}", candidates)
        return Plan(None, "no rule matches this wording", candidates)


SYSTEM_PROMPT = """You translate a question about a data catalog into ONE JSON object. Reply with the JSON only.

The catalog has systems, tables, columns, pipelines, reports and owners. Data flows
source system -> raw table -> staging -> core -> mart -> report.

Fields:
  "kind"         "owner"   who owns the entity
                 "members" what the entity contains (tables of a system, columns of a table)
                 "lineage" what the entity feeds or is built from
  "entity"       copy the qualified_name of ONE candidate exactly. The subject of the question.
  "entity_type"  only when the entity is a system: "system"
Lineage only:
  "direction"    "downstream" (what depends on it, what it feeds, impact of a change)
                 "upstream"   (what it is built from, where it comes from, what must run first)
  "target_type"  what the answer should list: "report" | "table" | "column" | "pipeline" | "system" | "owner".
                 Omit it to list everything.
  "target_layer" only with target_type table/column: "raw" | "staging" | "core" | "mart".
                 Use "raw" when the question asks for original source columns or tables.
  "max_hops"     1 when the question says directly / immediate / one step / as-is. Otherwise omit.

Examples:
  Who owns the core.dim_customer table?
  {"kind": "owner", "entity": "core.dim_customer"}
  Which reports break if crm.customers.email is removed?
  {"kind": "lineage", "entity": "crm.customers.email", "direction": "downstream", "target_type": "report"}
  Which jobs read stg.orders.amount as a direct input?
  {"kind": "lineage", "entity": "stg.orders.amount", "direction": "downstream", "target_type": "pipeline", "max_hops": 1}
  Which source systems are behind the Revenue Overview report?
  {"kind": "lineage", "entity": "rpt_revenue_overview", "direction": "upstream", "target_type": "system"}
  Who needs to know before stg.orders.amount changes?
  {"kind": "lineage", "entity": "stg.orders.amount", "direction": "downstream", "target_type": "owner"}

If none of the candidates is what the question is about, reply {"kind": "none"}."""


class LLMPlanner:
    def __init__(self, store: GraphStore, client: LLMClient, linker: EntityLinker | None = None, max_candidates: int = 6):
        self.store, self.client = store, client
        self.linker = linker or EntityLinker(store)
        self.max_candidates = max_candidates

    def plan(self, question: str) -> Plan:
        candidates = self.linker.link(question, limit=self.max_candidates)
        if not candidates:
            return Plan(None, "no catalog entity found in the question", candidates, attempts=0)
        listing = "\n".join(f"  {c.type:<9} {c.qualified_name}" for c in candidates)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Candidates:\n{listing}\n\nQuestion: {question}"}]
        allowed = {c.qualified_name.lower(): c for c in candidates}
        problem = ""
        for attempt in (1, 2):
            try:
                reply = self.client.chat(messages, max_tokens=200)
            except LLMError as e:
                return Plan(None, f"model call failed: {e}", candidates, attempt)
            try:
                raw = extract_json(reply)
                if raw.get("kind") == "none":
                    return Plan(None, "the model found no matching entity", candidates, attempt)
                query = CatalogQuery(**raw)
                chosen = allowed.get(query.entity.lower())
                if chosen is None:
                    raise ValueError(f'"entity" must be one of the candidates, copied exactly; got {query.entity!r}')
                if chosen.type == "system" and query.entity_type is None:
                    query = query.model_copy(update={"entity_type": "system"})
                return Plan(query, "model", candidates, attempt)
            except ValidationError as e:
                problem = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'query'}: {err['msg']}" for err in e.errors()[:3])
            except (ValueError, TypeError) as e:
                problem = str(e)
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"That was not a valid query ({problem}). Reply with the corrected JSON object only."}]
        return Plan(None, f"the model did not produce a valid query: {problem}", candidates, 2)


def same_query(a: CatalogQuery, b: CatalogQuery, store: GraphStore) -> bool:
    """Equal after resolving the entity, so 'Churn Watchlist' and 'rpt_churn_watchlist' count as the same plan."""
    def key(q: CatalogQuery):
        try:
            entity = store.resolve(q.entity, q.entity_type)[0]
        except LookupError:
            entity = q.entity.lower()
        return (q.kind, entity, q.direction, q.target_type, q.target_layer, q.max_hops)
    return key(a) == key(b)


def plan_to_json(plan: Plan) -> str:
    return json.dumps(plan.query.model_dump(exclude_defaults=True) if plan.query else None)

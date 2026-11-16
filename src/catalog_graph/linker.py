"""Entity linking: find which catalog entities a question is talking about.

Three passes, cheapest first, all without a model:

1. exact      a qualified name or display name appears in the question
2. composed   a table is mentioned and one of its column names appears nearby
              ("the email column in crm.customers", "crm customers email")
3. fuzzy      a near-miss spelling of a qualified name (difflib)

The result is a ranked list of candidates, not a single answer: the planner (rules or a
language model) chooses among them, and an embedding pass can be added behind the same
interface. Ranking prefers the most specific entity, so a column outranks the table
that contains it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .store import GraphStore

SPECIFICITY = {"column": 0, "report": 1, "pipeline": 2, "table": 3, "system": 4, "owner": 5}
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


@dataclass(frozen=True)
class Candidate:
    id: str
    type: str
    qualified_name: str
    method: str      # exact | composed | fuzzy
    score: float


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class EntityLinker:
    def __init__(self, store: GraphStore):
        rows = store.con.execute("SELECT id, type, qualified_name, name FROM nodes").fetchall()
        self.by_qname: dict[str, tuple[str, str, str]] = {}
        self.display: list[tuple[str, tuple[str, str, str]]] = []      # normalised display name -> node, longest first
        self.tables: dict[str, tuple[str, str, str]] = {}               # normalised "schema table" -> node
        self.columns_of: dict[str, list[tuple[str, tuple[str, str, str]]]] = {}
        for id_, type_, qname, name in rows:
            node = (id_, type_, qname)
            self.by_qname[qname.lower()] = node
            if type_ in ("report", "owner"):
                self.display.append((_norm(name), node))
            if type_ == "table":
                self.tables[_norm(qname)] = node
                self.columns_of.setdefault(qname, [])
        for id_, type_, qname, name in rows:
            if type_ == "column":
                self.columns_of[qname.rsplit(".", 1)[0]].append((_norm(name), (id_, type_, qname)))
        self.display.sort(key=lambda kv: -len(kv[0]))
        self.systems = {q for q, n in self.by_qname.items() if n[1] == "system"}

    def link(self, question: str, limit: int = 8) -> list[Candidate]:
        found: dict[str, Candidate] = {}

        def add(node: tuple[str, str, str], method: str, score: float) -> None:
            if node[0] not in found or score > found[node[0]].score:
                found[node[0]] = Candidate(node[0], node[1], node[2], method, score)

        text = _norm(question)
        padded = f" {text} "
        tokens = [t.rstrip(".") for t in WORD.findall(question)]

        # 1. exact qualified names and display names
        for tok in tokens:
            node = self.by_qname.get(tok.lower())
            if node and (node[1] != "system" or self._system_context(tok, question)):
                add(node, "exact", 1.0)
        for name, node in self.display:
            # Match the title as separate words only: "supplier scorecard" must not fire on mart.supplier_scorecard.
            if name and re.search(r"(?<![\w.])" + r"\s+".join(map(re.escape, name.split())) + r"(?![\w]|\.\w)", question, re.I):
                add(node, "exact", 1.0)

        # 2. tables written with spaces, and columns composed from a table plus a column name
        mentioned_tables = [n for n in found.values() if n.type == "table"]
        spaced = [(norm, node) for norm, node in self.tables.items() if f" {norm} " in padded]
        for norm, node in spaced:
            # "support subscriptions" inside "stg support subscriptions" is the staging table, not the raw one.
            if node[0] not in found and not any(norm != other and f" {norm} " in f" {other} " for other, _ in spaced):
                add(node, "composed", 0.9)
                mentioned_tables.append(Candidate(node[0], node[1], node[2], "composed", 0.9))
        for table in {t.qualified_name: t for t in mentioned_tables}.values():
            matches = [(len(cname), node) for cname, node in self.columns_of[table.qualified_name] if f" {cname} " in padded]
            if matches:
                longest = max(m[0] for m in matches)      # "total amount" beats "amount"
                for size, node in matches:
                    if size == longest and node[0] not in found:
                        add(node, "composed", 0.95)

        # 3. near-miss spellings: a dotted name that matched nothing exactly is probably a typo
        for tok in tokens:
            if "." in tok and tok.lower() not in self.by_qname:
                for close in difflib.get_close_matches(tok.lower(), self.by_qname, n=3, cutoff=0.86):
                    add(self.by_qname[close], "fuzzy", round(difflib.SequenceMatcher(None, tok.lower(), close).ratio(), 3))

        # Confident matches first, then the most specific entity: a column outranks the table that holds it.
        ranked = sorted(found.values(), key=lambda c: (c.score < 0.9, SPECIFICITY[c.type], -c.score, -len(c.qualified_name), c.qualified_name))
        return ranked[:limit]

    @staticmethod
    def _system_context(token: str, question: str) -> bool:
        """A bare word like "web" names a system only when the sentence says so."""
        return bool(re.search(rf"\b{re.escape(token)}\b\s+(source\s+)?system|system\s+{re.escape(token)}\b|(delivered by|come from|from)\s+(the\s+)?{re.escape(token)}\b(?!\.\w)", question, re.I))

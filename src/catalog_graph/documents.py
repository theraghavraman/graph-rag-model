"""The same catalog as plain text documents, for the vector-retrieval baseline (week 2).

Fairness matters here: every fact the graph holds is also written into these documents.
Pipeline documents spell out each column mapping, and report documents list every
column they show. If vector retrieval then loses on multi-hop questions, it is because
the answer needs several documents chained together, not because facts were withheld.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .model import Catalog


def build_documents(catalog: Catalog) -> list[dict]:
    by_id = {n.id: n for n in catalog.nodes}
    owner_of: dict[str, str] = {}
    children: dict[str, list[str]] = defaultdict(list)
    mappings: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))   # pipeline -> out col -> in cols
    shown: dict[str, list[str]] = defaultdict(list)
    for e in catalog.edges:
        if e.type == "OWNS":
            owner_of[e.dst] = by_id[e.src].name
        elif e.type == "CONTAINS":
            children[e.src].append(e.dst)
        elif e.type == "FLOWS" and e.via:
            mappings[e.via][by_id[e.dst].qualified_name].append(by_id[e.src].qualified_name)
        elif e.type == "FLOWS":
            shown[e.dst].append(by_id[e.src].qualified_name)

    docs: list[dict] = []
    for n in catalog.nodes:
        if n.type == "system":
            tables = ", ".join(by_id[c].qualified_name for c in children[n.id])
            text = f"Source system {n.name}. {n.description} Tables: {tables}."
        elif n.type == "table":
            cols = ", ".join(f"{by_id[c].name} ({by_id[c].attrs.get('data_type', 'unknown type')})" for c in children[n.id])
            text = f"Table {n.qualified_name} ({n.attrs.get('layer', 'unknown')} layer). {n.description} Owner: {owner_of.get(n.id, 'unassigned')}. Columns: {cols}."
        elif n.type == "pipeline":
            lines = "; ".join(f"{out} is computed from {', '.join(ins)}" for out, ins in mappings[n.id].items())
            text = f"Pipeline {n.name}. {n.description} Schedule: {n.attrs.get('schedule', 'not set')}. Owner: {owner_of.get(n.id, 'unassigned')}. Column mappings: {lines}."
        elif n.type == "report":
            text = f"Report {n.name} ({n.qualified_name}), built in {n.attrs.get('tool', 'an unknown tool')}, refreshed {n.attrs.get('refresh', 'on no fixed schedule')}. {n.description} Owner: {owner_of.get(n.id, 'unassigned')}. Columns shown: {', '.join(shown[n.id])}."
        else:
            continue      # columns live inside their table document; owners inside what they own
        docs.append({"id": n.id, "type": n.type, "title": n.qualified_name, "text": text})
    return docs


def write_jsonl(docs: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

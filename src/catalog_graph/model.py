"""The catalog as a property graph: typed nodes and typed, directed edges.

Node ids are readable and stable: ``<type>:<qualified name>``, for example
``column:crm.customers.email``. Three edge types exist:

CONTAINS  structure        system -> table, table -> column
FLOWS     data movement    column -> column (``via`` names the pipeline that computes
                           the target from the source), column -> report (displayed)
OWNS      responsibility   owner -> table | pipeline | report

Lineage is reachability over FLOWS edges only, in the direction data moves.

Why column -> column and not column -> pipeline -> column: routing lineage through a
pipeline node loses which input produced which output, so every input of a pipeline
would appear to feed every one of its outputs. Keeping the pipeline as an attribute of
the edge keeps column lineage exact and still answers pipeline-level questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

NODE_TYPES: tuple[str, ...] = ("system", "table", "column", "pipeline", "report", "owner")
EDGE_TYPES: tuple[str, ...] = ("CONTAINS", "FLOWS", "OWNS")


def node_id(node_type: str, qualified_name: str) -> str:
    return f"{node_type}:{qualified_name}"


@dataclass(frozen=True)
class Node:
    type: str
    qualified_name: str
    name: str
    description: str = ""
    attrs: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def id(self) -> str:
        return node_id(self.type, self.qualified_name)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: str
    via: Optional[str] = None      # FLOWS only: id of the pipeline that implements this step


@dataclass
class Catalog:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    _seen: set = field(default_factory=set, repr=False, compare=False)

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def link(self, src: Node | str, dst: Node | str, edge_type: str, via: Node | str | None = None) -> None:
        s = src if isinstance(src, str) else src.id
        d = dst if isinstance(dst, str) else dst.id
        v = via if via is None or isinstance(via, str) else via.id
        edge = Edge(s, d, edge_type, v)
        if edge not in self._seen:
            self._seen.add(edge)
            self.edges.append(edge)

    def of_type(self, node_type: str) -> list[Node]:
        return [n for n in self.nodes if n.type == node_type]

    def counts(self) -> dict[str, int]:
        out = {t: 0 for t in NODE_TYPES}
        for n in self.nodes:
            out[n.type] += 1
        for t in EDGE_TYPES:
            out[t] = sum(1 for e in self.edges if e.type == t)
        return out

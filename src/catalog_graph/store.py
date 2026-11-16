"""The catalog graph in DuckDB: two tables, and lineage as a recursive query.

No graph database is needed at this size. Reachability runs in SQL; the evidence
path for each result is rebuilt in Python from the hop counts, so every answer can
show how it was reached, not only what was reached.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

import duckdb

from .model import Catalog, node_id
from .query import CatalogQuery, Hit, QueryResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id             VARCHAR PRIMARY KEY,
    type           VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,
    name           VARCHAR NOT NULL,
    description    VARCHAR NOT NULL DEFAULT '',
    attrs          JSON    NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS edges (
    src  VARCHAR NOT NULL,
    dst  VARCHAR NOT NULL,
    type VARCHAR NOT NULL,
    via  VARCHAR,               -- FLOWS only: pipeline that implements this step
    PRIMARY KEY (src, dst, type)
);
CREATE INDEX IF NOT EXISTS edges_src ON edges (type, src);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (type, dst);
CREATE INDEX IF NOT EXISTS nodes_qname ON nodes (qualified_name);
"""

# Reachability over FLOWS edges. {here}/{there} are swapped to walk upstream.
REACH = """
WITH RECURSIVE reach(id, hops) AS (
    SELECT s.id, 0 FROM (SELECT unnest(?::VARCHAR[]) AS id) AS s
    UNION
    SELECT e.{there}, r.hops + 1
    FROM reach AS r JOIN edges AS e ON e.{here} = r.id AND e.type = 'FLOWS'
    WHERE r.hops < ?
)
SELECT id, min(hops) AS hops FROM reach GROUP BY id
"""


class EntityNotFound(LookupError):
    """No node has this name."""


class AmbiguousEntity(LookupError):
    """More than one node has this name; set entity_type or use the qualified name."""

    def __init__(self, entity: str, candidates: list[str]):
        super().__init__(f"{entity!r} matches {len(candidates)} nodes: {', '.join(candidates[:8])}")
        self.candidates = candidates


class GraphStore:
    def __init__(self, path: str = ":memory:", read_only: bool = False):
        self.con = duckdb.connect(path, read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- loading -----------------------------------------------------------------------------
    def load(self, catalog: Catalog, replace: bool = True) -> None:
        ids = {n.id for n in catalog.nodes}
        dangling = [e for e in catalog.edges if e.src not in ids or e.dst not in ids or (e.via is not None and e.via not in ids)]
        if dangling:
            raise ValueError(f"{len(dangling)} edges reference unknown nodes, e.g. {dangling[0]}")
        self.con.execute("BEGIN")
        try:
            if replace:
                self.con.execute("DELETE FROM edges")
                self.con.execute("DELETE FROM nodes")
            self.con.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                                 [(n.id, n.type, n.qualified_name, n.name, n.description, json.dumps(n.attrs, sort_keys=True)) for n in catalog.nodes])
            self.con.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)", [(e.src, e.dst, e.type, e.via) for e in catalog.edges])
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def counts(self) -> dict[str, int]:
        out = dict(self.con.execute("SELECT type, count(*) FROM nodes GROUP BY type ORDER BY type").fetchall())
        out.update(self.con.execute("SELECT type, count(*) FROM edges GROUP BY type ORDER BY type").fetchall())
        return out

    # ---- entity resolution -------------------------------------------------------------------
    def resolve(self, entity: str, entity_type: Optional[str] = None) -> tuple[str, str]:
        """Return (node id, node type) for an exact id, qualified name or display name. Case-insensitive.

        This is exact matching only. Fuzzy and embedding-based linking belong to the planner.
        """
        text = entity.strip()
        rows = self.con.execute(
            """SELECT id, type, CASE WHEN id = ? THEN 0 WHEN lower(qualified_name) = lower(?) THEN 1 ELSE 2 END AS rank
               FROM nodes
               WHERE (id = ? OR lower(qualified_name) = lower(?) OR lower(name) = lower(?)) AND (? IS NULL OR type = ?)
               ORDER BY rank, id""",
            [text, text, text, text, text, entity_type, entity_type],
        ).fetchall()
        if not rows:
            raise EntityNotFound(f"no {entity_type or 'catalog entity'} named {entity!r}")
        best = [r for r in rows if r[2] == rows[0][2]]
        if len(best) > 1:
            raise AmbiguousEntity(entity, [r[0] for r in best])
        return best[0][0], best[0][1]

    def _nodes(self, ids: Iterable[str]) -> dict[str, tuple[str, str, str]]:
        ids = list(ids)
        if not ids:
            return {}
        rows = self.con.execute("SELECT id, type, qualified_name, name FROM nodes WHERE id IN (SELECT unnest(?::VARCHAR[]))", [ids]).fetchall()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    def _children(self, parent: str) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT dst FROM edges WHERE type = 'CONTAINS' AND src = ? ORDER BY dst", [parent]).fetchall()]

    def _parents(self, children: Iterable[str], edge_type: str) -> list[tuple[str, str]]:
        children = list(children)
        if not children:
            return []
        return self.con.execute("SELECT src, dst FROM edges WHERE type = ? AND dst IN (SELECT unnest(?::VARCHAR[])) ORDER BY src, dst", [edge_type, children]).fetchall()

    def _hits(self, scored: dict[str, tuple[int, list[str]]]) -> list[Hit]:
        """scored: node id -> (hops, path as node ids)."""
        info = self._nodes(set(scored) | {p for _, path in scored.values() for p in path})
        hits = [Hit(id=i, type=info[i][0], qualified_name=info[i][1], name=info[i][2], hops=h, path=[info[p][1] for p in path])
                for i, (h, path) in scored.items()]
        return sorted(hits, key=lambda h: (h.hops, h.type, h.qualified_name))

    # ---- queries -----------------------------------------------------------------------------
    def execute(self, query: CatalogQuery) -> QueryResult:
        start_id, start_type = self.resolve(query.entity, query.entity_type)
        if query.kind == "owner":
            return self._owner(query, start_id, start_type)
        if query.kind == "members":
            if start_type not in ("system", "table"):
                raise ValueError(f"only a system or a table has members, not a {start_type}")
            return QueryResult(query=query, start=[start_id], hits=self._hits({c: (1, [start_id, c]) for c in self._children(start_id)}))
        return self._lineage(query, start_id, start_type)

    def _owner(self, query: CatalogQuery, start_id: str, start_type: str) -> QueryResult:
        target = start_id
        if start_type == "column":      # a column is owned through its table
            target = self._parents([start_id], "CONTAINS")[0][0]
        elif start_type in ("system", "owner"):
            raise ValueError(f"a {start_type} has no owner in this catalog")
        owners = self._parents([target], "OWNS")
        path = [start_id] if target == start_id else [start_id, target]
        return QueryResult(query=query, start=[start_id], hits=self._hits({o: (1, path + [o]) for o, _ in owners}))

    def _lineage(self, query: CatalogQuery, start_id: str, start_type: str) -> QueryResult:
        down = query.direction == "downstream"
        here, there = ("src", "dst") if down else ("dst", "src")
        offset = 0                       # hops already spent before the first traversed edge
        prefix: list[str] = []           # evidence shown before the first traversed node
        if start_type == "table":
            starts = self._children(start_id)
        elif start_type in ("column", "report"):
            starts = [start_id]
        elif start_type == "system":     # every column of every table in the system
            starts = [c for t in self._children(start_id) for c in self._children(t)]
        elif start_type == "pipeline":   # the step the pipeline itself performs counts as hop 1
            starts = [r[0] for r in self.con.execute(f"SELECT DISTINCT {there} FROM edges WHERE type = 'FLOWS' AND via = ? ORDER BY 1", [start_id]).fetchall()]
            offset, prefix = 1, [start_id]
        else:
            raise ValueError("lineage starts from a column, table, pipeline, report or system")
        budget = query.max_hops - offset
        hops = dict(self.con.execute(REACH.format(here=here, there=there), [starts, max(budget, 0)]).fetchall()) if budget >= 0 else {}
        paths = {n: prefix + p for n, p in self._paths(hops, here, there).items()}
        # A pipeline start has already taken one step, so the columns it touches are results.
        reached = {i: h + offset for i, h in hops.items() if h + offset > 0}
        info = self._nodes(reached)
        target = query.target_type
        scored: dict[str, tuple[int, list[str]]] = {}

        def keep(node: str, h: int, path: list[str]) -> None:
            if node not in scored or (h, path) < scored[node]:
                scored[node] = (h, path)

        # Pipelines are attributes of the traversed steps: a step a -> b performed at hop hops[a] + 1.
        pipelines: dict[str, tuple[int, list[str]]] = {}
        if hops and target in (None, "pipeline", "owner"):
            ids = list(hops)
            steps = self.con.execute(
                f"""SELECT {here}, via FROM edges
                    WHERE type = 'FLOWS' AND via IS NOT NULL AND {here} IN (SELECT unnest(?::VARCHAR[])) AND {there} IN (SELECT unnest(?::VARCHAR[]))
                    ORDER BY via, {here}""", [ids, ids]).fetchall()
            for a, via in steps:
                h = hops[a] + offset + 1
                if h <= query.max_hops and via != start_id and (via not in pipelines or (h, paths[a] + [via]) < pipelines[via]):
                    pipelines[via] = (h, paths[a] + [via])

        own_tables = set() if start_type in ("pipeline", "report") else {p for p, _ in self._parents(starts, "CONTAINS")}
        col_ids = [i for i in reached if info[i][0] == "column"]
        tables: dict[str, str] = {}      # table id -> nearest reached column in it
        if target in ("table", "system", "owner"):
            for t, c in self._parents(col_ids, "CONTAINS"):
                if t not in own_tables and t != start_id and (t not in tables or (reached[c], c) < (reached[tables[t]], tables[t])):
                    tables[t] = c

        if target in (None, "column", "report"):
            for i, h in reached.items():
                if target is None or info[i][0] == target:
                    keep(i, h, paths[i])
        if target in (None, "pipeline"):
            for p, (h, path) in pipelines.items():
                keep(p, h, path)
        if target == "table":
            for t, c in tables.items():
                keep(t, reached[c], paths[c] + [t])
        elif target == "system":
            for system, t in self._parents(tables, "CONTAINS"):
                if system != start_id:
                    c = tables[t]
                    keep(system, reached[c], paths[c] + [t, system])
        elif target == "owner":          # people responsible for anything the traversal touched
            owned: dict[str, tuple[int, list[str]]] = dict(pipelines)
            owned.update({i: (h, paths[i]) for i, h in reached.items() if info[i][0] == "report"})
            owned.update({t: (reached[c], paths[c] + [t]) for t, c in tables.items()})
            for owner, thing in self._parents(owned, "OWNS"):
                h, path = owned[thing]
                keep(owner, h, path + [owner])
        if query.target_layer is not None:
            in_layer = self._in_layer(scored, query.target_layer)
            scored = {i: v for i, v in scored.items() if i in in_layer}
        return QueryResult(query=query, start=starts, hits=self._hits(scored))

    def _in_layer(self, ids: Iterable[str], layer: str) -> set[str]:
        """Tables in the layer, and columns whose table is in the layer."""
        ids = list(ids)
        if not ids:
            return set()
        rows = self.con.execute(
            """SELECT n.id FROM nodes AS n
               LEFT JOIN nodes AS t ON n.type = 'column' AND t.type = 'table' AND t.qualified_name = json_extract_string(n.attrs, '$.table')
               WHERE n.id IN (SELECT unnest(?::VARCHAR[]))
                 AND json_extract_string(CASE WHEN n.type = 'table' THEN n.attrs ELSE t.attrs END, '$.layer') = ?""",
            [ids, layer]).fetchall()
        return {r[0] for r in rows}

    def _paths(self, hops: dict[str, int], here: str, there: str) -> dict[str, list[str]]:
        """One shortest path per reached node, chosen deterministically (smallest predecessor id)."""
        if not hops:
            return {}
        ids = list(hops)
        edges = self.con.execute(
            f"""SELECT {here}, {there} FROM edges
                WHERE type = 'FLOWS' AND {here} IN (SELECT unnest(?::VARCHAR[])) AND {there} IN (SELECT unnest(?::VARCHAR[]))""",
            [ids, ids],
        ).fetchall()
        pred: dict[str, str] = {}
        for a, b in edges:
            if hops[b] == hops[a] + 1 and (b not in pred or a < pred[b]):
                pred[b] = a
        paths: dict[str, list[str]] = {}
        for node in sorted(hops, key=lambda i: hops[i]):
            paths[node] = [node] if hops[node] == 0 else paths[pred[node]] + [node]
        return paths

    # ---- convenience -------------------------------------------------------------------------
    def lineage(self, entity: str, direction: str = "downstream", target_type: Optional[str] = None, max_hops: Optional[int] = None,
                entity_type: Optional[str] = None, target_layer: Optional[str] = None) -> QueryResult:
        fields = dict(kind="lineage", entity=entity, direction=direction, target_type=target_type, entity_type=entity_type, target_layer=target_layer)
        if max_hops is not None:
            fields["max_hops"] = max_hops
        return self.execute(CatalogQuery(**fields))

    def has_edge(self, src_qname_or_id: str, dst_qname_or_id: str) -> bool:
        a, _ = self.resolve(src_qname_or_id)
        b, _ = self.resolve(dst_qname_or_id)
        return bool(self.con.execute("SELECT 1 FROM edges WHERE src = ? AND dst = ? AND type = 'FLOWS'", [a, b]).fetchone())

    def is_acyclic(self) -> bool:
        """Kahn's algorithm over FLOWS edges."""
        edges = self.con.execute("SELECT src, dst FROM edges WHERE type = 'FLOWS'").fetchall()
        indeg: dict[str, int] = {}
        out: dict[str, list[str]] = {}
        for s, d in edges:
            indeg[d] = indeg.get(d, 0) + 1
            indeg.setdefault(s, 0)
            out.setdefault(s, []).append(d)
        queue = [n for n, k in indeg.items() if k == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in out.get(n, []):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        return seen == len(indeg)


__all__ = ["GraphStore", "EntityNotFound", "AmbiguousEntity", "node_id"]

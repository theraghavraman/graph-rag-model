"""Question set with ground truth computed from the graph itself.

Four groups, in rising difficulty for a retrieval system:

lookup     one fact about one entity (owner, members)
one_hop    direct neighbours in the lineage (one processing step)
multi_hop  answers that need the whole chain (source column -> ... -> report)
negative   questions whose correct answer is "nothing" - including near misses where
           the column is used downstream but never reaches a report

Each question stores the structured query that answers it (``gold_query``) and the
exact answer set (``answers``), so both the planner and the final answer can be scored.
Question wording varies and often uses display names ("the Churn Watchlist report")
the way a person would, not internal identifiers.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .model import Catalog, Node
from .query import CatalogQuery
from .store import GraphStore

GROUPS = ("lookup", "one_hop", "multi_hop", "negative")


@dataclass
class Question:
    id: str
    group: str
    question: str
    gold_query: dict
    answers: list[str]
    answer_type: str
    note: str = ""
    paths: list[list[str]] = field(default_factory=list)


def _col(node: Node, rng: random.Random) -> str:
    table = node.attrs["table"]
    return rng.choice([f"{node.qualified_name}", f"the {node.name} column in {table}", f"{table}.{node.name}", f"column {node.name} of the {table} table"])


def _report(node: Node, rng: random.Random) -> str:
    return rng.choice([f"the {node.name} report", f"the \"{node.name}\" dashboard", f"{node.name}", f"report {node.qualified_name}"])


def build_questions(catalog: Catalog, store: GraphStore, per_group: int = 30, seed: int = 11) -> list[Question]:
    rng = random.Random(seed)
    by_type = {t: catalog.of_type(t) for t in ("system", "table", "column", "pipeline", "report", "owner")}
    layer_of = {t.qualified_name: t.attrs["layer"] for t in by_type["table"]}
    raw_cols = [c for c in by_type["column"] if layer_of[c.attrs["table"]] == "raw"]
    mart_cols = [c for c in by_type["column"] if layer_of[c.attrs["table"]] == "mart"]
    derived_cols = [c for c in by_type["column"] if layer_of[c.attrs["table"]] != "raw"]
    out: list[Question] = []
    reaches_report = {c.id for c in raw_cols if store.lineage(c.qualified_name, target_type="report").hits}

    depth = {"n": 0}      # deepest result of the most recent query

    def run(**fields) -> tuple[CatalogQuery, list[str], list[list[str]]]:
        q = CatalogQuery(**fields)
        r = store.execute(q)
        depth["n"] = max((h.hops for h in r.hits), default=0)
        return q, sorted(r.names), [h.path for h in r.hits[:3]]

    def add(group: str, text: str, answer_type: str, note: str = "", want_empty: bool = False, min_answers: int = 1, max_answers: int = 25, **fields) -> bool:
        q, answers, paths = run(**fields)
        if want_empty != (not answers) or (answers and not (min_answers <= len(answers) <= max_answers)):
            return False
        if group == "multi_hop" and depth["n"] < 2:      # answerable in one step: not a multi-hop question
            return False
        # One question per (entity, query): no near-duplicates with different wording.
        if any(o.question == text or o.gold_query == q.model_dump(exclude_defaults=True) for o in out):
            return False
        n = sum(1 for o in out if o.group == group) + 1
        out.append(Question(f"{group}-{n:03d}", group, text, q.model_dump(exclude_defaults=True), answers, answer_type, note, paths))
        return True

    def fill(group: str, makers: list) -> None:
        """Give every question type an equal share of the group, then top up from whichever still has candidates."""
        made = {m: 0 for m in makers}
        share, extra = divmod(per_group, len(makers))
        for n, maker in enumerate(makers):
            quota, tries = share + (1 if n < extra else 0), 0
            while made[maker] < quota and tries < 400:
                tries += 1
                made[maker] += bool(maker())
        tries = 0
        while sum(made.values()) < per_group and tries < 2000:
            maker = makers[tries % len(makers)]
            tries += 1
            made[maker] += bool(maker())

    # ---- lookup -----------------------------------------------------------------------------
    def owner_of_table():
        t = rng.choice(by_type["table"])
        return add("lookup", rng.choice([f"Who owns the {t.qualified_name} table?", f"Who is responsible for {t.qualified_name}?", f"Which person owns table {t.qualified_name}?"]),
            "owner", kind="owner", entity=t.qualified_name)

    def owner_of_report():
        r = rng.choice(by_type["report"])
        return add("lookup", rng.choice([f"Who owns {_report(r, rng)}?", f"Who do I contact about {_report(r, rng)}?"]), "owner", kind="owner", entity=r.qualified_name)

    def owner_of_pipeline():
        p = rng.choice(by_type["pipeline"])
        return add("lookup", rng.choice([f"Who owns the pipeline {p.name}?", f"Who maintains {p.name}?"]), "owner", kind="owner", entity=p.qualified_name)

    def owner_via_column():
        c = rng.choice(by_type["column"])
        return add("lookup", f"Who owns the table that holds {_col(c, rng)}?", "owner", note="column resolves to its table's owner", kind="owner", entity=c.qualified_name)

    def tables_of_system():
        s = rng.choice(by_type["system"])
        return add("lookup", rng.choice([f"Which tables come from the {s.name} system?", f"List the raw tables delivered by {s.name}."]), "table", kind="members", entity=s.qualified_name, entity_type="system")

    def columns_of_table():
        t = rng.choice(by_type["table"])
        return add("lookup", rng.choice([f"What columns does {t.qualified_name} have?", f"List the columns of the {t.qualified_name} table."]), "column", kind="members", entity=t.qualified_name)

    fill("lookup", [owner_of_table, owner_of_report, owner_of_pipeline, owner_via_column, tables_of_system, columns_of_table])

    # ---- one hop ----------------------------------------------------------------------------
    def pipelines_reading():
        c = rng.choice(by_type["column"])
        return add("one_hop", rng.choice([f"Which pipelines read {_col(c, rng)} directly?", f"Which jobs take {_col(c, rng)} as a direct input?"]),
            "pipeline", kind="lineage", entity=c.qualified_name, target_type="pipeline", max_hops=1)

    def columns_built_from():
        c = rng.choice(by_type["column"])
        return add("one_hop", rng.choice([f"Which columns are computed directly from {_col(c, rng)}?", f"What is derived straight from {_col(c, rng)}, one step downstream?"]),
            "column", kind="lineage", entity=c.qualified_name, target_type="column", max_hops=1)

    def direct_inputs():
        c = rng.choice(derived_cols)
        return add("one_hop", rng.choice([f"Which columns is {_col(c, rng)} computed from directly?", f"What are the immediate inputs of {_col(c, rng)}?"]),
            "column", kind="lineage", entity=c.qualified_name, direction="upstream", target_type="column", max_hops=1)

    def reports_showing():
        c = rng.choice(mart_cols)
        return add("one_hop", rng.choice([f"Which reports display {_col(c, rng)} directly?", f"Which dashboards show {_col(c, rng)}?"]),
            "report", kind="lineage", entity=c.qualified_name, target_type="report", max_hops=1)

    def pipeline_inputs():
        p = rng.choice(by_type["pipeline"])
        return add("one_hop", rng.choice([f"Which tables does the pipeline {p.name} read from?", f"What are the input tables of {p.name}?"]),
            "table", kind="lineage", entity=p.qualified_name, direction="upstream", target_type="table", max_hops=1)

    fill("one_hop", [pipelines_reading, columns_built_from, direct_inputs, reports_showing, pipeline_inputs])

    # ---- multi hop --------------------------------------------------------------------------
    def reports_from_source_column():
        c = rng.choice([c for c in raw_cols if c.id in reaches_report])
        return add("multi_hop", rng.choice([f"Which reports depend on {_col(c, rng)}?", f"If {_col(c, rng)} changes, which reports are affected?", f"Which dashboards ultimately use {_col(c, rng)}?"]),
            "report", kind="lineage", entity=c.qualified_name, target_type="report")

    def systems_behind_report():
        r = rng.choice(by_type["report"])
        return add("multi_hop", rng.choice([f"Which source systems feed {_report(r, rng)}?", f"Where does the data in {_report(r, rng)} originally come from?"]),
            "system", kind="lineage", entity=r.qualified_name, direction="upstream", target_type="system")

    def tables_downstream_of_table():
        t = rng.choice([t for t in by_type["table"] if t.attrs["layer"] in ("raw", "staging")])
        return add("multi_hop", rng.choice([f"Which tables are built, directly or indirectly, from {t.qualified_name}?", f"What tables sit downstream of {t.qualified_name}?"]),
            "table", min_answers=2, kind="lineage", entity=t.qualified_name, target_type="table")

    def owners_to_notify():
        c = rng.choice([c for c in raw_cols if c.id in reaches_report])
        return add("multi_hop", rng.choice([f"Who should be told before {_col(c, rng)} is changed?", f"Which owners are affected downstream if {_col(c, rng)} is dropped?"]),
            "owner", min_answers=2, kind="lineage", entity=c.qualified_name, target_type="owner")

    def source_columns_of_mart_column():
        c = rng.choice(mart_cols)
        return add("multi_hop", rng.choice([f"Which source-system columns does {_col(c, rng)} originate from?", f"Trace {_col(c, rng)} back to its raw source columns."]),
                   "column", kind="lineage", entity=c.qualified_name, direction="upstream", target_type="column", target_layer="raw")

    def pipelines_behind_report():
        r = rng.choice(by_type["report"])
        return add("multi_hop", rng.choice([f"Which pipelines have to succeed for {_report(r, rng)} to be up to date?", f"Which jobs does {_report(r, rng)} depend on, end to end?"]),
            "pipeline", min_answers=2, max_answers=40, kind="lineage", entity=r.qualified_name, direction="upstream", target_type="pipeline")

    fill("multi_hop", [reports_from_source_column, systems_behind_report, tables_downstream_of_table, owners_to_notify, source_columns_of_mart_column, pipelines_behind_report])

    # ---- negative ---------------------------------------------------------------------------
    def dead_source_column():
        c = rng.choice([c for c in raw_cols if c.name in ("legacy_flag", "tmp_notes", "import_batch")])
        return add("negative", rng.choice([f"Which reports depend on {_col(c, rng)}?", f"Is {_col(c, rng)} used by any report?"]),
            "report", note="never loaded past the source", want_empty=True, kind="lineage", entity=c.qualified_name, target_type="report")

    def used_but_never_reported():
        c = rng.choice([c for c in raw_cols if c.id not in reaches_report])
        _, tables, _ = run(kind="lineage", entity=c.qualified_name, target_type="table")
        if len(tables) >= 2:      # a near miss: flows through the warehouse, reaches no report
            return add("negative", rng.choice([f"Which reports depend on {_col(c, rng)}?", f"Which dashboards would break if {_col(c, rng)} were removed?"]),
                "report", note=f"near miss: reaches {len(tables)} tables but no report", want_empty=True, kind="lineage", entity=c.qualified_name, target_type="report")

    def unread_model_column():
        c = rng.choice(derived_cols)
        return add("negative", rng.choice([f"Does anything downstream read {_col(c, rng)}?", f"Which pipelines or reports use {_col(c, rng)}?"]),
            "any", note="published but never read", want_empty=True, kind="lineage", entity=c.qualified_name)

    def no_direct_report():
        c = rng.choice([c for c in by_type["column"] if layer_of[c.attrs["table"]] == "staging"])
        return add("negative", f"Which reports display {_col(c, rng)} directly, without any further processing?",
            "report", note="staging columns are never shown directly", want_empty=True, kind="lineage", entity=c.qualified_name, target_type="report", max_hops=1)

    fill("negative", [dead_source_column, used_but_never_reported, unread_model_column, no_direct_report])
    return out


def write_jsonl(questions: list[Question], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]

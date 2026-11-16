"""A second question set with different wording, for testing how well planning generalises.

The original set (questions.py) uses a few fixed phrasings and writes entities almost
exactly as they are named. Real questions do neither. This module rewrites questions
from their structured query using looser language ("blast radius", "who needs a
heads-up") and looser entity mentions ("total amount in billing orders", lower-cased
report titles, an occasional typo).

It is a held-out set: the rule-based planner and the entity linker were written against
the original wording only and must not be tuned on this one. It is still templated, and
written by the same author, so it is a harder test, not an independent one.
"""

from __future__ import annotations

import random

from .model import Catalog
from .questions import Question, build_questions
from .store import GraphStore

# (kind, direction, target_type, direct-only?, target_layer, entity type) -> phrasings; {e} is the entity mention
WORDINGS: dict[tuple, list[str]] = {
    ("owner", None, None, False, None, "table"): ["Whose table is {e}?", "Point me to the person in charge of {e}.", "I need the owner for {e}."],
    ("owner", None, None, False, None, "report"): ["Who looks after {e}?", "Who's on the hook for {e}?"],
    ("owner", None, None, False, None, "pipeline"): ["Who looks after {e}?", "Who do I ping when {e} fails?"],
    ("owner", None, None, False, None, "column"): ["Who would I ask about {e}?", "Who is accountable for {e}?"],
    ("members", None, None, False, None, "system"): ["What do we ingest from {e}?", "Show me every table {e} sends us."],
    ("members", None, None, False, None, "table"): ["What fields are in {e}?", "Give me the schema of {e}."],
    ("lineage", "downstream", "pipeline", True, None, "column"): ["What jobs consume {e} first-hand?", "Name the pipelines that pick up {e} as an immediate input."],
    ("lineage", "downstream", "column", True, None, "column"): ["What gets calculated right from {e}?", "Next step after {e}: which fields?"],
    ("lineage", "upstream", "column", True, None, "column"): ["What goes straight into {e}?", "{e} is built from which fields, just one step back?"],
    ("lineage", "downstream", "report", True, None, "column"): ["On which dashboards does {e} appear as-is?", "Where is {e} surfaced to users with no extra transformation?"],
    ("lineage", "upstream", "table", True, None, "pipeline"): ["What does {e} pull from?", "Source tables for {e}?"],
    ("lineage", "downstream", "report", False, None, "column"): ["What breaks in reporting if we drop {e}?", "I'm about to rename {e}. Any dashboards I should worry about?", "Downstream dashboards for {e}?"],
    ("lineage", "upstream", "system", False, None, "report"): ["What upstream systems is {e} sourced from?", "Trace {e} to the operational systems behind it."],
    ("lineage", "downstream", "table", False, None, "table"): ["Everything that's derived from {e}, at any depth - which tables?", "Blast radius of {e} in terms of tables?"],
    ("lineage", "downstream", "owner", False, None, "column"): ["Whose stuff is impacted if {e} goes away?", "Who needs a heads-up about changes to {e}?"],
    ("lineage", "upstream", "column", False, "raw", "column"): ["What raw fields ultimately feed {e}?", "Where in the source systems does {e} start?"],
    ("lineage", "upstream", "pipeline", False, None, "report"): ["What has to run before {e} is fresh?", "Full list of jobs upstream of {e}?"],
    ("lineage", "downstream", None, False, None, "column"): ["Is {e} consumed anywhere?", "Anything using {e}?"],
}


def _typo(text: str, rng: random.Random) -> str:
    """Drop one letter from the last name part, as a hurried typist would."""
    head, _, tail = text.rpartition(".")
    if len(tail) < 6:
        return text
    i = rng.randrange(1, len(tail) - 1)
    return (head + "." if head else "") + tail[:i] + tail[i + 1:]


def _mention(node_type: str, qname: str, name: str, rng: random.Random) -> str:
    if node_type == "column":
        table, col = qname.rsplit(".", 1)
        schema, tname = table.split(".", 1)
        return rng.choice([
            f"{col.replace('_', ' ')} in {schema} {tname.replace('_', ' ')}",
            f"{table}'s {col} field",
            f"the {col} field on {table}",
            f"{schema} {tname} {col}",
            _typo(qname, rng),
        ])
    if node_type == "report":
        return rng.choice([f"the {name.lower()} dashboard", f"our {name} board", f"{name.lower()}"])
    if node_type == "table":
        schema, tname = qname.split(".", 1)
        return rng.choice([f"the {schema} {tname} table", f"{qname}", f"table {_typo(qname, rng)}"])
    if node_type == "pipeline":
        return rng.choice([f"the {name} job", f"job {name}"])
    if node_type == "system":
        return rng.choice([f"the {name} source system", f"the {name.upper()} system"])
    return name


def build_paraphrased(catalog: Catalog, store: GraphStore, per_group: int = 15, seed: int = 23) -> list[Question]:
    """Different entities (another seed) and different wording from the main set."""
    rng = random.Random(seed)
    by_id = {n.id: n for n in catalog.nodes}
    out: list[Question] = []
    for q in build_questions(catalog, store, per_group=per_group, seed=seed):
        g = q.gold_query
        node_id, node_type = store.resolve(g["entity"], g.get("entity_type"))
        key = (g["kind"], g.get("direction", "downstream") if g["kind"] == "lineage" else None, g.get("target_type"),
               g.get("max_hops") == 1, g.get("target_layer"), node_type)
        wording = rng.choice(WORDINGS[key])
        node = by_id[node_id]
        text = wording.format(e=_mention(node_type, node.qualified_name, node.name, rng))
        out.append(Question("p-" + q.id, q.group, text[0].upper() + text[1:], g, q.answers, q.answer_type, q.note, q.paths))
    return out

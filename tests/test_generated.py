"""Properties of the synthetic catalog and its question set."""

import json

from catalog_graph import CatalogQuery, GraphStore
from catalog_graph.documents import build_documents
from catalog_graph.generator import GeneratorConfig, generate
from catalog_graph.questions import GROUPS, build_questions


def test_generation_is_deterministic():
    a, b = generate(GeneratorConfig(seed=3)), generate(GeneratorConfig(seed=3))
    assert [n.id for n in a.nodes] == [n.id for n in b.nodes] and a.edges == b.edges
    assert [n.id for n in a.nodes] != [n.id for n in generate(GeneratorConfig(seed=4)).nodes]


def test_catalog_shape(generated):
    catalog, store = generated
    counts = catalog.counts()
    assert counts["system"] == 8 and counts["table"] == 135 and counts["report"] == 40 and counts["owner"] == 12
    assert counts["column"] > 1000
    assert counts["pipeline"] == counts["table"] - 40          # one pipeline per derived table
    ids = [n.id for n in catalog.nodes]
    assert len(ids) == len(set(ids))
    assert store.is_acyclic()
    assert store.counts() == {k: v for k, v in counts.items() if v}


def test_layers_only_feed_forward(generated):
    catalog, _ = generated
    by_id = {n.id: n for n in catalog.nodes}
    layer = {t.qualified_name: t.attrs["layer"] for t in catalog.of_type("table")}
    order = {"raw": 0, "staging": 1, "core": 2, "mart": 3}
    for e in catalog.edges:
        if e.type == "FLOWS" and e.via:
            src, dst = by_id[e.src], by_id[e.dst]
            assert order[layer[src.attrs["table"]]] < order[layer[dst.attrs["table"]]]
            assert by_id[e.via].attrs["target"] == dst.attrs["table"]      # the pipeline on the edge is the one that builds the target


def test_everything_has_exactly_one_owner(generated):
    catalog, _ = generated
    owned = [e.dst for e in catalog.edges if e.type == "OWNS"]
    assert len(owned) == len(set(owned))
    assert set(owned) == {n.id for n in catalog.nodes if n.type in ("table", "pipeline", "report")}


def test_dead_end_columns_reach_nothing(generated):
    catalog, store = generated
    dead = [n for n in catalog.of_type("column") if n.name in ("legacy_flag", "tmp_notes", "import_batch", "row_hash")]
    assert len(dead) > 80
    assert all(store.lineage(n.qualified_name).empty for n in dead)


def test_question_set_is_balanced_and_reproducible(generated, question_set):
    catalog, store = generated
    questions = question_set
    assert [sum(1 for q in questions if q.group == g) for g in GROUPS] == [30, 30, 30, 30]
    assert len({q.id for q in questions}) == 120 and len({q.question for q in questions}) == 120
    assert [q.question for q in questions] == [q.question for q in build_questions(catalog, store, per_group=30)]
    multi = [q for q in questions if q.group == "multi_hop"]
    assert {q.answer_type for q in multi} == {"report", "system", "table", "owner", "column", "pipeline"}
    assert sum(1 for q in multi if q.answer_type == "report") >= 5


def test_gold_queries_reproduce_the_stored_answers(generated, question_set):
    catalog, store = generated
    for q in question_set:
        result = store.execute(CatalogQuery(**q.gold_query))
        assert sorted(result.names) == q.answers, q.id
        assert (q.group == "negative") == (not q.answers), q.id


def test_multi_hop_questions_really_need_several_steps(generated, question_set):
    catalog, store = generated
    for q in question_set:
        if q.group == "multi_hop":
            hits = store.execute(CatalogQuery(**q.gold_query)).hits
            assert max(h.hops for h in hits) >= 2, q.id
        if q.group == "one_hop":
            assert q.gold_query["max_hops"] == 1


def test_near_miss_negatives_exist(generated, question_set):
    catalog, store = generated
    near = [q for q in question_set if q.note.startswith("near miss")]
    assert len(near) >= 5
    for q in near:      # used downstream, yet no report: the hard kind of "no"
        assert not store.lineage(q.gold_query["entity"], target_type="table").empty


def test_baseline_documents_contain_every_lineage_fact(generated):
    catalog, _ = generated
    docs = build_documents(catalog)
    assert len(docs) == sum(catalog.counts()[t] for t in ("system", "table", "pipeline", "report"))
    text = {d["id"]: d["text"] for d in docs}
    by_id = {n.id: n for n in catalog.nodes}
    for e in catalog.edges:
        if e.type == "FLOWS":
            holder = e.via or e.dst      # pipeline document for a mapping, report document for a displayed column
            assert by_id[e.src].qualified_name in text[holder]
            if e.via:
                assert by_id[e.dst].qualified_name in text[holder]
    json.dumps(docs)


def test_persisted_database_round_trip(tmp_path, generated):
    catalog, memory = generated
    path = str(tmp_path / "catalog.duckdb")
    with GraphStore(path) as store:
        store.load(catalog)
    with GraphStore(path, read_only=True) as store:
        q = "billing.orders.total_amount" if any(n.qualified_name == "billing.orders.total_amount" for n in catalog.nodes) else catalog.of_type("column")[0].qualified_name
        assert store.lineage(q, target_type="report").names == memory.lineage(q, target_type="report").names

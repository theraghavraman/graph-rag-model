"""Traversal answers on a graph small enough to verify by hand."""

import pytest
from pydantic import ValidationError

from catalog_graph import CatalogQuery, GraphStore
from catalog_graph.store import AmbiguousEntity, EntityNotFound


def names(result):
    return sorted(result.names)


def test_reports_downstream_of_a_source_column(tiny):
    assert names(tiny.lineage("crm.customers.email", target_type="report")) == ["rpt_churn"]
    assert names(tiny.lineage("crm.customers.city", target_type="report")) == ["rpt_exec"]


def test_lineage_is_exact_per_column_not_per_pipeline(tiny):
    # email and name pass through the same pipelines, but only name reaches the mart.
    assert names(tiny.lineage("crm.customers.email", target_type="column")) == ["core.dim_customer.email", "stg.customers.email", "stg.customers.row_hash"]
    assert "mart.kpi.customers" not in tiny.lineage("crm.customers.email").names
    assert "mart.kpi.customers" in tiny.lineage("crm.customers.name").names


def test_hops_count_processing_steps(tiny):
    hops = {h.qualified_name: h.hops for h in tiny.lineage("crm.customers.name").hits}
    assert hops["stg.customers.name"] == 1 and hops["pl_stg"] == 1
    assert hops["core.dim_customer.label"] == 2 and hops["pl_core"] == 2
    assert hops["mart.kpi.customers"] == 3 and hops["pl_mart"] == 3
    assert hops["rpt_exec"] == 4


def test_max_hops_limits_the_traversal(tiny):
    assert names(tiny.lineage("crm.customers.name", target_type="column", max_hops=1)) == ["stg.customers.name", "stg.customers.row_hash"]
    assert names(tiny.lineage("crm.customers.name", target_type="pipeline", max_hops=1)) == ["pl_stg"]
    assert names(tiny.lineage("crm.customers.name", target_type="pipeline", max_hops=2)) == ["pl_core", "pl_stg"]
    assert tiny.lineage("crm.customers.name", target_type="report", max_hops=3).empty


def test_upstream(tiny):
    assert names(tiny.lineage("rpt_exec", direction="upstream", target_type="column", target_layer="raw")) == ["crm.customers.city", "crm.customers.name"]
    assert names(tiny.lineage("rpt_exec", direction="upstream", target_type="pipeline")) == ["pl_core", "pl_mart", "pl_stg"]
    assert names(tiny.lineage("rpt_churn", direction="upstream", target_type="system")) == ["crm"]
    assert names(tiny.lineage("core.dim_customer.label", direction="upstream", target_type="column", max_hops=1)) == ["stg.customers.city", "stg.customers.name"]


def test_table_as_start_and_as_target(tiny):
    assert names(tiny.lineage("crm.customers", target_type="table")) == ["core.dim_customer", "mart.kpi", "stg.customers"]
    assert names(tiny.lineage("crm.customers", target_type="report")) == ["rpt_churn", "rpt_exec"]
    assert names(tiny.lineage("mart.kpi", direction="upstream", target_type="table")) == ["core.dim_customer", "crm.customers", "stg.customers"]
    assert names(tiny.lineage("mart.kpi", direction="upstream", target_type="table", target_layer="raw")) == ["crm.customers"]


def test_pipeline_as_start(tiny):
    assert names(tiny.lineage("pl_core", target_type="table")) == ["core.dim_customer", "mart.kpi"]
    assert names(tiny.lineage("pl_core", target_type="table", max_hops=1)) == ["core.dim_customer"]
    assert names(tiny.lineage("pl_core", direction="upstream", target_type="table", max_hops=1)) == ["stg.customers"]
    assert names(tiny.lineage("pl_core", target_type="report")) == ["rpt_churn", "rpt_exec"]
    assert "pl_core" not in tiny.lineage("pl_core").names


def test_owners_affected_downstream(tiny):
    assert names(tiny.lineage("crm.customers.city", target_type="owner")) == ["Asha Verma", "Omar Haddad"]
    assert names(tiny.lineage("core.dim_customer.label", target_type="owner")) == ["Omar Haddad"]


def test_negatives_return_nothing(tiny):
    assert tiny.lineage("crm.customers.legacy_flag").empty
    assert tiny.lineage("stg.customers.row_hash").empty
    assert tiny.lineage("rpt_exec").empty                                  # nothing is downstream of a report
    assert tiny.lineage("crm.customers.email", direction="upstream").empty  # nothing is upstream of a source column


def test_every_path_is_made_of_real_edges(tiny):
    for entity, direction in (("crm.customers.name", "downstream"), ("rpt_exec", "upstream")):
        for hit in tiny.lineage(entity, direction=direction, target_type="column").hits + tiny.lineage(entity, direction=direction, target_type="report").hits:
            assert hit.path[-1] == hit.qualified_name and len(hit.path) == hit.hops + 1
            for a, b in zip(hit.path, hit.path[1:]):
                assert tiny.has_edge(a, b) if direction == "downstream" else tiny.has_edge(b, a)


def test_evidence_path_reads_start_to_result(tiny):
    hit = tiny.lineage("crm.customers.name", target_type="report").hits[0]
    assert hit.path == ["crm.customers.name", "stg.customers.name", "core.dim_customer.label", "mart.kpi.customers", "rpt_exec"]


def test_owner_and_members(tiny):
    assert tiny.execute(CatalogQuery(kind="owner", entity="core.dim_customer")).names == ["Omar Haddad"]
    assert tiny.execute(CatalogQuery(kind="owner", entity="crm.customers.email")).names == ["Asha Verma"]      # through its table
    assert tiny.execute(CatalogQuery(kind="owner", entity="Churn Watchlist")).names == ["Asha Verma"]          # display name
    assert tiny.execute(CatalogQuery(kind="members", entity="crm", entity_type="system")).names == ["crm.customers"]
    assert len(tiny.execute(CatalogQuery(kind="members", entity="stg.customers")).hits) == 4


def test_entity_resolution(tiny):
    assert tiny.resolve("CRM.Customers.EMAIL") == ("column:crm.customers.email", "column")
    assert tiny.resolve("column:crm.customers.email")[1] == "column"
    with pytest.raises(EntityNotFound):
        tiny.resolve("crm.customers.nope")
    with pytest.raises(AmbiguousEntity) as err:      # "email" is the display name of three columns
        tiny.resolve("email")
    assert len(err.value.candidates) == 3
    with pytest.raises(AmbiguousEntity):             # the table and several columns are all called "customers"
        tiny.resolve("customers")
    assert tiny.resolve("customers", "column") == ("column:mart.kpi.customers", "column")


def test_only_validated_query_shapes_are_accepted():
    with pytest.raises(ValidationError):
        CatalogQuery(kind="drop_tables", entity="x")
    with pytest.raises(ValidationError):
        CatalogQuery(kind="lineage", entity="x", max_hops=0)
    with pytest.raises(ValidationError):
        CatalogQuery(kind="owner", entity="x", target_type="report")
    with pytest.raises(ValidationError):
        CatalogQuery(kind="lineage", entity="x", target_type="report", target_layer="raw")
    with pytest.raises(ValidationError):
        CatalogQuery(kind="lineage", entity="x", sql="SELECT 1")


def test_names_are_data_not_sql(tiny):
    with pytest.raises(EntityNotFound):
        tiny.lineage("x'; DROP TABLE nodes; --")
    assert tiny.counts()["column"] == 11


def test_cycles_terminate_and_are_detected():
    from catalog_graph import Catalog, Node
    c = Catalog()
    a, b = (c.add(Node("column", f"t.{n}", n, attrs={"data_type": "INT", "table": "t"})) for n in "ab")
    p = c.add(Node("pipeline", "p", "p"))
    c.link(a, b, "FLOWS", via=p)
    c.link(b, a, "FLOWS", via=p)
    with GraphStore() as store:
        store.load(c)
        assert not store.is_acyclic()
        assert store.lineage("t.a", target_type="column").names == ["t.b"]      # terminates; the start is never its own result


def test_load_rejects_edges_to_unknown_nodes():
    from catalog_graph import Catalog, Node
    c = Catalog()
    a = c.add(Node("column", "t.a", "a"))
    c.link(a, "column:t.missing", "FLOWS")
    with GraphStore() as store, pytest.raises(ValueError):
        store.load(c)

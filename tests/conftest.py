import pytest

from catalog_graph import Catalog, GraphStore, Node
from catalog_graph.generator import generate


def tiny_catalog() -> Catalog:
    """A hand-built warehouse small enough to check every answer by eye.

        crm.customers.email ──pl_stg──> stg.customers.email ──pl_core──> core.dim_customer.email ──> rpt_churn
        crm.customers.name  ──pl_stg──> stg.customers.name  ──pl_core──> core.dim_customer.label ──pl_mart──> mart.kpi.customers ──> rpt_exec
        crm.customers.city  ──pl_stg──> stg.customers.city  ──pl_core──┘ (label = name + city)
        crm.customers.legacy_flag        (never loaded)
        stg.customers.row_hash           (built from email + name, never read)
    """
    c = Catalog()

    def table(schema, name, layer):
        return c.add(Node("table", f"{schema}.{name}", name, attrs={"schema": schema, "layer": layer}))

    def col(t, name):
        n = c.add(Node("column", f"{t.qualified_name}.{name}", name, attrs={"data_type": "VARCHAR", "table": t.qualified_name}))
        c.link(t, n, "CONTAINS")
        return n

    crm = c.add(Node("system", "crm", "crm"))
    raw, stg, core, mart = table("crm", "customers", "raw"), table("stg", "customers", "staging"), table("core", "dim_customer", "core"), table("mart", "kpi", "mart")
    c.link(crm, raw, "CONTAINS")
    r_email, r_name, r_city, _legacy = (col(raw, n) for n in ("email", "name", "city", "legacy_flag"))
    s_email, s_name, s_city, s_hash = (col(stg, n) for n in ("email", "name", "city", "row_hash"))
    d_email, d_label = col(core, "email"), col(core, "label")
    m_customers = col(mart, "customers")
    pl_stg, pl_core, pl_mart = (c.add(Node("pipeline", n, n)) for n in ("pl_stg", "pl_core", "pl_mart"))
    churn = c.add(Node("report", "rpt_churn", "Churn Watchlist"))
    execu = c.add(Node("report", "rpt_exec", "Executive Summary"))
    asha, omar = c.add(Node("owner", "Asha Verma", "Asha Verma")), c.add(Node("owner", "Omar Haddad", "Omar Haddad"))

    for a, b in ((r_email, s_email), (r_name, s_name), (r_city, s_city), (r_email, s_hash), (r_name, s_hash)):
        c.link(a, b, "FLOWS", via=pl_stg)
    for a, b in ((s_email, d_email), (s_name, d_label), (s_city, d_label)):
        c.link(a, b, "FLOWS", via=pl_core)
    c.link(d_label, m_customers, "FLOWS", via=pl_mart)
    c.link(d_email, churn, "FLOWS")
    c.link(m_customers, execu, "FLOWS")
    for owner, things in ((asha, (raw, stg, pl_stg, churn)), (omar, (core, mart, pl_core, pl_mart, execu))):
        for thing in things:
            c.link(owner, thing, "OWNS")
    return c


@pytest.fixture()
def tiny() -> GraphStore:
    store = GraphStore()
    store.load(tiny_catalog())
    yield store
    store.close()


@pytest.fixture(scope="session")
def generated():
    catalog = generate()
    store = GraphStore()
    store.load(catalog)
    yield catalog, store
    store.close()


@pytest.fixture(scope="session")
def question_set(generated):
    from catalog_graph.questions import build_questions
    catalog, store = generated
    return build_questions(catalog, store, per_group=30)

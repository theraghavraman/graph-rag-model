"""Synthetic data catalog with column-level lineage and known ground truth.

The catalog is shaped like a small warehouse:

    source systems -> raw tables -> staging -> core (dims and facts) -> marts -> reports

Every derived table is written by exactly one pipeline, and every derived column
records which input columns it was computed from. Some columns are deliberately
dead ends (never read by anything) so that "does anything depend on X?" has true
negative answers. Generation is deterministic for a given seed.

All names, people and systems are invented.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .model import Catalog, Node

# (column name, type) per business entity. The first column is the key.
ENTITIES: dict[str, list[tuple[str, str]]] = {
    "customers": [("customer_id", "BIGINT"), ("full_name", "VARCHAR"), ("email", "VARCHAR"), ("phone", "VARCHAR"), ("city", "VARCHAR"), ("country", "VARCHAR"), ("segment", "VARCHAR"), ("signup_date", "DATE"), ("is_active", "BOOLEAN")],
    "orders": [("order_id", "BIGINT"), ("customer_id", "BIGINT"), ("order_date", "DATE"), ("status", "VARCHAR"), ("channel", "VARCHAR"), ("currency", "VARCHAR"), ("subtotal", "DECIMAL"), ("discount", "DECIMAL"), ("tax", "DECIMAL"), ("total_amount", "DECIMAL")],
    "order_items": [("order_item_id", "BIGINT"), ("order_id", "BIGINT"), ("product_id", "BIGINT"), ("quantity", "INTEGER"), ("unit_price", "DECIMAL"), ("line_discount", "DECIMAL"), ("line_total", "DECIMAL")],
    "payments": [("payment_id", "BIGINT"), ("order_id", "BIGINT"), ("paid_at", "TIMESTAMP"), ("method", "VARCHAR"), ("amount", "DECIMAL"), ("currency", "VARCHAR"), ("gateway_ref", "VARCHAR"), ("is_refunded", "BOOLEAN")],
    "invoices": [("invoice_id", "BIGINT"), ("customer_id", "BIGINT"), ("issued_on", "DATE"), ("due_on", "DATE"), ("net_amount", "DECIMAL"), ("vat_amount", "DECIMAL"), ("gross_amount", "DECIMAL"), ("state", "VARCHAR")],
    "products": [("product_id", "BIGINT"), ("sku", "VARCHAR"), ("title", "VARCHAR"), ("category", "VARCHAR"), ("brand", "VARCHAR"), ("list_price", "DECIMAL"), ("unit_cost", "DECIMAL"), ("launched_on", "DATE")],
    "shipments": [("shipment_id", "BIGINT"), ("order_id", "BIGINT"), ("carrier", "VARCHAR"), ("shipped_at", "TIMESTAMP"), ("delivered_at", "TIMESTAMP"), ("destination_city", "VARCHAR"), ("freight_cost", "DECIMAL"), ("weight_kg", "DECIMAL")],
    "tickets": [("ticket_id", "BIGINT"), ("customer_id", "BIGINT"), ("opened_at", "TIMESTAMP"), ("closed_at", "TIMESTAMP"), ("priority", "VARCHAR"), ("topic", "VARCHAR"), ("assignee", "VARCHAR"), ("satisfaction", "INTEGER")],
    "sessions": [("session_id", "VARCHAR"), ("visitor_id", "VARCHAR"), ("started_at", "TIMESTAMP"), ("device", "VARCHAR"), ("landing_page", "VARCHAR"), ("referrer", "VARCHAR"), ("page_views", "INTEGER"), ("duration_seconds", "INTEGER")],
    "employees": [("employee_id", "BIGINT"), ("display_name", "VARCHAR"), ("department", "VARCHAR"), ("job_title", "VARCHAR"), ("manager_id", "BIGINT"), ("hired_on", "DATE"), ("office", "VARCHAR")],
    "accounts": [("account_id", "BIGINT"), ("account_name", "VARCHAR"), ("industry", "VARCHAR"), ("tier", "VARCHAR"), ("region", "VARCHAR"), ("owner_rep", "VARCHAR"), ("opened_on", "DATE"), ("credit_limit", "DECIMAL")],
    "subscriptions": [("subscription_id", "BIGINT"), ("account_id", "BIGINT"), ("plan", "VARCHAR"), ("started_on", "DATE"), ("renews_on", "DATE"), ("monthly_fee", "DECIMAL"), ("seats", "INTEGER"), ("is_trial", "BOOLEAN")],
    "refunds": [("refund_id", "BIGINT"), ("payment_id", "BIGINT"), ("requested_at", "TIMESTAMP"), ("reason", "VARCHAR"), ("refund_amount", "DECIMAL"), ("approved_by", "VARCHAR")],
    "campaigns": [("campaign_id", "BIGINT"), ("campaign_name", "VARCHAR"), ("medium", "VARCHAR"), ("start_date", "DATE"), ("end_date", "DATE"), ("budget", "DECIMAL"), ("spend", "DECIMAL"), ("target_segment", "VARCHAR")],
    "leads": [("lead_id", "BIGINT"), ("campaign_id", "BIGINT"), ("captured_at", "TIMESTAMP"), ("company", "VARCHAR"), ("contact_email", "VARCHAR"), ("score", "INTEGER"), ("stage", "VARCHAR")],
    "suppliers": [("supplier_id", "BIGINT"), ("supplier_name", "VARCHAR"), ("supplier_country", "VARCHAR"), ("payment_terms", "VARCHAR"), ("rating", "INTEGER"), ("onboarded_on", "DATE")],
    "inventory": [("inventory_id", "BIGINT"), ("product_id", "BIGINT"), ("warehouse", "VARCHAR"), ("on_hand", "INTEGER"), ("reserved", "INTEGER"), ("reorder_point", "INTEGER"), ("counted_on", "DATE")],
    "returns": [("return_id", "BIGINT"), ("order_item_id", "BIGINT"), ("returned_on", "DATE"), ("condition", "VARCHAR"), ("restocked", "BOOLEAN"), ("return_reason", "VARCHAR")],
    "reviews": [("review_id", "BIGINT"), ("product_id", "BIGINT"), ("customer_id", "BIGINT"), ("stars", "INTEGER"), ("review_title", "VARCHAR"), ("posted_at", "TIMESTAMP")],
    "addresses": [("address_id", "BIGINT"), ("customer_id", "BIGINT"), ("street", "VARCHAR"), ("postal_code", "VARCHAR"), ("town", "VARCHAR"), ("state_code", "VARCHAR"), ("is_default", "BOOLEAN")],
}

SYSTEMS: list[tuple[str, str]] = [
    ("crm", "Customer relationship management"), ("erp", "Finance and resource planning"),
    ("shop", "Online storefront"), ("billing", "Billing and invoicing"),
    ("support", "Customer support desk"), ("web", "Web analytics"),
    ("wms", "Warehouse management"), ("mkt", "Marketing automation"),
]

# Source columns that exist in raw tables but are never loaded downstream.
DEAD_END_COLUMNS: list[tuple[str, str]] = [("legacy_flag", "VARCHAR"), ("tmp_notes", "VARCHAR"), ("import_batch", "VARCHAR")]

OWNERS: list[tuple[str, str]] = [
    ("Asha Verma", "Data Platform"), ("Daniel Okafor", "Data Platform"), ("Mei Tanaka", "Finance Analytics"),
    ("Lucas Ferreira", "Finance Analytics"), ("Priya Nair", "Sales Operations"), ("Tomasz Kowalski", "Sales Operations"),
    ("Hannah Schmidt", "Marketing Analytics"), ("Omar Haddad", "Marketing Analytics"), ("Ingrid Larsen", "Supply Chain"),
    ("Kwame Mensah", "Supply Chain"), ("Sofia Romano", "Customer Experience"), ("Arjun Mehta", "Customer Experience"),
]

CORE_NAMES = ["dim_customer", "dim_product", "dim_account", "dim_employee", "dim_supplier", "dim_campaign", "dim_geography", "dim_channel",
              "fct_orders", "fct_order_lines", "fct_payments", "fct_invoices", "fct_shipments", "fct_tickets", "fct_sessions", "fct_refunds",
              "fct_returns", "fct_inventory", "fct_subscriptions", "fct_leads", "fct_reviews", "fct_spend", "bridge_customer_account",
              "bridge_order_shipment", "fct_collections", "fct_fulfilment", "fct_pipeline", "fct_stock_moves", "fct_renewals", "fct_contacts"]
MART_NAMES = ["sales_daily", "sales_by_region", "revenue_monthly", "customer_ltv", "customer_churn", "product_margin", "product_ranking",
              "payment_health", "invoice_ageing", "cash_collected", "shipping_sla", "carrier_scorecard", "support_backlog", "support_csat",
              "web_funnel", "traffic_sources", "campaign_roi", "lead_conversion", "stock_cover", "returns_rate", "supplier_scorecard",
              "subscription_mrr", "renewal_forecast", "headcount", "account_health"]
REPORT_TITLES = ["Daily Sales Flash", "Regional Performance", "Revenue Overview", "Customer Lifetime Value", "Churn Watchlist", "Product Margin Review",
                 "Top Products", "Payments Health", "Invoice Ageing", "Cash Collection", "Shipping SLA", "Carrier Scorecard", "Support Backlog",
                 "Customer Satisfaction", "Web Funnel", "Traffic Sources", "Campaign ROI", "Lead Conversion", "Stock Cover", "Returns Rate",
                 "Supplier Scorecard", "Monthly Recurring Revenue", "Renewal Forecast", "Headcount Summary", "Account Health", "Executive Summary",
                 "Finance Close Pack", "Sales Pipeline", "Marketing Weekly", "Operations Weekly", "Fulfilment Exceptions", "Refund Monitor",
                 "Discount Leakage", "New Customer Cohorts", "Segment Profitability", "Warehouse Utilisation", "Subscription Seats", "Trial Conversion",
                 "Credit Exposure", "Review Sentiment"]
MEASURE_PREFIXES = ["total", "avg", "max", "min", "sum", "count", "share", "net"]

# Business words that point at an entity, so models are wired to plausible inputs
# (customer_churn reads customer and subscription data) and not to arbitrary ones.
RELATED: dict[str, list[str]] = {
    "sales": ["order"], "revenue": ["order", "invoice", "payment"], "ltv": ["customer", "order"], "churn": ["customer", "subscription"],
    "margin": ["product", "order"], "ranking": ["product", "review"], "health": ["payment", "account"], "ageing": ["invoice"],
    "cash": ["payment", "collection"], "collected": ["payment"], "sla": ["shipment"], "shipping": ["shipment"], "carrier": ["shipment"],
    "support": ["ticket"], "backlog": ["ticket"], "csat": ["ticket", "review"], "web": ["session"], "funnel": ["session", "lead"],
    "traffic": ["session"], "sources": ["session", "campaign"], "roi": ["campaign", "spend"], "conversion": ["lead"], "stock": ["inventory"],
    "cover": ["inventory"], "rate": ["return"], "scorecard": ["supplier", "shipment"], "mrr": ["subscription"], "renewal": ["subscription"],
    "forecast": ["subscription"], "headcount": ["employee"], "geography": ["address", "customer"], "channel": ["order", "session"],
    "lines": ["order", "item"], "collections": ["payment", "invoice"], "fulfilment": ["shipment", "order"], "pipeline": ["lead", "account"],
    "moves": ["inventory"], "renewals": ["subscription"], "contacts": ["customer", "lead"], "spend": ["campaign"], "region": ["address", "account"],
}


def _tokens(name: str) -> set[str]:
    """Singular business words in a model or table name."""
    out: set[str] = set()
    for word in name.lower().split("_"):
        if word in ("dim", "fct", "bridge", "by", "daily", "monthly"):
            continue
        out.add(word)
        out.update(RELATED.get(word, []))
        if word.endswith("ies"):
            out.add(word[:-3] + "y")
        elif word.endswith("s"):
            out.add(word[:-1])
    return out


@dataclass
class GeneratorConfig:
    seed: int = 7
    tables_per_system: int = 5
    core_tables: int = 30
    mart_tables: int = 25
    reports: int = 40
    unused_fraction: float = 0.12  # share of core/mart columns that nothing reads


def _snake(title: str) -> str:
    return title.lower().replace(" ", "_")


def generate(config: GeneratorConfig | None = None) -> Catalog:
    cfg = config or GeneratorConfig()
    rng = random.Random(cfg.seed)
    cat = Catalog()

    owners = [cat.add(Node("owner", name, name, f"{name}, {team} team.", {"team": team})) for name, team in OWNERS]

    def own(node: Node) -> None:
        cat.link(rng.choice(owners), node, "OWNS")

    def add_table(schema: str, name: str, layer: str, description: str) -> Node:
        return cat.add(Node("table", f"{schema}.{name}", name, description, {"schema": schema, "layer": layer}))

    def add_column(table: Node, name: str, dtype: str, description: str) -> Node:
        col = cat.add(Node("column", f"{table.qualified_name}.{name}", name, description, {"data_type": dtype, "table": table.qualified_name}))
        cat.link(table, col, "CONTAINS")
        return col

    def add_pipeline(target: Node, sources: list[Node], verb: str) -> Node:
        name = "pl_" + target.qualified_name.replace(".", "_")
        src_names = ", ".join(s.qualified_name for s in sources)
        pipe = cat.add(Node("pipeline", name, name, f"{verb} {target.qualified_name} from {src_names}.",
                            {"target": target.qualified_name, "sources": [s.qualified_name for s in sources],
                             "schedule": rng.choice(["hourly", "daily 02:00", "daily 05:30", "weekly Monday"])}))
        own(pipe)
        return pipe

    def derive(pipe: Node, out_col: Node, inputs: list[Node]) -> None:
        for c in inputs:
            cat.link(c, out_col, "FLOWS", via=pipe)

    columns_of: dict[str, list[Node]] = {}

    # ---- source systems and raw tables ---------------------------------------------------------
    entity_names = list(ENTITIES)
    rng.shuffle(entity_names)
    pool = entity_names * 2          # each entity appears in at most two systems
    raw_tables: list[Node] = []
    for i, (sys_name, sys_desc) in enumerate(SYSTEMS):
        system = cat.add(Node("system", sys_name, sys_name, f"{sys_desc} source system.", {}))
        picked: list[str] = []
        for entity in pool[i * cfg.tables_per_system:(i + 1) * cfg.tables_per_system]:
            if entity not in picked:
                picked.append(entity)
        for entity in picked:
            table = add_table(sys_name, entity, "raw", f"Raw {entity.replace('_', ' ')} as delivered by the {sys_name} system.")
            cat.link(system, table, "CONTAINS")
            own(table)
            cols = [add_column(table, n, t, f"{n.replace('_', ' ').capitalize()} of the {entity.replace('_', ' ')} record.") for n, t in ENTITIES[entity]]
            cols.append(add_column(table, "updated_at", "TIMESTAMP", "Last modification time in the source system."))
            for n, t in rng.sample(DEAD_END_COLUMNS, k=rng.choice([1, 1, 2])):
                add_column(table, n, t, "Legacy source field. Not loaded into the warehouse.")   # dead end by construction
            columns_of[table.id] = cols
            raw_tables.append(table)

    # ---- staging: one table per raw table ------------------------------------------------------
    staging: list[Node] = []
    for raw in raw_tables:
        sys_name = raw.attrs["schema"]
        table = add_table("stg", f"{sys_name}_{raw.name}", "staging", f"Cleaned and typed copy of {raw.qualified_name}.")
        own(table)
        pipe = add_pipeline(table, [raw], "Loads")
        out: list[Node] = []
        for src in columns_of[raw.id]:
            col = add_column(table, src.name, src.attrs["data_type"], f"{src.name.replace('_', ' ').capitalize()}, cleaned.")
            derive(pipe, col, [src])
            out.append(col)
        business = [c for c in columns_of[raw.id] if c.name != "updated_at"]
        row_hash = add_column(table, "row_hash", "VARCHAR", "Hash of the business columns, used for change detection.")
        derive(pipe, row_hash, rng.sample(business, k=min(4, len(business))))
        # row_hash is technical: keep it out of downstream models so it is a staging-level dead end.
        columns_of[table.id] = out
        staging.append(table)

    def build_layer(schema: str, layer: str, names: list[str], upstream: list[Node], n_sources: tuple[int, int], n_cols: tuple[int, int], verb: str, measures: bool) -> list[Node]:
        built: list[Node] = []
        for name in names:
            k = min(len(upstream), rng.randint(*n_sources))
            wanted = _tokens(name)
            shuffled = rng.sample(upstream, k=len(upstream))
            # Related inputs first, then anything, so every model still gets k sources.
            sources = sorted(shuffled, key=lambda t: not (wanted & (_tokens(t.name) | t.attrs.get("topics", set()))))[:k]
            table = add_table(schema, name, layer, f"{layer.capitalize()} model {name.replace('_', ' ')} built from {', '.join(s.qualified_name for s in sources)}.")
            own(table)
            pipe = add_pipeline(table, sources, verb)
            candidates = [c for s in sources for c in columns_of[s.id]]
            used_names: set[str] = set()
            out = []
            for _ in range(min(len(candidates), rng.randint(*n_cols))):
                base = rng.choice(candidates)
                numeric = [c for c in candidates if c.attrs["data_type"] in ("DECIMAL", "INTEGER") and c is not base]
                twins = [c for c in candidates if c.name == base.name and c is not base]      # same field from another source
                if measures and base.attrs["data_type"] in ("DECIMAL", "INTEGER") and rng.random() < 0.6:
                    inputs = [base] + ([rng.choice(numeric)] if numeric and rng.random() < 0.35 else [])      # e.g. a ratio of two measures
                    col_name = f"{rng.choice(MEASURE_PREFIXES)}_{base.name}"
                    desc = f"Aggregate computed from {', '.join(dict.fromkeys(i.qualified_name for i in inputs))}."
                    dtype = "DECIMAL"
                else:
                    inputs = [base] + ([rng.choice(twins)] if twins and rng.random() < 0.5 else [])           # e.g. coalesce across sources
                    col_name, dtype = base.name, base.attrs["data_type"]
                    desc = f"{base.name.replace('_', ' ').capitalize()} carried from {base.attrs['table']}."
                if col_name in used_names:
                    continue
                used_names.add(col_name)
                col = add_column(table, col_name, dtype, desc)
                derive(pipe, col, list(dict.fromkeys(inputs)))
                out.append(col)
            # A share of columns is published but never read: true negatives deeper in the graph.
            columns_of[table.id] = [c for c in out if rng.random() >= cfg.unused_fraction] or out[:1]
            table.attrs["topics"] = wanted | set().union(*(_tokens(s.name) for s in sources))
            built.append(table)
        return built

    core = build_layer("core", "core", CORE_NAMES[:cfg.core_tables], staging, (2, 3), (8, 13), "Builds", measures=False)
    marts = build_layer("mart", "mart", MART_NAMES[:cfg.mart_tables], core, (1, 3), (6, 10), "Aggregates", measures=True)

    # ---- reports -------------------------------------------------------------------------------
    for i, title in enumerate(REPORT_TITLES[:cfg.reports]):
        # The first reports sit on the mart of the same subject; the rest combine marts.
        primary = [marts[i]] if i < len(marts) else []
        extra = [m for m in rng.sample(marts, k=min(len(marts), 2)) if m not in primary][: (1 if primary and rng.random() < 0.5 else 2 - len(primary))]
        feeds = primary + extra + (rng.sample(core, k=1) if rng.random() < 0.25 else [])
        cols = [c for t in feeds for c in columns_of[t.id]]
        shown = rng.sample(cols, k=min(len(cols), rng.randint(4, 8)))
        report = cat.add(Node("report", "rpt_" + _snake(title), title,
                              f"{title} dashboard. Reads from {', '.join(t.qualified_name for t in feeds)}.",
                              {"tool": rng.choice(["Looker", "Power BI", "Tableau"]), "refresh": rng.choice(["hourly", "daily", "weekly"])}))
        own(report)
        for c in shown:
            cat.link(c, report, "FLOWS")

    for node in cat.nodes:
        node.attrs.pop("topics", None)      # wiring aid only, not catalog metadata
    return cat

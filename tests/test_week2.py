"""Linking, planning, retrieval, the two systems and scoring.

No real model is called. Model behaviour is scripted (ScriptedClient), and the HTTP
client is exercised against a local server that speaks the OpenAI wire format.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from catalog_graph import CatalogQuery, evaluate
from catalog_graph.documents import build_documents
from catalog_graph.linker import EntityLinker
from catalog_graph.llm import LLMClient, LLMError, extract_json
from catalog_graph.paraphrases import build_paraphrased
from catalog_graph.planner import LLMPlanner, RulePlanner, same_query
from catalog_graph.retrieval import BM25Retriever, EmbeddingRetriever, tokenize
from catalog_graph.systems import DocumentSystem, GraphSystem

from conftest import tiny_catalog


class ScriptedClient:
    """Stands in for a model: returns the next scripted reply and records what it was asked."""

    model, embed_model = "scripted", "scripted-embed"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []
        self.usage = type("U", (), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})()

    def chat(self, messages, **_):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def embed(self, texts, **_):      # a crude bag-of-words vector, enough to rank by shared words
        return [[float(sum(1 for t in tokenize(text) if hash(t) % 64 == i)) for i in range(64)] for text in texts]


# ---- entity linking -------------------------------------------------------------------------

def link(store, question):
    return [(c.qualified_name, c.method) for c in EntityLinker(store).link(question)]


def test_linker_exact_names_and_titles(tiny):
    assert link(tiny, "Which reports depend on crm.customers.email?")[0] == ("crm.customers.email", "exact")
    assert link(tiny, "Who owns the Churn Watchlist report?")[0] == ("rpt_churn", "exact")
    assert link(tiny, "who looks after the executive summary dashboard")[0] == ("rpt_exec", "exact")
    assert link(tiny, "What does pl_core read from?")[0] == ("pl_core", "exact")


def test_linker_composes_a_column_from_table_and_name(tiny):
    assert link(tiny, "the email column in crm.customers")[0] == ("crm.customers.email", "composed")
    assert link(tiny, "column label of the core.dim_customer table")[0] == ("core.dim_customer.label", "composed")
    assert link(tiny, "what uses row hash in stg customers")[0] == ("stg.customers.row_hash", "composed")


def test_linker_prefers_the_column_over_its_table(tiny):
    names = [n for n, _ in link(tiny, "Who owns the table that holds column city of the crm.customers table?")]
    assert names[:2] == ["crm.customers.city", "crm.customers"]


def test_linker_survives_a_typo(tiny):
    assert link(tiny, "anything using crm.customers.legcy_flag?")[0] == ("crm.customers.legacy_flag", "fuzzy")


def test_linker_does_not_confuse_a_system_with_its_schema(tiny):
    assert "crm" not in [n for n, _ in link(tiny, "What columns does crm.customers have?")]
    assert link(tiny, "Which tables come from the crm system?")[0] == ("crm", "exact")
    assert link(tiny, "List the raw tables delivered by crm.")[0] == ("crm", "exact")


def test_linker_returns_nothing_for_unrelated_text(tiny):
    assert link(tiny, "What is the weather like today?") == []


def test_report_title_does_not_fire_on_a_table_with_the_same_words(generated):
    _, store = generated
    names = [c.qualified_name for c in EntityLinker(store).link("Which person owns table mart.supplier_scorecard?")]
    assert names[0] == "mart.supplier_scorecard" and "rpt_supplier_scorecard" not in names


# ---- rule planner ---------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("Who owns the core.dim_customer table?", dict(kind="owner", entity="core.dim_customer")),
    ("Which tables come from the crm system?", dict(kind="members", entity="crm", entity_type="system")),
    ("What columns does stg.customers have?", dict(kind="members", entity="stg.customers")),
    ("Which reports depend on crm.customers.name?", dict(kind="lineage", entity="crm.customers.name", target_type="report")),
    ("Which dashboards show mart.kpi.customers?", dict(kind="lineage", entity="mart.kpi.customers", target_type="report", max_hops=1)),
    ("Which source systems feed the Executive Summary report?", dict(kind="lineage", entity="rpt_exec", direction="upstream", target_type="system")),
    ("Trace mart.kpi.customers back to its raw source columns.", dict(kind="lineage", entity="mart.kpi.customers", direction="upstream", target_type="column", target_layer="raw")),
    ("Who should be told before crm.customers.city is changed?", dict(kind="lineage", entity="crm.customers.city", target_type="owner")),
    ("What are the input tables of pl_core?", dict(kind="lineage", entity="pl_core", direction="upstream", target_type="table", max_hops=1)),
    ("Does anything downstream read stg.customers.row_hash?", dict(kind="lineage", entity="stg.customers.row_hash")),
])
def test_rule_planner_on_the_wording_it_was_written_for(tiny, question, expected):
    plan = RulePlanner(tiny).plan(question)
    assert plan.query == CatalogQuery(**expected)


def test_rule_planner_abstains_instead_of_guessing(tiny):
    plan = RulePlanner(tiny).plan("Blast radius of crm.customers in terms of tables?")
    assert plan.query is None and "no rule" in plan.reason and plan.candidates
    assert RulePlanner(tiny).plan("Who owns the nonexistent.thing table?").query is None


# ---- model planner --------------------------------------------------------------------------

Q = "What breaks in reporting if we drop crm.customers.name?"
GOOD = '{"kind": "lineage", "entity": "crm.customers.name", "direction": "downstream", "target_type": "report"}'


def test_llm_planner_accepts_a_valid_plan_even_inside_prose(tiny):
    client = ScriptedClient(f"Sure! Here is the query:\n```json\n{GOOD}\n```")
    plan = LLMPlanner(tiny, client).plan(Q)
    assert plan.query == CatalogQuery(kind="lineage", entity="crm.customers.name", target_type="report") and plan.attempts == 1
    assert "crm.customers.name" in client.seen[0][1]["content"]      # candidates were offered to the model


def test_llm_planner_retries_once_with_the_validation_error(tiny):
    client = ScriptedClient('{"kind": "lineage", "entity": "crm.customers.name", "max_hops": 0}', GOOD)
    plan = LLMPlanner(tiny, client).plan(Q)
    assert plan.query is not None and plan.attempts == 2
    assert "max_hops" in client.seen[1][-1]["content"]               # the model was told what was wrong


def test_llm_planner_abstains_after_two_invalid_plans(tiny):
    plan = LLMPlanner(tiny, ScriptedClient("I think the answer is rpt_exec.", '{"kind": "lineage"}')).plan(Q)
    assert plan.query is None and "valid query" in plan.reason


@pytest.mark.parametrize("reply", [
    '{"kind": "lineage", "entity": "crm.customers.name", "sql": "DROP TABLE nodes"}',      # unknown field
    '{"kind": "run_sql", "entity": "crm.customers.name"}',                                 # unknown kind
    '{"kind": "lineage", "entity": "core.dim_customer.label", "target_type": "report"}',   # real entity, but not a candidate for this question
    '{"kind": "lineage", "entity": "x\'; DROP TABLE nodes; --"}',
])
def test_llm_planner_rejects_anything_outside_the_query_shape(tiny, reply):
    plan = LLMPlanner(tiny, ScriptedClient(reply, reply)).plan(Q)
    assert plan.query is None
    assert tiny.counts()["column"] == 11


def test_llm_planner_handles_none_errors_and_missing_entities(tiny):
    assert LLMPlanner(tiny, ScriptedClient('{"kind": "none"}')).plan(Q).query is None
    assert "failed" in LLMPlanner(tiny, ScriptedClient(LLMError("connection refused"))).plan(Q).reason
    client = ScriptedClient()
    assert LLMPlanner(tiny, client).plan("What is the weather?").query is None and client.seen == []      # no model call without candidates


def test_llm_planner_marks_systems(tiny):
    plan = LLMPlanner(tiny, ScriptedClient('{"kind": "members", "entity": "crm"}')).plan("What do we ingest from the crm source system?")
    assert plan.query == CatalogQuery(kind="members", entity="crm", entity_type="system")


def test_same_query_treats_titles_and_ids_alike(tiny):
    a = CatalogQuery(kind="owner", entity="Churn Watchlist")
    assert same_query(a, CatalogQuery(kind="owner", entity="rpt_churn"), tiny)
    assert not same_query(a, CatalogQuery(kind="owner", entity="rpt_exec"), tiny)
    assert not same_query(CatalogQuery(kind="lineage", entity="rpt_churn", direction="upstream"), CatalogQuery(kind="lineage", entity="rpt_churn", direction="upstream", max_hops=1), tiny)


@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('noise {"a": {"b": "}"}} trailing', {"a": {"b": "}"}}),
    ('```json\n{"a": "say \\"hi\\" {x}"}\n```', {"a": 'say "hi" {x}'}),
    ('{broken} then {"ok": true}', {"ok": True}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_fails_clearly():
    with pytest.raises(ValueError):
        extract_json("no json here")


# ---- HTTP client against a local OpenAI-compatible server --------------------------------------

@pytest.fixture()
def server():
    state = {"requests": [], "fail_next": 0, "status": 500}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body, self.headers.get("Authorization")))
            if state["fail_next"] > 0:
                state["fail_next"] -= 1
                self.send_response(state["status"]); self.end_headers(); self.wfile.write(b'{"error": "nope"}')
                return
            if self.path.endswith("/chat/completions"):
                out = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
            else:
                out = {"data": [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in reversed(list(enumerate(body["input"])))]}
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(out).encode())

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", state
    httpd.shutdown()


def test_client_chat_and_embeddings(server):
    url, state = server
    client = LLMClient(base_url=url, api_key="k", model="m", embed_model="e")
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert (client.usage.calls, client.usage.prompt_tokens, client.usage.completion_tokens) == (1, 7, 2)
    assert client.embed(["a", "bbb"]) == [[1.0, 1.0], [3.0, 1.0]]      # returned out of order by the server, restored by index
    path, body, auth = state["requests"][0]
    assert path == "/v1/chat/completions" and body["model"] == "m" and body["temperature"] == 0.0 and auth == "Bearer k"


def test_client_retries_server_errors_but_not_bad_requests(server):
    url, state = server
    client = LLMClient(base_url=url, model="m", retries=2)
    state["fail_next"] = 2
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello" and len(state["requests"]) == 3
    state.update(fail_next=1, status=400, requests=[])
    with pytest.raises(LLMError, match="HTTP 400"):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(state["requests"]) == 1


def test_client_needs_a_model_name_and_a_reachable_server():
    with pytest.raises(LLMError, match="CATALOG_LLM_MODEL"):
        LLMClient(model="").chat([])
    with pytest.raises(LLMError):
        LLMClient(base_url="http://127.0.0.1:9/v1", model="m", retries=0, timeout=2).chat([{"role": "user", "content": "x"}])


# ---- retrieval and the document baseline -------------------------------------------------------

@pytest.fixture()
def docs():
    return build_documents(tiny_catalog())


def test_bm25_ranks_the_named_document_first(docs):
    bm25 = BM25Retriever(docs)
    assert bm25.search("Who owns the core.dim_customer table?", k=3)[0]["title"] == "core.dim_customer"
    assert bm25.search("What does pl_mart do?", k=1)[0]["title"] == "pl_mart"
    assert bm25.search("zzz qqq", k=3) == []


def test_embedding_retriever_caches_document_vectors(docs, tmp_path):
    cache = tmp_path / "vectors.json"
    client = ScriptedClient()
    first = EmbeddingRetriever(docs, client, cache=cache).search("pipeline pl_core mappings", k=3)
    assert cache.exists() and len(first) == 3

    class NoDocumentEmbedding(ScriptedClient):
        def embed(self, texts, **kw):
            assert len(texts) == 1, "documents should have come from the cache"
            return super().embed(texts, **kw)

    assert EmbeddingRetriever(docs, NoDocumentEmbedding(), cache=cache).search("pipeline pl_core mappings", k=3) == first


def test_document_system_maps_model_wording_onto_catalog_names(tiny, docs):
    client = ScriptedClient('{"answers": ["Churn Watchlist", "rpt_exec", "Churn Watchlist", "made_up_report", 7]}')
    pred = DocumentSystem(tiny, BM25Retriever(docs), client, k=4).ask("Which reports use crm.customers?")
    assert pred.answers == ["rpt_churn", "rpt_exec", "made_up_report"] and len(pred.retrieved) == 4
    assert "[1]" in client.seen[0][1]["content"]


def test_document_system_abstains_on_unreadable_replies(tiny, docs):
    assert DocumentSystem(tiny, BM25Retriever(docs), ScriptedClient("I am not sure.")).ask("Which reports use crm.customers?").abstained
    assert DocumentSystem(tiny, BM25Retriever(docs), ScriptedClient(LLMError("down"))).ask("x crm.customers").abstained


# ---- scoring -----------------------------------------------------------------------------------

def test_score_sets():
    assert evaluate.score_sets([], []) == (1.0, 1.0, 1.0, True)
    assert evaluate.score_sets(["a"], []) == (0.0, 0.0, 0.0, False)
    assert evaluate.score_sets([], ["a"]) == (0.0, 0.0, 0.0, False)
    assert evaluate.score_sets(["A", "b"], ["a", "c"]) == (0.5, 0.5, 0.5, False)
    assert evaluate.score_sets(["a", "b"], ["b", "a"])[3]


def test_an_abstention_is_never_a_correct_answer(tiny):
    from catalog_graph.questions import Question
    negative = Question("n1", "negative", "Blast radius of crm.customers.legacy_flag?", dict(kind="lineage", entity="crm.customers.legacy_flag"), [], "any")
    rows = evaluate.run(GraphSystem(tiny, RulePlanner(tiny)), [negative], tiny)      # the rules do not know this wording
    assert rows[0].abstained and not rows[0].exact and rows[0].f1 == 0.0
    summary = evaluate.summarise(rows)
    assert summary["negative"]["abstained"] == 1.0 and summary["negative"]["false_positive"] == 0.0


def test_graph_system_end_to_end_with_a_planner_that_is_always_right(generated, question_set):
    """Plumbing check: if the planner returns the gold query, the system scores 100%. Not a model result."""
    _, store = generated
    replies = [json.dumps(q.gold_query) for q in question_set]
    rows = evaluate.run(GraphSystem(store, LLMPlanner(store, ScriptedClient(*replies))), question_set, store)
    summary = evaluate.summarise(rows)
    assert summary["overall"]["exact"] == 1.0 and summary["overall"]["plan_correct"] == 1.0 and summary["negative"]["false_positive"] == 0.0


def test_retrieval_bound_shape_and_meaning(generated, question_set):
    catalog, _ = generated
    bound = evaluate.retrieval_bound(BM25Retriever(build_documents(catalog)), question_set, ks=(5, 20))
    assert set(bound) == {"k=5", "k=20"} and "negative" not in bound["k=5"]
    for group in ("lookup", "one_hop", "multi_hop"):
        assert bound["k=20"][group]["answer_recall_bound"] >= bound["k=5"][group]["answer_recall_bound"]
    # The finding the project is about: one-hop answers are retrievable, multi-hop answers mostly are not.
    assert bound["k=5"]["one_hop"]["answer_recall_bound"] > 0.9
    assert bound["k=5"]["multi_hop"]["answer_recall_bound"] < 0.5


def test_markdown_table(tiny):
    from catalog_graph.questions import Question
    q = Question("l1", "lookup", "Who owns the core.dim_customer table?", dict(kind="owner", entity="core.dim_customer"), ["Omar Haddad"], "owner")
    table = evaluate.markdown_table(evaluate.summarise(evaluate.run(GraphSystem(tiny, RulePlanner(tiny)), [q], tiny)), "demo")
    assert "| lookup | 1 | 1.0 | 1.0 |" in table and "**demo**" in table


# ---- the held-out question set -----------------------------------------------------------------

def test_paraphrased_set_is_consistent_and_differently_worded(generated, question_set):
    catalog, store = generated
    held_out = build_paraphrased(catalog, store)
    assert len(held_out) == 60 and [sum(1 for q in held_out if q.group == g) for g in ("lookup", "one_hop", "multi_hop", "negative")] == [15] * 4
    assert not {q.question for q in held_out} & {q.question for q in question_set}
    for q in held_out:
        assert sorted(store.execute(CatalogQuery(**q.gold_query)).names) == q.answers, q.id
    assert [q.question for q in held_out] == [q.question for q in build_paraphrased(catalog, store)]


def test_linker_on_held_out_wording_is_measured_not_assumed(generated):
    catalog, store = generated
    accuracy = evaluate.linking_accuracy(EntityLinker(store), build_paraphrased(catalog, store), store)
    assert 0.7 < accuracy["overall"]["top1"] <= 1.0

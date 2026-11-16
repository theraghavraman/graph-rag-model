import json

from catalog_graph.cli import main


def test_build_then_query(tmp_path, capsys):
    out = tmp_path / "data"
    assert main(["build", "--out", str(out), "--per-group", "8"]) == 0
    assert {p.name for p in out.iterdir()} == {"catalog.duckdb", "questions.jsonl", "questions_paraphrased.jsonl", "documents.jsonl"}
    first = json.loads((out / "questions.jsonl").read_text().splitlines()[0])
    capsys.readouterr()

    db = str(out / "catalog.duckdb")
    assert main(["query", "--db", db, "--as-json", "--json", json.dumps(first["gold_query"])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert sorted(h["qualified_name"] for h in result["hits"]) == first["answers"]

    assert main(["query", "lineage", "crm.nope.nope", "--db", db]) == 1
    assert "no catalog entity" in capsys.readouterr().err
    assert main(["query", "--db", db, "--json", '{"kind": "lineage", "entity": "x", "max_hops": 0}']) == 2
    assert main(["stats", "--db", db]) == 0


def test_ask_and_eval(tmp_path, capsys):
    data, results = tmp_path / "data", tmp_path / "results"
    assert main(["build", "--out", str(data), "--per-group", "8"]) == 0
    question = json.loads((data / "questions.jsonl").read_text().splitlines()[0])["question"]
    capsys.readouterr()

    assert main(["ask", question, "--data", str(data)]) == 0
    assert "query    {" in capsys.readouterr().out
    assert main(["ask", "Blast radius of everything?", "--data", str(data)]) == 1
    assert "could not work out" in capsys.readouterr().out

    assert main(["eval", "--system", "graph-rules", "--data", str(data), "--out", str(results)]) == 0
    saved = json.loads((results / "graph-rules__questions.json").read_text())
    assert saved["summary"]["overall"]["questions"] == 32 and saved["meta"]["system"] == "graph-rules" and len(saved["rows"]) == 32
    assert main(["eval", "--system", "retrieval-bound", "--data", str(data), "--out", str(results)]) == 0
    assert main(["eval", "--system", "linking", "--data", str(data), "--out", str(results), "--questions", str(data / "questions_paraphrased.jsonl")]) == 0
    assert (results / "linking__questions_paraphrased.json").exists()

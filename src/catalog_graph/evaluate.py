"""Scoring: the same questions, the same metrics, for every system.

Per question, the predicted set of entity names is compared with the gold set:

precision / recall / F1   set overlap; an empty prediction for an empty gold set scores 1.0
exact                     the two sets are identical
false positive            (negative questions only) anything at all was returned

An abstention ("I could not work out what to look up") is never counted as correct, not
even on a negative question: saying nothing is different from establishing that the
answer is "nothing". Abstentions are reported in their own column.

For the graph system the plan is scored too: did the planner produce the gold query?
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .planner import same_query
from .query import CatalogQuery
from .questions import GROUPS, Question
from .retrieval import Retriever
from .store import GraphStore
from .systems import Prediction


@dataclass
class Row:
    id: str
    group: str
    question: str
    gold: list[str]
    predicted: list[str]
    precision: float
    recall: float
    f1: float
    exact: bool
    abstained: bool
    plan_correct: bool | None
    seconds: float
    error: str = ""


def score_sets(predicted: list[str], gold: list[str]) -> tuple[float, float, float, bool]:
    p, g = {x.lower() for x in predicted}, {x.lower() for x in gold}
    if not p and not g:
        return 1.0, 1.0, 1.0, True
    hit = len(p & g)
    precision = hit / len(p) if p else 0.0
    recall = hit / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, p == g


def run(system, questions: list[Question], store: GraphStore | None = None, progress=None) -> list[Row]:
    rows: list[Row] = []
    for i, q in enumerate(questions):
        pred: Prediction = system.ask(q.question)
        precision, recall, f1, exact = (0.0, 0.0, 0.0, False) if pred.abstained else score_sets(pred.answers, q.answers)
        plan_correct = None
        if store is not None and getattr(system, "planner", None) is not None:
            plan_correct = pred.query is not None and same_query(pred.query, CatalogQuery(**q.gold_query), store)
        rows.append(Row(q.id, q.group, q.question, q.answers, pred.answers, precision, recall, f1, exact, pred.abstained, plan_correct, pred.seconds, pred.error))
        if progress:
            progress(i + 1, len(questions))
    return rows


def summarise(rows: list[Row]) -> dict:
    def block(sub: list[Row]) -> dict:
        if not sub:
            return {}
        out = {
            "questions": len(sub),
            "precision": round(statistics.fmean(r.precision for r in sub), 3),
            "recall": round(statistics.fmean(r.recall for r in sub), 3),
            "f1": round(statistics.fmean(r.f1 for r in sub), 3),
            "exact": round(sum(r.exact for r in sub) / len(sub), 3),
            "abstained": round(sum(r.abstained for r in sub) / len(sub), 3),
            "median_ms": round(statistics.median(r.seconds for r in sub) * 1000, 1),
        }
        planned = [r for r in sub if r.plan_correct is not None]
        if planned:
            out["plan_correct"] = round(sum(bool(r.plan_correct) for r in planned) / len(planned), 3)
        negatives = [r for r in sub if not r.gold]
        if negatives:
            out["false_positive"] = round(sum(bool(r.predicted) for r in negatives) / len(negatives), 3)
        return out
    return {"overall": block(rows), **{g: block([r for r in rows if r.group == g]) for g in GROUPS}}


def linking_accuracy(linker, questions: list[Question], store: GraphStore) -> dict:
    """How often the entity linker finds the entity a question is about. No model involved."""
    out: dict = {}
    for group in (*GROUPS, "overall"):
        sub = [q for q in questions if group == "overall" or q.group == group]
        top1 = in_list = 0
        for q in sub:
            gold = store.resolve(q.gold_query["entity"], q.gold_query.get("entity_type"))[0]
            ids = [c.id for c in linker.link(q.question)]
            top1 += bool(ids) and ids[0] == gold
            in_list += gold in ids
        out[group] = {"questions": len(sub), "top1": round(top1 / len(sub), 3), "in_candidates": round(in_list / len(sub), 3)}
    return out


def retrieval_bound(retriever: Retriever, questions: list[Question], ks: tuple[int, ...] = (5, 10, 20)) -> dict:
    """An upper limit for ANY reader on top of this retriever, with no model involved.

    For each question with a non-empty answer, what share of the gold answer names appear
    anywhere in the top-k retrieved documents? A reader cannot list a name it was never
    shown, so its recall cannot exceed this. It is generous: a name being present does not
    mean a reader could tell it is an answer.
    """
    out: dict = {}
    for k in ks:
        per_group: dict[str, list[float]] = {g: [] for g in GROUPS}
        complete: dict[str, list[bool]] = {g: [] for g in GROUPS}
        for q in questions:
            if not q.answers:
                continue
            text = " ".join(d["text"] for d in retriever.search(q.question, k=k)).lower()
            found = [a.lower() in text for a in q.answers]
            per_group[q.group].append(sum(found) / len(found))
            complete[q.group].append(all(found))
        out[f"k={k}"] = {g: {"questions": len(v), "answer_recall_bound": round(statistics.fmean(v), 3), "all_answers_present": round(sum(complete[g]) / len(v), 3)}
                         for g, v in per_group.items() if v}
    return out


def markdown_table(summary: dict, title: str) -> str:
    cols = [("questions", "n"), ("plan_correct", "plan correct"), ("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
            ("exact", "exact"), ("false_positive", "false positives"), ("abstained", "abstained"), ("median_ms", "median ms")]
    used = [c for c in cols if any(c[0] in v for v in summary.values())]
    lines = [f"**{title}**", "", "| group | " + " | ".join(label for _, label in used) + " |", "|---|" + "---:|" * len(used)]
    for group in (*GROUPS, "overall"):
        block = summary.get(group) or {}
        if block:
            lines.append(f"| {group} | " + " | ".join(str(block.get(key, "")) for key, _ in used) + " |")
    return "\n".join(lines)


def save(rows: list[Row], summary: dict, out_dir: str | Path, name: str, meta: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps({"meta": {**meta, "ran_at": time.strftime("%Y-%m-%d")}, "summary": summary, "rows": [asdict(r) for r in rows]}, indent=1, ensure_ascii=False))
    return path

"""Turning a query result into an answer a person can check.

The default answer is a template: it lists every result and shows the evidence path for
the first few. It cannot invent anything. A language model may rephrase it, but only
from the same results, and the structured answer set (used for scoring) never comes
from the model.
"""

from __future__ import annotations

from .llm import LLMClient, LLMError
from .planner import Plan
from .query import QueryResult

PLURAL = {"report": "reports", "table": "tables", "column": "columns", "pipeline": "pipelines", "system": "source systems", "owner": "owners", None: "downstream objects"}


def describe(query) -> str:
    if query.kind == "owner":
        return f"owner of {query.entity}"
    if query.kind == "members":
        return f"contents of {query.entity}"
    what = PLURAL[query.target_type]
    if query.target_layer:
        what = f"{query.target_layer} {what}"
    reach = "directly " if query.max_hops == 1 else ""
    return f"{what} {reach}{'downstream of' if query.direction == 'downstream' else 'upstream of'} {query.entity}"


def render(result: QueryResult, show_paths: int = 3) -> str:
    """Deterministic answer text with evidence."""
    what = describe(result.query)
    if result.empty:
        return f"Nothing found: no {what}."
    lines = [f"{len(result.hits)} result{'s' if len(result.hits) != 1 else ''} for {what}:"]
    lines += [f"  - {h.qualified_name}" + (f" ({h.hops} step{'s' if h.hops != 1 else ''})" if result.query.kind == "lineage" else "") for h in result.hits]
    shown = [h for h in result.hits if len(h.path) > 2][:show_paths]
    if shown:
        lines.append("Evidence:")
        lines += ["  " + " -> ".join(h.path) for h in shown]
    return "\n".join(lines)


def abstain(plan: Plan) -> str:
    text = f"I could not work out what to look up: {plan.reason}."
    if plan.candidates:
        text += " Closest catalog entries: " + ", ".join(c.qualified_name for c in plan.candidates[:4]) + "."
    return text


def phrase(client: LLMClient, question: str, result: QueryResult) -> str:
    """Optional: let a model word the answer. Falls back to the template if the call fails."""
    facts = render(result, show_paths=5)
    try:
        return client.chat([
            {"role": "system", "content": "Answer the question in two or three plain sentences using ONLY the facts given. Name every result. If the facts say nothing was found, say so. Do not add anything that is not in the facts."},
            {"role": "user", "content": f"Question: {question}\n\nFacts:\n{facts}"}], max_tokens=300).strip() or facts
    except LLMError:
        return facts

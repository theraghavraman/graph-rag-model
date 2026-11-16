"""The only query shapes the system will execute.

A language model never writes SQL or graph queries here. Its job (week 2) is to fill in
this small structure; anything that does not validate is rejected before it reaches the
database. That keeps the answers exact and the attack surface small.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

EntityType = Literal["system", "table", "column", "pipeline", "report", "owner"]
MAX_HOPS = 64


class CatalogQuery(BaseModel):
    """One question about the catalog.

    kind = "owner"    who owns ``entity`` (a column resolves to its table's owner)
    kind = "members"  what ``entity`` contains: tables of a system, columns of a table
    kind = "lineage"  what ``entity`` feeds (downstream) or is built from (upstream)

    ``max_hops`` counts processing steps: one pipeline run from a column to the column
    it produces is 1 hop, and a column shown on a report is 1 hop. A pipeline in the
    results has the hop number of the step it performs. A table as the start means
    "any of its columns"; a pipeline as the start means "the columns it writes"
    (downstream) or "the columns it reads" (upstream).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["owner", "members", "lineage"]
    entity: str = Field(min_length=1, max_length=300, description="Qualified name, e.g. crm.customers.email")
    entity_type: Optional[EntityType] = Field(default=None, description="Set only to disambiguate a name shared by two node types")
    direction: Literal["upstream", "downstream"] = "downstream"
    target_type: Optional[EntityType] = Field(default=None, description="Keep only results of this type")
    target_layer: Optional[Literal["raw", "staging", "core", "mart"]] = Field(default=None, description="With target_type table or column: keep only this warehouse layer, e.g. raw for 'original source columns'")
    max_hops: int = Field(default=MAX_HOPS, ge=1, le=MAX_HOPS)

    @model_validator(mode="after")
    def _shape(self) -> "CatalogQuery":
        if self.kind != "lineage" and (self.target_type is not None or self.target_layer is not None or self.direction != "downstream" or self.max_hops != MAX_HOPS):
            raise ValueError("direction, target_type, target_layer and max_hops apply to lineage queries only")
        if self.target_layer is not None and self.target_type not in ("table", "column"):
            raise ValueError("target_layer needs target_type 'table' or 'column'")
        return self


class Hit(BaseModel):
    id: str
    type: str
    qualified_name: str
    name: str
    hops: int = 0
    path: list[str] = Field(default_factory=list, description="Qualified names from the start to this result, in data-flow order of traversal")


class QueryResult(BaseModel):
    query: CatalogQuery
    start: list[str] = Field(description="Node ids the traversal started from")
    hits: list[Hit]

    @property
    def names(self) -> list[str]:
        return [h.qualified_name for h in self.hits]

    @property
    def empty(self) -> bool:
        return not self.hits

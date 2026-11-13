"""Contratos Pydantic del subsistema RAG."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(default=1, ge=1)
    category: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    category: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    chunk_index: int = Field(ge=0)

    def metadata(self) -> dict[str, Any]:
        return self.model_dump(exclude={"content"})


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=4_000)
    limit: int = Field(default=5, ge=2, le=10)


class RetrievedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    content: str
    source: str
    page: int = Field(ge=1)
    category: str
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    semantic_score: float = Field(ge=-1, le=1)
    lexical_score: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)


class RetrieverStats(BaseModel):
    backend: str
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    embedding_model: str

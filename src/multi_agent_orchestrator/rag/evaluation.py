"""Evaluacion reproducible de Precision@k y Recall@k."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from multi_agent_orchestrator.rag.schemas import RetrievedDocument


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3)
    relevant_document_ids: list[str] = Field(min_length=1)


class QueryMetrics(BaseModel):
    question: str
    retrieved_document_ids: list[str]
    relevant_document_ids: list[str]
    hits: int = Field(ge=0)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)


class EvaluationReport(BaseModel):
    k: int = Field(ge=1)
    queries: int = Field(ge=1)
    mean_precision_at_k: float = Field(ge=0, le=1)
    mean_recall_at_k: float = Field(ge=0, le=1)
    results: list[QueryMetrics]


class Retriever(Protocol):
    async def retrieve(
        self, query: str, *, limit: int | None = None
    ) -> list[RetrievedDocument]: ...


def load_golden_set(path: Path) -> list[GoldenCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("El Golden Set debe ser una lista JSON")
    cases = [GoldenCase.model_validate(item) for item in raw]
    if len(cases) < 5:
        raise ValueError("El Golden Set debe contener al menos cinco preguntas")
    return cases


async def evaluate_retrieval(
    retriever: Retriever,
    cases: list[GoldenCase],
    *,
    k: int = 5,
) -> EvaluationReport:
    if not cases:
        raise ValueError("Se requiere al menos un caso de evaluacion")
    rows: list[QueryMetrics] = []
    for case in cases:
        retrieved = await retriever.retrieve(case.question, limit=k)
        retrieved_ids = [item.document_id for item in retrieved[:k]]
        relevant = set(case.relevant_document_ids)
        hits = len(set(retrieved_ids) & relevant)
        rows.append(
            QueryMetrics(
                question=case.question,
                retrieved_document_ids=retrieved_ids,
                relevant_document_ids=case.relevant_document_ids,
                hits=hits,
                precision_at_k=hits / k,
                recall_at_k=hits / len(relevant),
            )
        )
    return EvaluationReport(
        k=k,
        queries=len(rows),
        mean_precision_at_k=sum(row.precision_at_k for row in rows) / len(rows),
        mean_recall_at_k=sum(row.recall_at_k for row in rows) / len(rows),
        results=rows,
    )

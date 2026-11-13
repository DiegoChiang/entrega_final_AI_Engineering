"""Herramientas validadas de recuperacion y calculo."""

from __future__ import annotations

import json
from typing import Any, Protocol

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from multi_agent_orchestrator.rag.schemas import RetrievalQuery


class Retriever(Protocol):
    async def retrieve(self, query: str, *, limit: int | None = None) -> list[Any]: ...


class PriorityInput(BaseModel):
    error_rate_pct: float = Field(ge=0, le=100)
    latency_p95_ms: float = Field(ge=0)
    affected_users: int = Field(ge=0)
    duration_minutes: float = Field(ge=0)


def build_search_knowledge_tool(retriever: Retriever) -> BaseTool:
    """Crea una herramienta RAG acotada al recuperador inicializado."""

    @tool("search_incident_knowledge", args_schema=RetrievalQuery)
    async def search_incident_knowledge(query: str, limit: int = 5) -> str:
        """Recupera evidencia técnica mediante búsqueda híbrida semántica y BM25."""

        results = await retriever.retrieve(query, limit=limit)
        return json.dumps(
            [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item
                for item in results
            ],
            ensure_ascii=False,
        )

    return search_incident_knowledge


@tool(args_schema=PriorityInput)
def calculate_incident_priority(
    error_rate_pct: float,
    latency_p95_ms: float,
    affected_users: int,
    duration_minutes: float,
) -> str:
    """Calcula un puntaje reproducible y una prioridad operativa P1-P4."""

    error_component = min(error_rate_pct / 15.0, 1.0) * 35
    latency_component = min(latency_p95_ms / 3000.0, 1.0) * 25
    users_component = min(affected_users / 2000.0, 1.0) * 25
    duration_component = min(duration_minutes / 30.0, 1.0) * 15
    score = round(error_component + latency_component + users_component + duration_component, 1)
    if score >= 75:
        priority = "P1"
    elif score >= 50:
        priority = "P2"
    elif score >= 25:
        priority = "P3"
    else:
        priority = "P4"
    return json.dumps(
        {
            "priority": priority,
            "risk_score": score,
            "components": {
                "errors": round(error_component, 1),
                "latency": round(latency_component, 1),
                "users": round(users_component, 1),
                "duration": round(duration_component, 1),
            },
        },
        ensure_ascii=False,
    )

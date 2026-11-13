"""Especialista de investigacion conectado al RAG híbrido."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from multi_agent_orchestrator.state import OrchestratorState, ResearchFinding
from multi_agent_orchestrator.tools import Retriever, build_search_knowledge_tool


class ResearchNarrative(BaseModel):
    summary: str = Field(min_length=20)
    focus_areas: list[str] = Field(min_length=1, max_length=4)


class ResearchAgent:
    """Recupera fuentes reales y entrega evidencia estructurada al Supervisor."""

    def __init__(self, model: Any, retriever: Retriever) -> None:
        self._writer = model.with_structured_output(ResearchNarrative)
        self._search_tool = build_search_knowledge_tool(retriever)
        self.tools = (self._search_tool,)

    async def __call__(self, state: OrchestratorState) -> dict[str, Any]:
        tool_output = await self._search_tool.ainvoke(
            {"query": state["request"], "limit": 5}
        )
        rows: list[dict[str, Any]] = json.loads(tool_output)
        findings: list[ResearchFinding] = [
            {
                "source_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "title": row["title"],
                "summary": row["content"],
                "source": row["source"],
                "page": int(row["page"]),
                "metrics": {
                    key: float(value) for key, value in row.get("metrics", {}).items()
                },
                "tags": list(row.get("tags", [])),
                "semantic_score": float(row["semantic_score"]),
                "lexical_score": float(row["lexical_score"]),
                "combined_score": float(row["combined_score"]),
            }
            for row in rows
        ]
        if len({item["source_id"] for item in findings}) < 2:
            raise RuntimeError("El RAG no recupero al menos dos fuentes diferenciadas")
        try:
            narrative = await self._writer.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Eres especialista en investigacion operativa. Resume los hallazgos "
                            "sin inventar fuentes ni metricas. Identifica los focos que debe "
                            "revisar el analista."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Instruccion del supervisor: {state['supervisor_instruction']}\n"
                            f"Solicitud: {state['request']}\n"
                            f"Evidencia RAG: {json.dumps(findings, ensure_ascii=False)}"
                        )
                    ),
                ]
            )
        except ValidationError:
            narrative = ResearchNarrative(
                summary=(
                    f"Se recuperaron {len(findings)} fuentes documentales para analizar "
                    f"{state['request']}"
                ),
                focus_areas=list(
                    dict.fromkeys(tag for item in findings for tag in item["tags"])
                )[:4]
                or ["evidencia"],
            )
        source_ids = [item["source_id"] for item in findings]
        return {
            "research_findings": findings,
            "contributions": [
                {
                    "agent": "researcher",
                    "summary": narrative.summary,
                    "evidence_ids": source_ids,
                }
            ],
            "completed_agents": ["researcher"],
            "validation_feedback": [],
            "task_completed": False,
        }

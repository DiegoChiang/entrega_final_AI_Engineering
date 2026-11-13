"""Dobles de prueba para ejecutar el grafo sin red."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from multi_agent_orchestrator.rag.schemas import RetrievedDocument


class FakeRetriever:
    async def retrieve(
        self, query: str, *, limit: int | None = None
    ) -> list[RetrievedDocument]:
        del query
        rows = [
            RetrievedDocument(
                chunk_id="chunk-runbook",
                document_id="runbook-checkout-01",
                title="Runbook de checkout",
                content="Errores de 12.4% durante 18 minutos requieren escalamiento.",
                source="runbooks/checkout.md",
                page=1,
                category="runbook",
                tags=["checkout", "pagos"],
                metrics={"error_rate_pct": 12.4, "duration_minutes": 18.0},
                semantic_score=0.93,
                lexical_score=1.0,
                combined_score=0.96,
            ),
            RetrievedDocument(
                chunk_id="chunk-telemetry",
                document_id="telemetry-checkout-02",
                title="Telemetria de checkout",
                content="La latencia p95 fue 2450 ms y hubo 1840 usuarios afectados.",
                source="telemetry/checkout.json",
                page=1,
                category="telemetry",
                tags=["checkout", "latencia"],
                metrics={"latency_p95_ms": 2450.0, "affected_users": 1840.0},
                semantic_score=0.91,
                lexical_score=0.9,
                combined_score=0.91,
            ),
            RetrievedDocument(
                chunk_id="chunk-postmortem",
                document_id="postmortem-payments-03",
                title="Postmortem de pagos",
                content="La mitigacion limito reintentos y amplio el pool.",
                source="postmortems/payments.md",
                page=2,
                category="postmortem",
                tags=["pagos", "mitigacion"],
                metrics={"recovery_minutes": 27.0},
                semantic_score=0.84,
                lexical_score=0.7,
                combined_score=0.78,
            ),
        ]
        return rows[: limit or 5]


class ScriptedStructuredModel:
    def __init__(self, schema: type[Any]) -> None:
        self.schema = schema

    async def ainvoke(self, messages: list[Any]) -> Any:
        schema_name = self.schema.__name__
        if schema_name == "RouteDecision":
            status = messages[-1].content
            if '"research_sources": []' in status:
                target = "researcher"
            elif '"analysis_result": null' in status:
                target = "analyst"
            elif '"task_completed": false' in status:
                target = "validator"
            else:
                target = "synthesizer"
            return self.schema(
                next_agent=target,
                reasoning=f"El estado requiere {target}.",
                instruction=f"Ejecuta la responsabilidad de {target} con los datos disponibles.",
            )
        if schema_name == "ResearchNarrative":
            return self.schema(
                summary=(
                    "La evidencia vincula errores sostenidos, alta latencia y afectacion de "
                    "usuarios."
                ),
                focus_areas=["errores", "latencia", "usuarios afectados"],
            )
        if schema_name == "AnalysisNarrative":
            return self.schema(
                drivers=["Tasa de errores sostenida", "Latencia p95 elevada"],
                recommendation=(
                    "Escalar al equipo de pagos, limitar reintentos y vigilar la recuperacion."
                ),
            )
        raise AssertionError(f"Esquema no programado: {schema_name}")


class ScriptedModel:
    def with_structured_output(self, schema: type[Any]) -> ScriptedStructuredModel:
        return ScriptedStructuredModel(schema)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(
            content=(
                "Prioridad P1 (81.3/100). Escalar al equipo de pagos, limitar reintentos y "
                "monitorear la recuperacion [runbook-checkout-01] "
                "[telemetry-checkout-02] [postmortem-payments-03]."
            )
        )

import json

from multi_agent_orchestrator.tools import (
    build_search_knowledge_tool,
    calculate_incident_priority,
)
from tests.fakes import FakeRetriever


async def test_research_tool_returns_validated_rag_sources() -> None:
    tool = build_search_knowledge_tool(FakeRetriever())
    payload = await tool.ainvoke({"query": "errores checkout pagos latencia", "limit": 3})
    findings = json.loads(payload)

    assert len(findings) == 3
    assert "runbook-checkout-01" in {item["document_id"] for item in findings}
    assert all(item["combined_score"] > 0 for item in findings)


def test_priority_tool_is_reproducible() -> None:
    payload = calculate_incident_priority.invoke(
        {
            "error_rate_pct": 12.4,
            "latency_p95_ms": 2450,
            "affected_users": 1840,
            "duration_minutes": 18,
        }
    )
    result = json.loads(payload)

    assert result["priority"] == "P1"
    assert result["risk_score"] == 81.3
    assert set(result["components"]) == {"errors", "latency", "users", "duration"}

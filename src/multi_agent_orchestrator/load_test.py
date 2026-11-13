"""Prueba de cinco peticiones concurrentes contra la API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from multi_agent_orchestrator.api.schemas import TERMINAL_STATUSES, JobStatus

SCENARIOS = (
    "Investiga el aumento de errores en checkout, calcula su prioridad y propone acciones "
    "inmediatas basadas en evidencia.",
    "Analiza el atraso del indice de busqueda de productos y determina una respuesta operativa.",
    "Investiga la presion de memoria en Redis, estima el riesgo y recomienda una mitigacion.",
    "Compara la saturacion del adaptador de pagos con incidentes previos y prioriza la respuesta.",
    "Determina cuando un incidente requiere aprobacion humana y sustenta la decision con fuentes.",
)


async def submit_and_wait(
    client: httpx.AsyncClient, index: int, query: str
) -> dict[str, Any]:
    started = perf_counter()
    response = await client.post("/jobs", json={"request": f"Caso {index}: {query}"})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    approved = False

    while True:
        status_response = await get_with_retry(client, f"/jobs/{job_id}")
        status_response.raise_for_status()
        job = status_response.json()
        status = JobStatus(job["status"])
        if status == JobStatus.WAITING_APPROVAL and not approved:
            approval = await client.post(
                f"/jobs/{job_id}/approval",
                json={"approved": True, "feedback": "Aprobado durante la prueba de carga."},
            )
            approval.raise_for_status()
            approved = True
        elif status in TERMINAL_STATUSES:
            return {
                "job_id": job_id,
                "status": status,
                "latency_seconds": round(perf_counter() - started, 3),
                "approval_used": approved,
                "source_document_ids": [
                    source["document_id"] for source in (job.get("result") or {}).get("sources", [])
                ],
            }
        await asyncio.sleep(1)


async def get_with_retry(
    client: httpx.AsyncClient, path: str, attempts: int = 5
) -> httpx.Response:
    """Tolera cierres transitorios del transporte durante el sondeo."""

    for attempt in range(attempts):
        try:
            return await client.get(path)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))
    raise RuntimeError("Numero de intentos de transporte agotado")


async def run_load_test(base_url: str) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    async with httpx.AsyncClient(base_url=base_url, timeout=180, limits=limits) as client:
        results = await asyncio.gather(
            *(
                submit_and_wait(client, index, query)
                for index, query in enumerate(SCENARIOS, start=1)
            )
        )
    latencies = sorted(float(result["latency_seconds"]) for result in results)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "request_count": len(results),
        "concurrency": 5,
        "p95_latency_seconds": latencies[p95_index],
        "all_completed": all(
            result["status"] == JobStatus.COMPLETED for result in results
        ),
        "jobs": results,
        "phoenix_dashboard": "http://localhost:6006",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cinco escenarios concurrentes con RAG y HITL")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final-system-tests.json"),
    )
    args = parser.parse_args()
    result = asyncio.run(run_load_test(args.base_url))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

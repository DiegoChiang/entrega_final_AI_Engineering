"""Ejecuta el Golden Set contra el RAG configurado."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from multi_agent_orchestrator.config import get_settings
from multi_agent_orchestrator.rag.evaluation import evaluate_retrieval, load_golden_set
from multi_agent_orchestrator.rag.service import build_retriever


async def run(golden_set: Path, output: Path) -> None:
    settings = get_settings()
    retriever = await build_retriever(settings)
    cases = await asyncio.to_thread(load_golden_set, golden_set)
    report = await evaluate_retrieval(retriever, cases, k=5)

    def write_report() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    await asyncio.to_thread(write_report)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua la recuperacion hibrida")
    parser.add_argument("--golden-set", type=Path, default=Path("evaluation/golden_set.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rag-evaluation.json"),
    )
    arguments = parser.parse_args()
    asyncio.run(run(arguments.golden_set, arguments.output))


if __name__ == "__main__":
    main()

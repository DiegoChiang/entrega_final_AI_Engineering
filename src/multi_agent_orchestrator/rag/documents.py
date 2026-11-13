"""Carga validada de documentos Markdown, texto, JSON y PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multi_agent_orchestrator.rag.schemas import KnowledgeDocument

SUPPORTED_SUFFIXES = {".md", ".txt", ".json", ".pdf"}


def _relative_source(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_text(path: Path, root: Path) -> list[KnowledgeDocument]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [
        KnowledgeDocument(
            document_id=path.stem,
            title=path.stem.replace("-", " ").replace("_", " ").title(),
            content=content,
            source=_relative_source(path, root),
            category=path.suffix.lstrip(".").casefold(),
            tags=[part for part in path.stem.replace("-", "_").split("_") if part],
        )
    ]


def _load_json(path: Path, root: Path) -> list[KnowledgeDocument]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("documents", []) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(f"El JSON debe contener una lista de documentos: {path}")
    documents: list[KnowledgeDocument] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Documento JSON invalido en posicion {index}: {path}")
        payload = dict(record)
        payload.setdefault("document_id", f"{path.stem}-{index}")
        payload.setdefault("title", payload["document_id"])
        payload.setdefault("source", _relative_source(path, root))
        payload.setdefault("category", "json")
        documents.append(KnowledgeDocument.model_validate(payload))
    return documents


def _load_pdf(path: Path, root: Path) -> list[KnowledgeDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depende de la instalacion
        raise RuntimeError("Instala la dependencia 'pypdf'") from exc
    documents: list[KnowledgeDocument] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            documents.append(
                KnowledgeDocument(
                    document_id=path.stem,
                    title=path.stem.replace("-", " ").title(),
                    content=content,
                    source=_relative_source(path, root),
                    page=page_number,
                    category="pdf",
                    tags=[path.stem.casefold()],
                )
            )
    return documents


def load_documents(data_dir: Path) -> list[KnowledgeDocument]:
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de conocimiento: {data_dir}")
    documents: list[KnowledgeDocument] = []
    for path in sorted(item for item in data_dir.rglob("*") if item.is_file()):
        suffix = path.suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix in {".md", ".txt"}:
            documents.extend(_load_text(path, data_dir))
        elif suffix == ".json":
            documents.extend(_load_json(path, data_dir))
        else:
            documents.extend(_load_pdf(path, data_dir))
    if not documents:
        raise ValueError(f"No se encontraron documentos validos en {data_dir}")
    return documents

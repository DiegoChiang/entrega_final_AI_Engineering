"""Fragmentacion determinista con solapamiento y metadatos."""

from __future__ import annotations

import hashlib

from multi_agent_orchestrator.rag.schemas import KnowledgeChunk, KnowledgeDocument


def _split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    clean = " ".join(text.split())
    if len(clean) <= chunk_size:
        return [clean]
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + max(1, chunk_size // 2), end)
            if boundary > start:
                end = boundary
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def split_documents(
    documents: list[KnowledgeDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe estar entre 0 y chunk_size")
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for index, content in enumerate(
            _split_text(document.content, chunk_size=chunk_size, overlap=chunk_overlap)
        ):
            digest = hashlib.sha256(
                f"{document.document_id}:{document.page}:{index}:{content}".encode()
            ).hexdigest()
            chunks.append(
                KnowledgeChunk(
                    chunk_id=digest,
                    document_id=document.document_id,
                    title=document.title,
                    content=content,
                    source=document.source,
                    page=document.page,
                    category=document.category,
                    tags=document.tags,
                    metrics=document.metrics,
                    chunk_index=index,
                )
            )
    return chunks

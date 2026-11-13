"""Backends semanticos local y Pinecone con contratos asíncronos."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Protocol

from multi_agent_orchestrator.config import Settings
from multi_agent_orchestrator.rag.schemas import KnowledgeChunk


class AsyncEmbeddings(Protocol):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


class SemanticBackend(Protocol):
    async def initialize(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: AsyncEmbeddings,
        *,
        fingerprint: str,
    ) -> None: ...

    async def search(self, vector: list[float], *, k: int) -> list[tuple[str, float]]: ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class LocalVectorBackend:
    """Indice denso persistido en JSON para despliegue local reproducible."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._vectors: dict[str, list[float]] = {}

    async def initialize(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: AsyncEmbeddings,
        *,
        fingerprint: str,
    ) -> None:
        cached = await asyncio.to_thread(self._load, fingerprint)
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        if cached is not None and set(cached) == chunk_ids:
            self._vectors = cached
            return
        vectors = await embeddings.aembed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("La cantidad de embeddings no coincide con los fragmentos")
        self._vectors = {
            chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors, strict=True)
        }
        await asyncio.to_thread(self._save, fingerprint)

    async def search(self, vector: list[float], *, k: int) -> list[tuple[str, float]]:
        ranked = [
            (chunk_id, cosine_similarity(vector, candidate))
            for chunk_id, candidate in self._vectors.items()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def _load(self, fingerprint: str) -> dict[str, list[float]] | None:
        if not self._path.exists():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        vectors = payload.get("vectors")
        if not isinstance(vectors, dict):
            return None
        return {str(key): [float(value) for value in values] for key, values in vectors.items()}

    def _save(self, fingerprint: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"fingerprint": fingerprint, "vectors": self._vectors}),
            encoding="utf-8",
        )
        temporary.replace(self._path)


class PineconeVectorBackend:
    """Indice Pinecone Serverless sin bloquear el event loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index: Any | None = None

    async def initialize(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: AsyncEmbeddings,
        *,
        fingerprint: str,
    ) -> None:
        del fingerprint
        self._index = await asyncio.to_thread(self._ensure_index)
        vectors = await embeddings.aembed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("La cantidad de embeddings no coincide con los fragmentos")
        records = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            metadata = chunk.metadata()
            metadata["metrics"] = json.dumps(chunk.metrics, ensure_ascii=False)
            records.append(
                {"id": chunk.chunk_id, "values": vector, "metadata": metadata}
            )
        for start in range(0, len(records), 100):
            batch = records[start : start + 100]
            await asyncio.to_thread(
                self._index.upsert,
                vectors=batch,
                namespace=self._settings.pinecone_namespace,
            )

    async def search(self, vector: list[float], *, k: int) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("El indice Pinecone no fue inicializado")
        response = await asyncio.to_thread(
            self._index.query,
            vector=vector,
            top_k=k,
            namespace=self._settings.pinecone_namespace,
            include_metadata=False,
        )
        matches = getattr(response, "matches", None)
        if matches is None:
            matches = response.get("matches", [])
        rows: list[tuple[str, float]] = []
        for match in matches:
            if isinstance(match, dict):
                rows.append((str(match.get("id", "")), float(match.get("score", 0.0))))
            else:
                rows.append((str(match.id), float(match.score)))
        return rows

    def _ensure_index(self) -> Any:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover - depende de la instalacion
            raise RuntimeError("Instala la dependencia 'pinecone'") from exc
        client = Pinecone(api_key=self._settings.require_pinecone_key())
        names = set(client.list_indexes().names())
        if self._settings.pinecone_index_name not in names:
            client.create_index(
                name=self._settings.pinecone_index_name,
                dimension=self._settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
        description = client.describe_index(self._settings.pinecone_index_name)
        dimension = int(getattr(description, "dimension", self._settings.embedding_dimension))
        metric = str(getattr(description, "metric", "cosine"))
        if dimension != self._settings.embedding_dimension or metric != "cosine":
            raise ValueError("La dimension o metrica del indice Pinecone no coincide")
        return client.Index(self._settings.pinecone_index_name)

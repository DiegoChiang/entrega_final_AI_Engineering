"""Recuperacion híbrida semantica y lexical con trazabilidad."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

import httpx
from opentelemetry import trace
from rank_bm25 import BM25Okapi

from multi_agent_orchestrator.config import Settings
from multi_agent_orchestrator.rag.backends import (
    AsyncEmbeddings,
    LocalVectorBackend,
    PineconeVectorBackend,
    SemanticBackend,
)
from multi_agent_orchestrator.rag.chunking import split_documents
from multi_agent_orchestrator.rag.documents import load_documents
from multi_agent_orchestrator.rag.schemas import (
    KnowledgeChunk,
    RetrievalQuery,
    RetrievedDocument,
    RetrieverStats,
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9áéíóúñ]+", text.casefold())


class HybridRetriever:
    """Fusiona similitud densa y BM25 sobre el mismo corpus validado."""

    def __init__(
        self,
        embeddings: AsyncEmbeddings,
        semantic_backend: SemanticBackend,
        *,
        backend_name: str,
        embedding_model: str,
        top_k: int,
        candidate_k: int,
        semantic_weight: float,
        lexical_weight: float,
    ) -> None:
        self._embeddings = embeddings
        self._semantic_backend = semantic_backend
        self._backend_name = backend_name
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._semantic_weight = semantic_weight
        self._lexical_weight = lexical_weight
        self._chunks: list[KnowledgeChunk] = []
        self._chunks_by_id: dict[str, KnowledgeChunk] = {}
        self._bm25: BM25Okapi | None = None
        self._document_count = 0

    @property
    def stats(self) -> RetrieverStats:
        return RetrieverStats(
            backend=self._backend_name,
            documents=self._document_count,
            chunks=len(self._chunks),
            embedding_model=self._embedding_model,
        )

    async def initialize(
        self,
        chunks: list[KnowledgeChunk],
        *,
        document_count: int,
    ) -> None:
        if not chunks:
            raise ValueError("El RAG requiere al menos un fragmento")
        fingerprint = hashlib.sha256(
            json.dumps(
                [chunk.model_dump(mode="json") for chunk in chunks],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        tracer = trace.get_tracer("multi_agent_orchestrator.rag")
        with tracer.start_as_current_span("rag.initialize") as span:
            await self._semantic_backend.initialize(
                chunks,
                self._embeddings,
                fingerprint=fingerprint,
            )
            self._chunks = chunks
            self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            self._bm25 = await asyncio.to_thread(
                BM25Okapi,
                [_tokens(chunk.content) for chunk in chunks],
            )
            self._document_count = document_count
            span.set_attribute("rag.backend", self._backend_name)
            span.set_attribute("rag.documents", document_count)
            span.set_attribute("rag.chunks", len(chunks))

    async def retrieve(self, query: str, *, limit: int | None = None) -> list[RetrievedDocument]:
        request = RetrievalQuery(query=query.strip(), limit=limit or self._top_k)
        if self._bm25 is None:
            raise RuntimeError("El RAG no fue inicializado")
        tracer = trace.get_tracer("multi_agent_orchestrator.rag")
        with tracer.start_as_current_span("rag.retrieve") as span:
            vector = await self._embeddings.aembed_query(request.query)
            semantic_rows = await self._semantic_backend.search(
                vector,
                k=self._candidate_k,
            )
            lexical_values = await asyncio.to_thread(self._bm25.get_scores, _tokens(request.query))
            lexical_max = max((float(value) for value in lexical_values), default=0.0)
            lexical_scores = {
                chunk.chunk_id: (float(score) / lexical_max if lexical_max > 0 else 0.0)
                for chunk, score in zip(self._chunks, lexical_values, strict=True)
            }
            semantic_scores = dict(semantic_rows)
            candidates = set(semantic_scores)
            lexical_ranked = sorted(
                lexical_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self._candidate_k]
            candidates.update(chunk_id for chunk_id, _ in lexical_ranked)
            ranked = sorted(
                candidates,
                key=lambda chunk_id: (
                    self._semantic_weight * max(0.0, semantic_scores.get(chunk_id, 0.0))
                    + self._lexical_weight * lexical_scores.get(chunk_id, 0.0)
                ),
                reverse=True,
            )
            results: list[RetrievedDocument] = []
            seen_documents: set[str] = set()
            for chunk_id in ranked:
                chunk = self._chunks_by_id.get(chunk_id)
                if chunk is None or chunk.document_id in seen_documents:
                    continue
                seen_documents.add(chunk.document_id)
                semantic_score = semantic_scores.get(chunk_id, 0.0)
                lexical_score = lexical_scores.get(chunk_id, 0.0)
                combined = (
                    self._semantic_weight * max(0.0, semantic_score)
                    + self._lexical_weight * lexical_score
                )
                results.append(
                    RetrievedDocument(
                        **chunk.model_dump(exclude={"chunk_index"}),
                        semantic_score=semantic_score,
                        lexical_score=lexical_score,
                        combined_score=min(1.0, combined),
                    )
                )
                if len(results) == request.limit:
                    break
            span.set_attribute("rag.query", request.query)
            span.set_attribute("rag.result_count", len(results))
            span.set_attribute(
                "rag.document_ids",
                json.dumps([result.document_id for result in results]),
            )
            return results


def build_embeddings(settings: Settings) -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        api_key=settings.require_openai_key(),
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dimension,
        http_client=httpx.Client(verify=settings.llm_verify_ssl),
        http_async_client=httpx.AsyncClient(verify=settings.llm_verify_ssl),
    )


async def build_retriever(settings: Settings) -> HybridRetriever:
    documents = await asyncio.to_thread(load_documents, settings.rag_data_dir)
    chunks = await asyncio.to_thread(
        split_documents,
        documents,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    backend: SemanticBackend
    if settings.rag_backend == "pinecone":
        backend = PineconeVectorBackend(settings)
    else:
        backend = LocalVectorBackend(settings.rag_index_path)
    retriever = HybridRetriever(
        build_embeddings(settings),
        backend,
        backend_name=settings.rag_backend,
        embedding_model=settings.openai_embedding_model,
        top_k=settings.rag_top_k,
        candidate_k=settings.rag_candidate_k,
        semantic_weight=settings.rag_semantic_weight,
        lexical_weight=settings.rag_lexical_weight,
    )
    await retriever.initialize(chunks, document_count=len(documents))
    return retriever

from pathlib import Path

from multi_agent_orchestrator.rag.backends import LocalVectorBackend
from multi_agent_orchestrator.rag.chunking import split_documents
from multi_agent_orchestrator.rag.documents import load_documents
from multi_agent_orchestrator.rag.evaluation import (
    GoldenCase,
    evaluate_retrieval,
)
from multi_agent_orchestrator.rag.schemas import RetrievedDocument
from multi_agent_orchestrator.rag.service import HybridRetriever


class KeywordEmbeddings:
    terms = ("checkout", "pagos", "busqueda", "redis", "memoria", "prioridad")

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        clean = text.casefold()
        return [float(clean.count(term)) for term in cls.terms]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vector(text)


async def test_hybrid_retriever_combines_semantic_and_lexical_search(tmp_path: Path) -> None:
    documents = load_documents(Path("data/knowledge"))
    chunks = split_documents(documents, chunk_size=700, chunk_overlap=100)
    retriever = HybridRetriever(
        KeywordEmbeddings(),
        LocalVectorBackend(tmp_path / "index.json"),
        backend_name="local",
        embedding_model="keyword-test",
        top_k=5,
        candidate_k=15,
        semantic_weight=0.6,
        lexical_weight=0.4,
    )
    await retriever.initialize(chunks, document_count=len(documents))

    results = await retriever.retrieve("errores de checkout y proveedor de pagos", limit=5)

    assert results[0].document_id in {
        "runbook-checkout-01",
        "telemetry-checkout-02",
        "postmortem-payments-03",
    }
    assert len({item.document_id for item in results}) == len(results)
    assert retriever.stats.documents == len(documents)
    assert (tmp_path / "index.json").exists()


class EvaluationRetriever:
    async def retrieve(
        self, query: str, *, limit: int | None = None
    ) -> list[RetrievedDocument]:
        del query
        rows = [
            RetrievedDocument(
                chunk_id=f"chunk-{index}",
                document_id=document_id,
                title=document_id,
                content="evidencia",
                source="test.json",
                page=1,
                category="test",
                semantic_score=0.9,
                lexical_score=0.8,
                combined_score=0.86,
            )
            for index, document_id in enumerate(["doc-a", "doc-b", "doc-c"])
        ]
        return rows[: limit or 5]


async def test_retrieval_evaluation_calculates_precision_and_recall() -> None:
    report = await evaluate_retrieval(
        EvaluationRetriever(),
        [GoldenCase(question="consulta valida", relevant_document_ids=["doc-a", "doc-b"])],
        k=5,
    )

    assert report.mean_precision_at_k == 0.4
    assert report.mean_recall_at_k == 1.0

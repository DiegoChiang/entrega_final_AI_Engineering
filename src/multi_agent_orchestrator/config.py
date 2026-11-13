"""Configuracion central del sistema de inteligencia."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variables validadas para API, modelos, RAG, persistencia y trazas."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    anthropic_model: str = "claude-sonnet-4-5"
    llm_temperature: float = Field(default=0, ge=0, le=2)
    llm_timeout_seconds: float = Field(default=45, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_verify_ssl: bool = True

    rag_backend: Literal["local", "pinecone"] = "local"
    rag_data_dir: Path = Path("data/knowledge")
    rag_index_path: Path = Path("state/rag-index.json")
    rag_top_k: int = Field(default=5, ge=2, le=10)
    rag_candidate_k: int = Field(default=15, ge=5, le=50)
    rag_chunk_size: int = Field(default=700, ge=300, le=2_000)
    rag_chunk_overlap: int = Field(default=100, ge=0, le=500)
    rag_semantic_weight: float = Field(default=0.6, ge=0, le=1)
    rag_lexical_weight: float = Field(default=0.4, ge=0, le=1)
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1)
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = Field(default="incident-intelligence", min_length=1)
    pinecone_namespace: str = Field(default="technical-knowledge", min_length=1)
    pinecone_cloud: str = Field(default="aws", min_length=1)
    pinecone_region: str = Field(default="us-east-1", min_length=1)

    supervisor_max_steps: int = Field(default=8, ge=4, le=20)
    recursion_limit: int = Field(default=30, ge=10, le=100)
    redis_url: str = "redis://localhost:6379"
    job_ttl_seconds: int = Field(default=86_400, ge=300)
    job_queue_size: int = Field(default=100, ge=5, le=10_000)
    worker_count: int = Field(default=2, ge=1, le=32)
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "incident-intelligence-system"
    phoenix_enabled: bool = True

    @field_validator("rag_chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: object) -> int:
        chunk_size = getattr(info, "data", {}).get("rag_chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP debe ser menor que RAG_CHUNK_SIZE")
        return value

    @model_validator(mode="after")
    def validate_retrieval(self) -> Settings:
        if abs((self.rag_semantic_weight + self.rag_lexical_weight) - 1.0) > 1e-9:
            raise ValueError("Los pesos semantico y lexico deben sumar 1")
        if self.rag_candidate_k < self.rag_top_k:
            raise ValueError("RAG_CANDIDATE_K debe ser mayor o igual que RAG_TOP_K")
        return self

    @staticmethod
    def _required_secret(value: SecretStr | None, variable: str) -> str:
        if value is None or not value.get_secret_value().strip():
            raise ValueError(f"Falta la variable de entorno {variable}")
        return value.get_secret_value()

    def require_openai_key(self) -> str:
        return self._required_secret(self.openai_api_key, "OPENAI_API_KEY")

    def require_anthropic_key(self) -> str:
        return self._required_secret(self.anthropic_api_key, "ANTHROPIC_API_KEY")

    def require_pinecone_key(self) -> str:
        return self._required_secret(self.pinecone_api_key, "PINECONE_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

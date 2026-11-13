"""Contratos Pydantic de la API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.REJECTED, JobStatus.FAILED}


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=10, max_length=4_000)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    feedback: str = Field(default="", max_length=1_000)


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    title: str
    source: str
    page: int = Field(ge=1)
    combined_score: float = Field(ge=0, le=1)


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: str
    risk_score: float = Field(ge=0, le=100)
    drivers: list[str]
    recommendation: str
    source_ids: list[str]


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str
    analysis_result: AnalysisOutput | None = None
    approval_status: str
    completed_agents: list[str]
    sources: list[SourceReference] = Field(default_factory=list)


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    request: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    approval_request: dict[str, Any] | None = None
    result: JobResult | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    redis: str
    queue_depth: int
    rag_backend: str
    rag_documents: int = Field(ge=0)
    rag_chunks: int = Field(ge=0)

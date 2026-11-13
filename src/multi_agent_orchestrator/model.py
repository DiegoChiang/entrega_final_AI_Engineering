"""Construccion intercambiable del modelo de lenguaje."""

from __future__ import annotations

from typing import Any

import httpx

from multi_agent_orchestrator.config import Settings


def build_model(settings: Settings) -> Any:
    """Construye un chat model asíncrono a partir de la configuración validada."""

    if settings.llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depende de la instalacion
            raise RuntimeError("Instala la dependencia 'langchain-anthropic'") from exc
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.require_anthropic_key(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.require_openai_key(),
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        http_client=httpx.Client(verify=settings.llm_verify_ssl),
        http_async_client=httpx.AsyncClient(verify=settings.llm_verify_ssl),
    )

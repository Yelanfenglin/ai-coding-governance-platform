"""Application configuration with isolation and provider safety checks."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class Settings(BaseSettings):
    """RAG service settings loaded from ``RAG_`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: SecretStr

    milvus_uri: str
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "rag_specs"
    milvus_batch_size: int = 128
    milvus_search_ef: int = 64

    embedding_provider: Literal["dashscope", "deterministic", "hf"] = "dashscope"
    embedding_model: str = "text-embedding-v3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 16
    embedding_fallback: Literal["none"] = "none"

    dashscope_api_key: SecretStr = SecretStr("")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    hf_token: SecretStr = SecretStr("")
    hf_timeout_seconds: float = 30.0
    hf_max_retries: int = 3

    reranker_enabled: bool = False
    reranker_model: str = "gte-rerank-v2"
    llm_enabled: bool = False
    llm_provider: Literal["dashscope", "deepseek"] = "dashscope"
    llm_model: str = "qwen-plus"
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_thinking: Literal["enabled", "disabled"] = "enabled"
    deepseek_reasoning_effort: Literal["low", "high", "max"] = "high"
    planner_enabled: bool = False
    planner_model: str = ""
    generation_failure_mode: Literal["error", "retrieval-only"] = "error"

    paddle_cache: Path = Path("artifacts/paddlex")
    paddle_layout_model: str = "PP-DocLayoutV3"
    paddle_ocr_tier: str = "small"
    ocr_mode: Literal["off", "auto", "required"] = "off"
    layout_mode: Literal["off", "auto", "required"] = "required"
    render_dpi: int = 168

    storage_dir: Path = Path("storage")
    api_prefix: str = "/api/rag"
    api_key: SecretStr = SecretStr("")
    upload_max_bytes: int = 104_857_600
    query_top_k_default: int = 8
    query_top_k_max: int = 50
    context_top_n: int = 4
    context_neighbor_window: int = 0
    bm25_enabled: bool = False
    bm25_top_n: int = 30
    bm25_index_ttl_seconds: float = 600.0
    bm25_cache_max_scopes: int = 32
    fusion_dense_slots: int = 3
    fusion_sparse_slots: int = 2
    rrf_k0: float = 60.0
    multi_source_routing_enabled: bool = True
    multi_source_max_products: int = 3
    multi_query_enabled: bool = True
    multi_query_max_queries: int = 3
    ingest_concurrency: int = 1
    retain_superseded_vectors: bool = True
    query_trace_enabled: bool = False
    query_trace_path: Path | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_isolation_and_modes(self) -> Self:
        try:
            database_url = make_url(self.database_url.get_secret_value())
        except (ArgumentError, TypeError, ValueError):
            raise ValueError("RAG_DATABASE_URL is not a valid SQLAlchemy URL") from None

        if database_url.drivername != "postgresql+psycopg":
            raise ValueError("RAG_DATABASE_URL must use the postgresql+psycopg driver")
        if database_url.database != "rag_kb":
            raise ValueError("RAG_DATABASE_URL must target the isolated rag_kb database")

        if self.milvus_collection not in {"rag_specs", "rag_specs_v2"}:
            raise ValueError(
                "RAG_MILVUS_COLLECTION must be rag_specs or rag_specs_v2"
            )
        self._validate_http_url("RAG_MILVUS_URI", self.milvus_uri)
        self._validate_http_url("RAG_DASHSCOPE_BASE_URL", self.dashscope_base_url)

        if self.embedding_dimension != 1024:
            raise ValueError("RAG_EMBEDDING_DIMENSION must be 1024")
        if not self.embedding_model.strip():
            raise ValueError("RAG_EMBEDDING_MODEL must not be empty")
        if self.embedding_provider == "dashscope" and not self._has_secret(self.dashscope_api_key):
            raise ValueError(
                "RAG_DASHSCOPE_API_KEY is required when the embedding provider is dashscope"
            )
        if self.embedding_provider == "hf" and not self._has_secret(self.hf_token):
            raise ValueError("RAG_HF_TOKEN is required when the embedding provider is hf")
        if self.reranker_enabled and not self._has_secret(self.dashscope_api_key):
            raise ValueError(
                "RAG_DASHSCOPE_API_KEY is required when reranking is enabled"
            )
        if self.llm_enabled or self.planner_enabled:
            if self.llm_provider == "deepseek" and not self._has_secret(
                self.deepseek_api_key
            ):
                raise ValueError(
                    "RAG_DEEPSEEK_API_KEY is required when llm_provider=deepseek"
                )
            if self.llm_provider == "dashscope" and not self._has_secret(
                self.dashscope_api_key
            ):
                raise ValueError(
                    "RAG_DASHSCOPE_API_KEY is required when llm_provider=dashscope"
                )
        if self.reranker_enabled and not self.reranker_model.strip():
            raise ValueError("RAG_RERANKER_MODEL must not be empty when reranking is enabled")
        if self.llm_enabled and not self.llm_model.strip():
            raise ValueError("RAG_LLM_MODEL must not be empty when generation is enabled")
        if self.planner_enabled and not (self.planner_model or self.llm_model).strip():
            raise ValueError(
                "RAG_PLANNER_MODEL/RAG_LLM_MODEL must not be empty when planning is enabled"
            )

        positive_values = {
            "RAG_MILVUS_BATCH_SIZE": self.milvus_batch_size,
            "RAG_MILVUS_SEARCH_EF": self.milvus_search_ef,
            "RAG_EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
            "RAG_HF_TIMEOUT_SECONDS": self.hf_timeout_seconds,
            "RAG_UPLOAD_MAX_BYTES": self.upload_max_bytes,
            "RAG_QUERY_TOP_K_DEFAULT": self.query_top_k_default,
            "RAG_QUERY_TOP_K_MAX": self.query_top_k_max,
            "RAG_CONTEXT_TOP_N": self.context_top_n,
            "RAG_BM25_TOP_N": self.bm25_top_n,
            "RAG_BM25_INDEX_TTL_SECONDS": self.bm25_index_ttl_seconds,
            "RAG_BM25_CACHE_MAX_SCOPES": self.bm25_cache_max_scopes,
            "RAG_RRF_K0": self.rrf_k0,
            "RAG_MULTI_SOURCE_MAX_PRODUCTS": self.multi_source_max_products,
            "RAG_MULTI_QUERY_MAX_QUERIES": self.multi_query_max_queries,
            "RAG_INGEST_CONCURRENCY": self.ingest_concurrency,
        }
        invalid_positive = [name for name, value in positive_values.items() if value <= 0]
        if invalid_positive:
            raise ValueError(f"{', '.join(invalid_positive)} must be greater than zero")
        if not math.isfinite(self.hf_timeout_seconds):
            raise ValueError("RAG_HF_TIMEOUT_SECONDS must be finite")
        if self.hf_max_retries < 0:
            raise ValueError("RAG_HF_MAX_RETRIES must not be negative")
        if self.query_top_k_default > self.query_top_k_max:
            raise ValueError("RAG_QUERY_TOP_K_DEFAULT must not exceed RAG_QUERY_TOP_K_MAX")
        if self.context_top_n > self.query_top_k_max:
            raise ValueError("RAG_CONTEXT_TOP_N must not exceed RAG_QUERY_TOP_K_MAX")
        if not 0 <= self.context_neighbor_window <= 2:
            raise ValueError("RAG_CONTEXT_NEIGHBOR_WINDOW must be between 0 and 2")
        if self.fusion_dense_slots < 0 or self.fusion_sparse_slots < 0:
            raise ValueError("RAG_FUSION_*_SLOTS must not be negative")
        if self.fusion_dense_slots + self.fusion_sparse_slots < 1:
            raise ValueError("at least one RAG_FUSION_*_SLOTS must be positive")
        if self.multi_source_max_products < 2:
            raise ValueError("RAG_MULTI_SOURCE_MAX_PRODUCTS must be at least 2")
        if self.multi_query_max_queries < 1:
            raise ValueError("RAG_MULTI_QUERY_MAX_QUERIES must be at least 1")
        if not 72 <= self.render_dpi <= 600:
            raise ValueError("RAG_RENDER_DPI must be between 72 and 600")
        if not self.api_prefix.startswith("/"):
            raise ValueError("RAG_API_PREFIX must start with '/'")

        self.storage_dir = self._resolve_path(self.storage_dir)
        self.paddle_cache = self._resolve_path(self.paddle_cache)
        return self

    @staticmethod
    def _has_secret(value: SecretStr) -> bool:
        return bool(value.get_secret_value().strip())

    @staticmethod
    def _resolve_path(value: Path) -> Path:
        return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()

    @staticmethod
    def _validate_http_url(name: str, value: str) -> None:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{name} must be an absolute HTTP(S) URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings without creating external connections."""

    return Settings()  # type: ignore[call-arg]

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.settings import Settings, get_settings


def _settings_data(tmp_path: Path) -> dict[str, object]:
    return {
        "database_url": "postgresql+psycopg://rag_user:db-password@127.0.0.1/rag_kb",
        "milvus_uri": "http://127.0.0.1:19531",
        "embedding_provider": "deterministic",
        "embedding_model": "deterministic-hash-v1",
        "storage_dir": tmp_path / "storage",
        "paddle_cache": tmp_path / "paddle-cache",
    }


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    data = _settings_data(tmp_path)
    data.update(overrides)
    return Settings(_env_file=None, **data)


def test_settings_resolve_paths_and_keep_isolated_defaults(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert settings.storage_dir == (tmp_path / "storage").resolve()
    assert settings.paddle_cache == (tmp_path / "paddle-cache").resolve()
    assert settings.milvus_collection == "rag_specs"
    assert settings.embedding_dimension == 1024
    assert settings.embedding_fallback == "none"


def test_relative_paths_are_resolved_from_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, storage_dir="relative-storage", paddle_cache="relative-cache")

    assert settings.storage_dir == (tmp_path / "relative-storage").resolve()
    assert settings.paddle_cache == (tmp_path / "relative-cache").resolve()


def test_secret_values_do_not_appear_in_repr(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        api_key="shared-api-secret",
        dashscope_api_key="dashscope-secret",
        hf_token="hf-secret",
        milvus_token="milvus-secret",
    )

    rendered = repr(settings)
    for secret in (
        "db-password",
        "shared-api-secret",
        "dashscope-secret",
        "hf-secret",
        "milvus-secret",
    ):
        assert secret not in rendered
    assert "**********" in rendered


def test_database_validation_error_does_not_expose_password(tmp_path: Path) -> None:
    password = "validation-password-must-not-leak"
    with pytest.raises(ValidationError) as captured:
        _settings(
            tmp_path,
            database_url=(
                f"postgresql+psycopg://rag_user:{password}@127.0.0.1/not_rag_kb"
            ),
        )

    assert password not in str(captured.value)
    assert password not in repr(captured.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "database_url",
            "postgresql+psycopg://rag_user:secret@127.0.0.1/not_rag_kb",
        ),
        (
            "database_url",
            "postgresql+asyncpg://rag_user:secret@127.0.0.1/rag_kb",
        ),
        ("milvus_collection", "stockpilot_quality_policies"),
        ("embedding_dimension", 768),
        ("embedding_provider", "unknown"),
    ],
)
def test_isolation_and_embedding_invariants_reject_invalid_values(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _settings(tmp_path, **{field: value})


def test_v2_collection_name_is_allowed(tmp_path: Path) -> None:
    settings = _settings(tmp_path, milvus_collection="rag_specs_v2")
    assert settings.milvus_collection == "rag_specs_v2"


def test_selected_remote_embedding_provider_requires_its_credential(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="RAG_DASHSCOPE_API_KEY"):
        _settings(
            tmp_path,
            embedding_provider="dashscope",
            embedding_model="text-embedding-v3",
            dashscope_api_key="",
        )

    with pytest.raises(ValidationError, match="RAG_HF_TOKEN"):
        _settings(
            tmp_path,
            embedding_provider="hf",
            embedding_model="BAAI/bge-m3",
            hf_token="",
        )


def test_top_k_and_remote_feature_combinations_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="RAG_QUERY_TOP_K_DEFAULT"):
        _settings(tmp_path, query_top_k_default=51, query_top_k_max=50)

    with pytest.raises(ValidationError, match="RAG_DASHSCOPE_API_KEY"):
        _settings(tmp_path, reranker_enabled=True)

    with pytest.raises(ValidationError, match="RAG_LLM_MODEL"):
        _settings(
            tmp_path,
            llm_enabled=True,
            llm_model=" ",
            dashscope_api_key="configured-for-validation",
        )


def test_get_settings_uses_rag_environment_and_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in tuple(os.environ):
        if name.startswith("RAG_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "RAG_DATABASE_URL",
        "postgresql+psycopg://rag_user:environment-secret@127.0.0.1/rag_kb",
    )
    monkeypatch.setenv("RAG_MILVUS_URI", "http://127.0.0.1:19531")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "deterministic-hash-v1")
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
    finally:
        get_settings.cache_clear()

    assert first is second
    assert first.embedding_provider == "deterministic"
    assert "environment-secret" not in repr(first)

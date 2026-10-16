"""Settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = field(default_factory=lambda: Path(_env("CORPUS_DIR", "corpus")))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "index")))
    traces_path: Path = field(default_factory=lambda: Path(_env("TRACES_PATH", "traces.db")))
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "hashing"))
    generator: str = field(default_factory=lambda: _env("GENERATOR", "extractive"))
    store: str = field(default_factory=lambda: _env("STORE", "memory"))
    postgres_dsn: str = field(
        default_factory=lambda: _env("POSTGRES_DSN", "postgresql://rag:rag@localhost:5433/rag")
    )
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3.1:8b"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "not-needed"))
    price_input_per_m: float = field(default_factory=lambda: float(_env("PRICE_INPUT_PER_M", "0")))
    price_output_per_m: float = field(
        default_factory=lambda: float(_env("PRICE_OUTPUT_PER_M", "0"))
    )
    chunk_words: int = 120
    chunk_overlap: int = 30
    top_k: int = 4
    # Below this relevance the assistant refuses instead of guessing. Chosen with
    # `course-rag sweep` on the eval set (max-margin point of the best balanced accuracy);
    # re-run the sweep whenever the embedder, chunking or corpus changes.
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.238")))


def get_settings() -> Settings:
    return Settings()

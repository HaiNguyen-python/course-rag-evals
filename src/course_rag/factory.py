"""Wire the components together from settings."""

from __future__ import annotations

from .answer import Assistant, ExtractiveGenerator, OpenAICompatibleGenerator
from .config import Settings
from .embeddings import make_embedder
from .retrieval import Retriever
from .store import MemoryStore, PgVectorStore


def make_store(settings: Settings, dim: int):
    if settings.store == "postgres":
        return PgVectorStore(settings.postgres_dsn, dim)
    return MemoryStore(settings.index_dir)


def make_generator(settings: Settings):
    if settings.generator == "openai":
        return OpenAICompatibleGenerator(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model
        )
    return ExtractiveGenerator()


def make_assistant(settings: Settings, mode: str = "hybrid", store=None, embedder=None):
    embedder = embedder or make_embedder(
        settings.embedder, base_url=settings.llm_base_url, api_key=settings.llm_api_key
    )
    store = store or make_store(settings, embedder.dim)
    retriever = Retriever(store, embedder, mode=mode)
    return Assistant(
        retriever,
        make_generator(settings),
        settings.min_relevance,
        settings.top_k,
        settings.price_input_per_m,
        settings.price_output_per_m,
    )

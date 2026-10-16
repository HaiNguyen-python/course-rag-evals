"""Embedding backends behind one small interface.

- ``hashing``: offline, deterministic, no model download. Word + character n-grams hashed
  into a fixed vector. Good enough for CI and a baseline; weak on synonyms.
- ``sentence-transformers``: a local neural model (pip install ".[neural]").
- ``openai``: any OpenAI-compatible /embeddings endpoint.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._words = HashingVectorizer(
            n_features=dim,
            ngram_range=(1, 2),
            stop_words="english",
            alternate_sign=False,
            norm=None,
        )
        self._chars = HashingVectorizer(
            n_features=dim,
            analyzer="char_wb",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm=None,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        words = self._words.transform(texts).toarray()
        chars = self._chars.transform(texts).toarray()
        return _normalise(np.log1p(words) * 2.0 + np.log1p(chars) * 0.5).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.name = model
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OpenAIEmbedder:
    def __init__(
        self, base_url: str, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536
    ):
        import httpx

        self._client = httpx.Client(
            base_url=base_url, timeout=60, headers={"Authorization": f"Bearer {api_key}"}
        )
        self.name, self.dim, self._model = model, dim, model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.post("/embeddings", json={"model": self._model, "input": texts})
        resp.raise_for_status()
        vectors = [d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"])]
        return _normalise(np.asarray(vectors, dtype=np.float32))


def make_embedder(kind: str, **kwargs) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if kind == "openai":
        return OpenAIEmbedder(kwargs["base_url"], kwargs["api_key"])
    raise ValueError(f"unknown embedder {kind!r}")

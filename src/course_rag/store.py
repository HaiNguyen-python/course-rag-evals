"""Vector stores: in-memory (persisted to files) and Postgres + pgvector.

Both keep a table of document hashes so indexing can be incremental:
unchanged documents are skipped, changed ones re-embedded, removed ones deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .documents import Chunk


class VectorStore(Protocol):
    def doc_hashes(self) -> dict[str, str]: ...
    def upsert_document(
        self, doc_id: str, content_hash: str, chunks: list[Chunk], vectors: np.ndarray
    ) -> None: ...
    def delete_document(self, doc_id: str) -> None: ...
    def chunks(self) -> list[Chunk]: ...
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def save(self) -> None: ...


class MemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._hashes: dict[str, str] = {}
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        if path and (path / "chunks.json").exists():
            self._load()

    def doc_hashes(self) -> dict[str, str]:
        return dict(self._hashes)

    def upsert_document(self, doc_id, content_hash, chunks, vectors) -> None:
        self.delete_document(doc_id)
        self._hashes[doc_id] = content_hash
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vec

    def delete_document(self, doc_id: str) -> None:
        self._hashes.pop(doc_id, None)
        for cid in [c for c, ch in self._chunks.items() if ch.doc_id == doc_id]:
            del self._chunks[cid]
            del self._vectors[cid]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def search(self, query_vector, k):
        if not self._chunks:
            return []
        ids = list(self._vectors)
        matrix = np.stack([self._vectors[i] for i in ids])
        sims = matrix @ query_vector
        top = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in top]

    def save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        ids = list(self._chunks)
        (self.path / "chunks.json").write_text(
            json.dumps({"hashes": self._hashes, "chunks": [vars(self._chunks[i]) for i in ids]})
        )
        vectors = np.stack([self._vectors[i] for i in ids]) if ids else np.zeros((0, 1))
        np.save(self.path / "vectors.npy", vectors)

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads((self.path / "chunks.json").read_text())
        vectors = np.load(self.path / "vectors.npy")
        self._hashes = data["hashes"]
        for raw, vec in zip(data["chunks"], vectors, strict=True):
            chunk = Chunk(**raw)
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vec


class PgVectorStore:
    """Postgres + pgvector. Cosine distance with an HNSW index."""

    def __init__(self, dsn: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_documents (doc_id text PRIMARY KEY, "
            "content_hash text NOT NULL, indexed_at timestamptz DEFAULT now())"
        )
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS rag_chunks (chunk_id text PRIMARY KEY, "
            f"doc_id text NOT NULL REFERENCES rag_documents ON DELETE CASCADE, "
            f"title text, section text, text text NOT NULL, embedding vector({dim}) NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )

    def doc_hashes(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT doc_id, content_hash FROM rag_documents").fetchall())

    def upsert_document(self, doc_id, content_hash, chunks, vectors) -> None:
        with self.conn.transaction():
            self.conn.execute("DELETE FROM rag_documents WHERE doc_id = %s", (doc_id,))
            self.conn.execute(
                "INSERT INTO rag_documents (doc_id, content_hash) VALUES (%s, %s)",
                (doc_id, content_hash),
            )
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO rag_chunks VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (c.chunk_id, c.doc_id, c.title, c.section, c.text, v)
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )

    def delete_document(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM rag_documents WHERE doc_id = %s", (doc_id,))

    def chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, title, section, text FROM rag_chunks"
        ).fetchall()
        return [Chunk(*r) for r in rows]

    def search(self, query_vector, k):
        rows = self.conn.execute(
            "SELECT chunk_id, 1 - (embedding <=> %s) AS similarity FROM rag_chunks "
            "ORDER BY embedding <=> %s LIMIT %s",
            (query_vector, query_vector, k),
        ).fetchall()
        return [(cid, float(sim)) for cid, sim in rows]

    def save(self) -> None:  # writes are already committed
        pass

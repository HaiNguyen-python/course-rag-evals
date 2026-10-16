"""Incremental indexing: only new or changed documents are embedded again."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .documents import chunk_document, load_documents
from .embeddings import Embedder
from .store import VectorStore

log = logging.getLogger("course_rag.ingest")


@dataclass
class IndexStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_embedded: int = 0


def index_corpus(
    corpus_dir: Path,
    store: VectorStore,
    embedder: Embedder,
    chunk_words: int = 120,
    overlap: int = 30,
) -> IndexStats:
    stats = IndexStats()
    known = store.doc_hashes()
    docs = load_documents(corpus_dir)
    for doc in docs:
        if known.get(doc.doc_id) == doc.content_hash:
            stats.unchanged.append(doc.doc_id)
            continue
        chunks = chunk_document(doc, chunk_words, overlap)
        vectors = embedder.embed([f"{c.title}. {c.section}. {c.text}" for c in chunks])
        store.upsert_document(doc.doc_id, doc.content_hash, chunks, vectors)
        (stats.updated if doc.doc_id in known else stats.added).append(doc.doc_id)
        stats.chunks_embedded += len(chunks)
    for doc_id in set(known) - {d.doc_id for d in docs}:
        store.delete_document(doc_id)
        stats.removed.append(doc_id)
    store.save()
    log.info("index: %s", stats)
    return stats

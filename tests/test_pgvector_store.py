"""Runs when PGVECTOR_TEST_DSN points at Postgres with the pgvector extension
(CI uses the pgvector/pgvector:pg16 image)."""

import os

import pytest

DSN = os.getenv("PGVECTOR_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PGVECTOR_TEST_DSN not set")


def test_pgvector_store_matches_memory_store(corpus, embedder):
    import psycopg

    from course_rag.ingest import index_corpus
    from course_rag.store import MemoryStore, PgVectorStore

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS rag_chunks, rag_documents")
    pg = PgVectorStore(DSN, embedder.dim)
    mem = MemoryStore()
    index_corpus(corpus, pg, embedder)
    index_corpus(corpus, mem, embedder)
    assert pg.doc_hashes() == mem.doc_hashes()

    q = embedder.embed(["What is a slowly changing dimension?"])[0]
    assert pg.search(q, 3)[0][0] == mem.search(q, 3)[0][0]

    # Re-indexing an unchanged corpus is a no-op; deleting cascades to chunks.
    assert index_corpus(corpus, pg, embedder).chunks_embedded == 0
    pg.delete_document("05-data-modelling")
    assert all(c.doc_id != "05-data-modelling" for c in pg.chunks())

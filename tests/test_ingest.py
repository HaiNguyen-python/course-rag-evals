from course_rag.ingest import index_corpus
from course_rag.store import MemoryStore


def test_incremental_indexing(corpus, embedder, tmp_path):
    store = MemoryStore(tmp_path / "index")
    first = index_corpus(corpus, store, embedder)
    assert len(first.added) == 6 and first.chunks_embedded > 0

    # Nothing changed: nothing is embedded again.
    again = index_corpus(corpus, MemoryStore(tmp_path / "index"), embedder)
    assert len(again.unchanged) == 6 and again.chunks_embedded == 0

    # Edit one file, delete another, add a new one.
    (corpus / "03-data-quality.md").write_text("# Data Quality\n\nNew text about freshness checks.")
    (corpus / "06-mlops-basics.md").unlink()
    (corpus / "07-new.md").write_text("# New Topic\n\nSomething new about lakehouses.")
    store = MemoryStore(tmp_path / "index")
    stats = index_corpus(corpus, store, embedder)
    assert stats.updated == ["03-data-quality"]
    assert stats.removed == ["06-mlops-basics"]
    assert stats.added == ["07-new"]
    assert not any(c.doc_id == "06-mlops-basics" for c in store.chunks())
    assert any("freshness" in c.text for c in store.chunks())

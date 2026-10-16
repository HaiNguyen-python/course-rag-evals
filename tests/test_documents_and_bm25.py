import pytest

from course_rag.bm25 import BM25, tokenize
from course_rag.documents import Document, chunk_document, content_hash, load_documents
from course_rag.retrieval import rrf, term_coverage


def test_load_documents_reads_titles(corpus):
    docs = load_documents(corpus)
    assert len(docs) == 6
    assert docs[0].title == "Medallion Architecture"


def test_chunks_respect_sections_and_overlap():
    body = " ".join(f"w{i}" for i in range(250))
    doc = Document("d", "T", f"# T\n\n## A\n\n{body}\n\n## B\n\nshort text", content_hash(body))
    chunks = chunk_document(doc, words=100, overlap=20)
    assert [c.section for c in chunks].count("B") == 1
    a_chunks = [c for c in chunks if c.section == "A"]
    assert len(a_chunks) == 3
    # Overlap: the last 20 words of one chunk start the next one.
    assert a_chunks[0].text.split()[-20:] == a_chunks[1].text.split()[:20]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "T", "text", "h"), words=10, overlap=10)


def test_bm25_ranks_the_matching_document_first():
    index = BM25().fit(["a", "b"], ["kafka streaming events", "star schema dimension tables"])
    scores = index.scores("what is a star schema?")
    assert max(scores, key=scores.get) == "b"
    assert "a" not in scores


def test_tokenize_drops_stopwords():
    assert tokenize("What is the Bronze layer?") == ["bronze", "layer"]


def test_rrf_rewards_agreement():
    fused = rrf([["x", "y", "z"], ["y", "x", "w"]])
    assert set(sorted(fused, key=fused.get, reverse=True)[:2]) == {"x", "y"}
    assert fused["w"] < fused["y"]


def test_term_coverage():
    assert term_coverage("bronze layer", "the bronze layer stores raw data") == 1.0
    assert term_coverage("capital of australia", "bronze layer") == 0.0

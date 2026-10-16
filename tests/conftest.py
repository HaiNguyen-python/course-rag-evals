import shutil
from pathlib import Path

import pytest

from course_rag.answer import Assistant, ExtractiveGenerator
from course_rag.embeddings import HashingEmbedder
from course_rag.ingest import index_corpus
from course_rag.retrieval import Retriever
from course_rag.store import MemoryStore

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture()
def corpus(tmp_path):
    target = tmp_path / "corpus"
    shutil.copytree(CORPUS, target)
    return target


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder()


@pytest.fixture()
def assistant(corpus, embedder):
    store = MemoryStore()
    index_corpus(corpus, store, embedder)
    return Assistant(Retriever(store, embedder), ExtractiveGenerator(), min_relevance=0.238)

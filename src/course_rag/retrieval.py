"""Hybrid retrieval: vector search + BM25, fused with Reciprocal Rank Fusion (RRF).

RRF only uses ranks, so it can combine scores on different scales without tuning weights:
    score(chunk) = sum over retrievers of 1 / (k + rank)
"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25, tokenize
from .documents import Chunk
from .embeddings import Embedder
from .store import VectorStore

RRF_K = 60
MODES = ("vector", "bm25", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    hits: list[Hit]
    relevance: float  # 0..1, used to decide whether to refuse


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def term_coverage(query: str, text: str) -> float:
    """Share of the query's content words that appear in the text."""
    q = set(tokenize(query))
    return len(q & set(tokenize(text))) / len(q) if q else 0.0


class Retriever:
    def __init__(
        self, store: VectorStore, embedder: Embedder, mode: str = "hybrid", candidates: int = 20
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.store, self.embedder, self.mode, self.candidates = store, embedder, mode, candidates
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the keyword index after the store changes."""
        self._chunks = {c.chunk_id: c for c in self.store.chunks()}
        ids = list(self._chunks)
        self._bm25 = BM25().fit(
            ids, [self._chunks[i].section + " " + self._chunks[i].text for i in ids]
        )

    def retrieve(self, query: str, k: int = 4) -> RetrievalResult:
        rankings: list[list[str]] = []
        vector_sim: dict[str, float] = {}
        if self.mode in ("vector", "hybrid"):
            qv = self.embedder.embed([query])[0]
            vec_hits = self.store.search(qv, self.candidates)
            vector_sim = dict(vec_hits)
            rankings.append([cid for cid, _ in vec_hits])
        if self.mode in ("bm25", "hybrid"):
            bm = self._bm25.scores(query)
            rankings.append(sorted(bm, key=lambda c: bm[c], reverse=True)[: self.candidates])

        fused = rrf(rankings)
        top = sorted(fused, key=lambda c: fused[c], reverse=True)[:k]
        hits = [Hit(self._chunks[cid], fused[cid]) for cid in top if cid in self._chunks]
        if not hits:
            return RetrievalResult([], 0.0)
        best = hits[0].chunk
        coverage = term_coverage(query, best.section + " " + best.text)
        cosine = max(vector_sim.get(h.chunk.chunk_id, 0.0) for h in hits) if vector_sim else 0.0
        relevance = coverage if self.mode == "bm25" else 0.5 * coverage + 0.5 * max(cosine, 0.0)
        return RetrievalResult(hits, round(relevance, 4))

"""Answer generation with citations, refusal and cost tracking.

Two generators share one interface:
- ``ExtractiveGenerator`` (offline, default): picks the most relevant sentences from the
  retrieved chunks. Deterministic, so it is used in CI.
- ``OpenAICompatibleGenerator``: any /chat/completions API (OpenAI, Ollama, vLLM, ...).
Both must cite chunk ids like [03-data-quality#2]; the assistant refuses when retrieval
relevance is below a threshold, instead of guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .bm25 import tokenize
from .retrieval import Hit, Retriever

REFUSAL = "I can't find this in the course material."
CITATION_RE = re.compile(r"\[([\w.-]+#\d+)\]")

SYSTEM_PROMPT = (
    "You are a teaching assistant. Answer ONLY from the context below. "
    "After each sentence, cite the chunk id in square brackets, e.g. [01-intro#0]. "
    f"If the context does not contain the answer, reply exactly: {REFUSAL}"
)


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    refused: bool
    retrieved: list[tuple[str, float]]
    relevance: float
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Rough estimate (about 0.75 words per token) when the API reports no usage."""
    return max(1, round(len(text.split()) / 0.75))


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.chunk_id}] ({h.chunk.title} > {h.chunk.section})\n{h.chunk.text}" for h in hits
    )


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> tuple[str, int, int]:
        q = set(tokenize(question))
        scored = []
        for rank, h in enumerate(hits):
            for pos, sent in enumerate(re.split(r"(?<=[.!?])\s+", h.chunk.text)):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    # Prefer overlap, then higher-ranked chunks, then earlier sentences.
                    scored.append((overlap, -rank, -pos, sent.strip(), h.chunk.chunk_id))
        if not scored:
            return REFUSAL, 0, 0
        best = sorted(scored, reverse=True)[: self.max_sentences]
        text = " ".join(f"{s} [{cid}]" for *_, s, cid in best)
        return text, 0, 0


class OpenAICompatibleGenerator:
    name = "openai"

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout: float = 60, transport=None
    ):
        import httpx

        self.model = model
        # `transport` lets tests plug in httpx.MockTransport instead of a real server.
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def generate(self, question: str, hits: list[Hit]) -> tuple[str, int, int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
        ]
        resp = self._client.post(
            "/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": 0},
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        prompt = " ".join(m["content"] for m in messages)
        return (
            text,
            usage.get("prompt_tokens", estimate_tokens(prompt)),
            usage.get("completion_tokens", estimate_tokens(text)),
        )


class Assistant:
    def __init__(
        self,
        retriever: Retriever,
        generator,
        min_relevance: float = 0.2,
        top_k: int = 4,
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
    ):
        self.retriever, self.generator = retriever, generator
        self.min_relevance, self.top_k = min_relevance, top_k
        self.price_in, self.price_out = price_in_per_m, price_out_per_m

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        result = self.retriever.retrieve(question, self.top_k)
        retrieved = [(h.chunk.chunk_id, round(h.score, 5)) for h in result.hits]
        if result.relevance < self.min_relevance:
            text, tin, tout = REFUSAL, 0, 0
        else:
            text, tin, tout = self.generator.generate(question, result.hits)
        retrieved_ids = {cid for cid, _ in retrieved}
        citations = [c for c in dict.fromkeys(CITATION_RE.findall(text))]
        return Answer(
            question=question,
            text=text,
            citations=citations,
            refused=REFUSAL.lower().rstrip(".") in text.lower(),
            retrieved=retrieved,
            relevance=result.relevance,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=(tin * self.price_in + tout * self.price_out) / 1_000_000,
            extra={"invalid_citations": [c for c in citations if c not in retrieved_ids]},
        )

"""HTTP API: POST /ask returns an answer with citations, and logs a trace."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .factory import make_assistant
from .tracing import Tracer

app = FastAPI(title="Course RAG assistant", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    refused: bool
    sources: list[Source]
    latency_ms: float
    cost_usd: float


@lru_cache(maxsize=1)
def get_assistant():
    return make_assistant(get_settings())


@lru_cache(maxsize=1)
def get_tracer():
    return Tracer(get_settings().traces_path)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, assistant=Depends(get_assistant), tracer=Depends(get_tracer)):
    a = assistant.ask(req.question)
    tracer.log(a, assistant.generator.name)
    return AskResponse(
        answer=a.text,
        citations=a.citations,
        refused=a.refused,
        sources=[Source(chunk_id=c, score=s) for c, s in a.retrieved],
        latency_ms=a.latency_ms,
        cost_usd=a.cost_usd,
    )

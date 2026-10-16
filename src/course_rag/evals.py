"""Evaluation harness.

Metrics (per setup):
- hit_rate@k   share of answerable questions where an expected document is in the top k
- mrr          mean reciprocal rank of the first relevant chunk
- answer_rate  share of answerable questions that were answered (not refused)
- refusal_rate share of unanswerable questions that were correctly refused
- citation_ok  share of answers whose citations all point to retrieved chunks
- grounded     share of answers citing at least one chunk from an expected document
- faithfulness share of answer content words found in the cited chunks (a cheap proxy)
- latency p50 / p95 and cost per 1,000 questions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .answer import CITATION_RE, Answer, Assistant
from .bm25 import tokenize


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_docs: list[str]
    answerable: bool = True
    kind: str = "direct"  # direct | paraphrase | unanswerable


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text())["questions"]
    out = []
    for q in raw:
        answerable = q.get("answerable", True)
        default_kind = (
            "unanswerable"
            if not answerable
            else ("paraphrase" if q["id"].startswith("p") else "direct")
        )
        out.append(
            EvalQuestion(
                q["id"],
                q["question"],
                q.get("expected_docs", []),
                answerable,
                q.get("kind", default_kind),
            )
        )
    return out


def retrieval_by_kind(questions: list[EvalQuestion], answers: list[Answer]) -> dict:
    """Hit rate and MRR per question kind: shows *where* a retriever wins or loses."""
    out: dict[str, dict[str, float]] = {}
    for kind in sorted({q.kind for q in questions if q.answerable}):
        pairs = [(q, a) for q, a in zip(questions, answers, strict=True) if q.kind == kind]
        rr = []
        for q, a in pairs:
            ranks = [
                i for i, (cid, _) in enumerate(a.retrieved, 1) if doc_of(cid) in q.expected_docs
            ]
            rr.append(1 / ranks[0] if ranks else 0.0)
        out[kind] = {
            "n": len(pairs),
            "hit_rate_at_k": float(np.mean([r > 0 for r in rr])),
            "mrr": float(np.mean(rr)),
        }
    return out


@dataclass
class SweepResult:
    thresholds: list[float]
    answer_rate: list[float]  # answerable questions that pass the threshold
    refusal_rate: list[float]  # unanswerable questions that are refused
    best: float


def sweep_threshold(
    relevances: list[float], answerable: list[bool], step: float = 0.005, upper: float = 0.6
) -> SweepResult:
    """Choose the refusal threshold from data instead of guessing.

    For each candidate threshold t: answer if relevance >= t, refuse otherwise. The score
    is balanced accuracy (mean of answer rate and refusal rate), so the 6 unanswerable
    questions count as much as the 36 answerable ones. Several thresholds usually tie;
    we take the middle of the best range, which leaves the largest safety margin on both
    sides (like a max-margin classifier).
    """
    rel = np.asarray(relevances, float)
    ok = np.asarray(answerable, bool)
    grid = np.round(np.arange(0, upper + step / 2, step), 4)
    ans = np.array([(rel[ok] >= t).mean() for t in grid])
    ref = np.array([(rel[~ok] < t).mean() for t in grid])
    balanced = (ans + ref) / 2
    best_idx = np.flatnonzero(balanced == balanced.max())
    # First contiguous block of best thresholds, then its midpoint.
    block = [best_idx[0]]
    for i in best_idx[1:]:
        if i != block[-1] + 1:
            break
        block.append(i)
    best = float(np.round((grid[block[0]] + grid[block[-1]]) / 2, 3))
    return SweepResult(grid.tolist(), ans.tolist(), ref.tolist(), best)


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def faithfulness(answer: Answer, chunk_text: dict[str, str]) -> float:
    words = tokenize(CITATION_RE.sub(" ", answer.text))
    if not words:
        return 0.0
    cited = set(tokenize(" ".join(chunk_text.get(c, "") for c in answer.citations)))
    return sum(w in cited for w in words) / len(words)


def score(
    questions: list[EvalQuestion], answers: list[Answer], chunk_text: dict[str, str]
) -> dict[str, float]:
    hits, rr, answered, refused_ok = [], [], [], []
    cite_ok, grounded, faith = [], [], []
    for q, a in zip(questions, answers, strict=True):
        if q.answerable:
            ranks = [
                i for i, (cid, _) in enumerate(a.retrieved, 1) if doc_of(cid) in q.expected_docs
            ]
            hits.append(bool(ranks))
            rr.append(1 / ranks[0] if ranks else 0.0)
            answered.append(not a.refused)
            if not a.refused:
                retrieved = {cid for cid, _ in a.retrieved}
                cite_ok.append(bool(a.citations) and set(a.citations) <= retrieved)
                grounded.append(any(doc_of(c) in q.expected_docs for c in a.citations))
                faith.append(faithfulness(a, chunk_text))
        else:
            refused_ok.append(a.refused)
    lat = [a.latency_ms for a in answers]
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return {
        "hit_rate_at_k": mean(hits),
        "mrr": mean(rr),
        "answer_rate": mean(answered),
        "refusal_rate": mean(refused_ok),
        "citation_ok": mean(cite_ok),
        "grounded": mean(grounded),
        "faithfulness": mean(faith),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "cost_per_1000_usd": 1000 * mean([a.cost_usd for a in answers]),
    }


def run(assistant: Assistant, questions: list[EvalQuestion]) -> tuple[dict, list[Answer]]:
    answers = [assistant.ask(q.question) for q in questions]
    chunk_text = {c.chunk_id: c.text for c in assistant.retriever.store.chunks()}
    return score(questions, answers, chunk_text), answers


def check_thresholds(metrics: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Return a list of failures; empty means the gate passes."""
    return [
        f"{name}: {metrics[name]:.3f} < {minimum}"
        for name, minimum in thresholds.items()
        if metrics.get(name, 0.0) < minimum
    ]


def to_markdown(results: dict[str, dict[str, float]]) -> str:
    cols = [
        "hit_rate_at_k",
        "mrr",
        "answer_rate",
        "refusal_rate",
        "citation_ok",
        "grounded",
        "faithfulness",
        "latency_p50_ms",
    ]
    lines = ["| setup | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
    return "\n".join(lines)


def write_report(results: dict, answers: dict[str, list[Answer]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.md").write_text(to_markdown(results) + "\n")
    payload = {"metrics": results, "answers": {k: [vars(a) for a in v] for k, v in answers.items()}}
    path = out_dir / "eval_results.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path

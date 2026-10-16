"""Log every question to SQLite: latency, tokens, cost, what was retrieved and cited."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .answer import Answer

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    question TEXT, answer TEXT, refused INTEGER, relevance REAL,
    retrieved TEXT, citations TEXT, latency_ms REAL,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, generator TEXT
)
"""


class Tracer:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)

    def log(self, a: Answer, generator: str) -> None:
        self.conn.execute(
            "INSERT INTO traces (question, answer, refused, relevance, retrieved, citations, "
            "latency_ms, input_tokens, output_tokens, cost_usd, generator) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                a.question,
                a.text,
                int(a.refused),
                a.relevance,
                json.dumps(a.retrieved),
                json.dumps(a.citations),
                a.latency_ms,
                a.input_tokens,
                a.output_tokens,
                a.cost_usd,
                generator,
            ),
        )
        self.conn.commit()

    def summary(self) -> dict:
        row = self.conn.execute(
            "SELECT count(*), avg(latency_ms), avg(refused), sum(cost_usd), "
            "sum(input_tokens), sum(output_tokens) FROM traces"
        ).fetchone()
        keys = [
            "questions",
            "avg_latency_ms",
            "refusal_rate",
            "total_cost_usd",
            "input_tokens",
            "output_tokens",
        ]
        return dict(zip(keys, row, strict=True))

"""A small Okapi BM25 keyword index (written out so every step is visible)."""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    "a an and are as at be by can do does for from how in is it of on or should that the this "
    "to was what when which who why with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()

    def fit(self, ids: list[str], texts: list[str]) -> BM25:
        self.ids = list(ids)
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.df = Counter(term for tf in self.tfs for term in tf)
        return self

    def scores(self, query: str) -> dict[str, float]:
        n = len(self.ids)
        if n == 0:
            return {}
        avg_len = sum(self.lengths) / n
        out: dict[str, float] = {}
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / avg_len)
                    out[self.ids[i]] = out.get(self.ids[i], 0.0) + idf * f * (self.k1 + 1) / denom
        return out

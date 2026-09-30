"""Shared retrieval contract used by every ranking method (integration layer)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ResourceMissingError(RuntimeError):
    """A prepared resource (index, embeddings, model, processed data) is not available."""


@dataclass(frozen=True)
class RankedList:
    """Ranked documents for one query, best first. `doc_idx` are positions in the corpus."""
    doc_idx: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.doc_idx)


class BaseRetriever:
    """Subclasses implement `score_batch` (dense score matrix: queries x documents)."""

    name = "base"
    min_score = 0.0   # results must score strictly above this (lexical: score 0 = no shared term)

    def score_batch(self, queries: list[str]) -> np.ndarray:
        raise NotImplementedError

    def search_batch(self, queries: list[str], depth: int, chunk_size: int = 128) -> list[RankedList]:
        out: list[RankedList] = []
        for start in range(0, len(queries), chunk_size):
            scores = self.score_batch(queries[start:start + chunk_size])
            for row in scores:
                # stable sort => deterministic ties (lower corpus position wins)
                order = np.argsort(-row, kind="stable")[:depth]
                order = order[row[order] > self.min_score]
                out.append(RankedList(order, row[order]))
        return out

    def search(self, query: str, depth: int) -> RankedList:
        return self.search_batch([query], depth)[0]

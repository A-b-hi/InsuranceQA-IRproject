"""Student 3 - Reciprocal Rank Fusion.

    RRF(d) = sum over ranking lists i of  1 / (k + rank_i(d))       (rank is 1-based)
Only rank positions are used, never the raw scores (which are not comparable across methods).
A document missing from a list simply gets no contribution from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FusedItem:
    doc: int
    score: float
    ranks: dict          # method -> 1-based rank in that method's list, or None if absent


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[int]], k: float = 60,
                           top_k: int | None = None) -> list[FusedItem]:
    if k < 0:
        raise ValueError("RRF constant k must be >= 0")
    if not rankings:
        raise ValueError("at least one ranking list is required")
    scores: dict[int, float] = {}
    ranks: dict[int, dict] = {}
    names = list(rankings)
    for name, docs in rankings.items():
        seen = set()
        position = 0
        for doc in docs:
            doc = int(doc)
            if doc in seen:          # ignore repeats inside one list
                continue
            seen.add(doc)
            position += 1
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + position)
            ranks.setdefault(doc, dict.fromkeys(names))[name] = position
    ordered = sorted(scores, key=lambda d: (-scores[d], d))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [FusedItem(d, scores[d], ranks[d]) for d in ordered]

"""Student 3 - retrieval metrics (binary relevance from the official InsuranceQA labels).

For one query with ranked list r_1..r_n and relevant set R:
  P@k      = |{relevant in top k}| / k
  Recall@k = |{relevant in top k}| / |R|
  AP       = (1/|R|) * sum_k P@k * rel(r_k)                     (MAP = mean over queries)
  nDCG@k   = DCG@k / IDCG@k,  DCG@k = sum_{i<=k} rel_i / log2(i + 1)
  PR curve = 11-point interpolated precision (max precision at recall >= level), averaged over queries.
Relevant documents that are not retrieved within the ranked list count as misses.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

PR_LEVELS = tuple(round(i / 10, 1) for i in range(11))


def precision_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def average_precision(ranked: Sequence[int], relevant: set) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def interpolated_precision(ranked: Sequence[int], relevant: set, levels=PR_LEVELS) -> list[float]:
    points, hits = [], 0                       # (recall, precision) at every relevant hit
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            points.append((hits / len(relevant), hits / i))
    out = []
    for level in levels:
        out.append(max((p for r, p in points if r >= level - 1e-12), default=0.0))
    return out


def evaluate_ranking(ranked_lists: Sequence[Sequence[int]], relevant_sets: Sequence[set],
                     ks=(10, 100), ndcg_k: int = 10, levels=PR_LEVELS) -> dict:
    """Mean metrics over all queries that have at least one relevant document."""
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must have the same length")
    rows = [(r, rel) for r, rel in zip(ranked_lists, relevant_sets) if rel]
    if not rows:
        raise ValueError("no query with a relevant document to evaluate")
    metrics = {f"P@{k}": float(np.mean([precision_at_k(r, rel, k) for r, rel in rows])) for k in ks}
    metrics.update({f"Recall@{k}": float(np.mean([recall_at_k(r, rel, k) for r, rel in rows])) for k in ks})
    metrics["MAP"] = float(np.mean([average_precision(r, rel) for r, rel in rows]))
    metrics[f"nDCG@{ndcg_k}"] = float(np.mean([ndcg_at_k(r, rel, ndcg_k) for r, rel in rows]))
    curve = np.mean([interpolated_precision(r, rel, levels) for r, rel in rows], axis=0)
    return {"queries": len(rows), "metrics": metrics,
            "pr_curve": [{"recall": lv, "precision": float(p)} for lv, p in zip(levels, curve)]}

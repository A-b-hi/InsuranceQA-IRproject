"""Student 3 - evaluation metrics (hand-computed expectations) and saved real results."""
import json
import math

import pytest

import config
from src.evaluation import (average_precision, evaluate_ranking, interpolated_precision, ndcg_at_k,
                            precision_at_k, recall_at_k)

RANKED, REL = [1, 2, 3, 4, 5], {2, 5}


def test_precision_and_recall():
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(0.4)
    assert precision_at_k(RANKED, REL, 2) == pytest.approx(0.5)
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(1.0)
    assert recall_at_k(RANKED, REL, 1) == 0.0


def test_average_precision():
    assert average_precision(RANKED, REL) == pytest.approx((1 / 2 + 2 / 5) / 2)
    assert average_precision([9, 8], REL) == 0.0           # nothing retrieved
    assert average_precision([2], {2, 5}) == pytest.approx(0.5)   # misses count against AP


def test_ndcg():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([2, 5, 1], REL, 3) == pytest.approx(1.0)


def test_interpolated_precision_curve():
    curve = interpolated_precision(RANKED, REL, levels=(0.0, 0.5, 0.6, 1.0))
    assert curve == pytest.approx([0.5, 0.5, 0.4, 0.4])
    assert interpolated_precision([9], REL, levels=(0.0, 1.0)) == [0.0, 0.0]


def test_evaluate_ranking_means_and_skips_unlabelled_queries():
    out = evaluate_ranking([RANKED, [2, 1]], [REL, {2}], ks=(2,))
    assert out["queries"] == 2
    assert out["metrics"]["P@2"] == pytest.approx((0.5 + 0.5) / 2)
    skipped = evaluate_ranking([RANKED, [1]], [REL, set()], ks=(2,))
    assert skipped["queries"] == 1
    assert [p["recall"] for p in out["pr_curve"]][0] == 0.0 and len(out["pr_curve"]) == 11


def test_invalid_input():
    with pytest.raises(ValueError):
        evaluate_ranking([RANKED], [REL, REL])
    with pytest.raises(ValueError):
        evaluate_ranking([RANKED], [set()])


def test_real_ranking_metrics_are_valid(engine, dataset):
    qs = dataset.queries["test"][:40]
    rel = [{dataset.id_to_index[g] for g in q.gold_ids} for q in qs]
    for method in ("tfidf", "bm25", "semantic", "hybrid"):
        ranked = [rl.doc_idx.tolist() for rl in engine.rank_batch(method, [q.question for q in qs], 1000)]
        out = evaluate_ranking(ranked, rel)
        assert all(0.0 <= v <= 1.0 for v in out["metrics"].values())
        precisions = [p["precision"] for p in out["pr_curve"]]
        assert precisions == sorted(precisions, reverse=True)       # interpolated precision never rises


def test_saved_evaluation_results_are_complete():
    path = config.EVAL_DIR / "results.json"
    if not path.is_file():
        pytest.skip("run scripts/evaluate.py first")
    data = json.loads(path.read_text())
    assert set(data["methods"]) == {"tfidf", "bm25", "semantic", "hybrid"}
    assert data["config"]["queries"] == 2000 and data["config"]["split"] == "test"
    for m in data["methods"].values():
        assert {"P@10", "Recall@10", "Recall@100", "MAP", "nDCG@10"} <= set(m["metrics"])
        assert len(m["pr_curve"]) == 11
    assert (config.EVAL_DIR / "pr_curve.png").stat().st_size > 10_000

"""Student 3 - Reciprocal Rank Fusion (hand-computed expectations)."""
import pytest

from src.rrf import reciprocal_rank_fusion


def test_scores_follow_formula():
    fused = reciprocal_rank_fusion({"a": [10, 20, 30], "b": [20, 10, 40]}, k=60)
    by_doc = {f.doc: f for f in fused}
    assert by_doc[10].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_doc[30].score == pytest.approx(1 / 63)
    assert by_doc[40].score == pytest.approx(1 / 63)


def test_documents_in_only_some_lists_are_handled():
    fused = reciprocal_rank_fusion({"a": [1, 2], "b": [3], "c": []})
    by_doc = {f.doc: f for f in fused}
    assert set(by_doc) == {1, 2, 3}
    assert by_doc[3].ranks == {"a": None, "b": 1, "c": None}
    assert by_doc[2].ranks["a"] == 2


def test_consensus_beats_a_single_high_rank():
    fused = reciprocal_rank_fusion({"a": [1, 9], "b": [2, 9], "c": [3, 9]})
    assert fused[0].doc == 9


def test_constant_k_is_configurable():
    assert reciprocal_rank_fusion({"a": [1]}, k=0)[0].score == pytest.approx(1.0)
    assert reciprocal_rank_fusion({"a": [1]}, k=10)[0].score == pytest.approx(1 / 11)


def test_ties_are_deterministic_and_top_k_truncates():
    fused = reciprocal_rank_fusion({"a": [5, 6], "b": [6, 5]}, top_k=1)
    assert [f.doc for f in fused] == [5]


def test_duplicate_entries_in_one_list_count_once():
    fused = reciprocal_rank_fusion({"a": [1, 1, 2]}, k=60)
    assert {f.doc: f.score for f in fused}[2] == pytest.approx(1 / 62)


def test_invalid_input():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({})
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": [1]}, k=-1)


def test_engine_hybrid_equals_independent_fusion_of_the_three_rankings(engine, dataset, real_question):
    q = real_question.question
    lists = {m: engine.retrievers[m].search(q, engine.rrf_depth) for m in ("tfidf", "bm25", "semantic")}
    expected = reciprocal_rank_fusion({m: rl.doc_idx.tolist() for m, rl in lists.items()}, k=engine.rrf_k, top_k=10)
    hybrid = engine.search(q, "hybrid", 10)
    assert [r.document_id for r in hybrid] == [dataset.doc_ids[f.doc] for f in expected]
    assert [r.score for r in hybrid] == pytest.approx([f.score for f in expected])
    assert hybrid[0].components.keys() == {"tfidf", "bm25", "semantic"}

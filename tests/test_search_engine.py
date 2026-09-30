"""Integration - central search layer, end to end on the real InsuranceQA data."""
import pytest

import config
from helpers import make_cfg
from src.retrieval_base import ResourceMissingError
from src.search_engine import (METHODS, InvalidMethodError, InvalidQueryError, SearchEngine)

FIELDS = {"document_id", "question", "answer", "score", "rank", "method", "domain", "components"}


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_consistent_ranked_real_answers(engine, dataset, real_question, method):
    results = engine.search(real_question.question, method, 10)
    assert 0 < len(results) <= 10
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(r.method == method and set(r.to_dict()) == FIELDS for r in results)
    for r in results:                                   # the actual InsuranceQA answer is returned
        doc = dataset.get(r.document_id)
        assert r.answer == doc.answer and r.question == doc.question and r.answer.strip()
    assert len({r.document_id for r in results}) == len(results)


def test_official_answers_are_found_for_some_real_questions(engine, dataset):
    """Sanity: semantic search puts an official answer in the top 10 for a clear majority of real questions."""
    qs = dataset.queries["test"][:100]
    hits = sum(any(r.document_id in set(q.gold_ids) for r in engine.search(q.question, "semantic", 10)) for q in qs)
    assert hits >= 50


def test_hybrid_reports_the_rank_in_each_contributing_list(engine, real_question):
    top = engine.search(real_question.question, "hybrid", 3)[0]
    assert set(top.components) == {"tfidf", "bm25", "semantic"}
    assert any(v is not None for v in top.components.values())


def test_compare_returns_all_methods_and_matches_single_searches(engine, real_question):
    out = engine.compare(real_question.question, 5)
    assert list(out) == list(METHODS)
    for m in METHODS:
        single = engine.search(real_question.question, m, 5)
        assert [r.document_id for r in out[m]] == [r.document_id for r in single]


def test_top_k_is_clamped(engine):
    assert len(engine.search("life insurance", "bm25", 10_000)) == config.MAX_TOP_K
    assert len(engine.search("life insurance", "bm25", 0)) == 1


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "\x00\x07"])
def test_empty_and_blank_queries_rejected(engine, bad):
    with pytest.raises(InvalidQueryError):
        engine.search(bad, "hybrid")


@pytest.mark.parametrize("bad", [None, 42, ["a"], b"bytes"])
def test_malformed_query_types_rejected(engine, bad):
    with pytest.raises(InvalidQueryError):
        engine.search(bad, "hybrid")


def test_overlong_query_rejected(engine):
    with pytest.raises(InvalidQueryError):
        engine.search("a " * 400, "hybrid")


def test_unknown_method_rejected(engine):
    with pytest.raises(InvalidMethodError):
        engine.search("life insurance", "pagerank")


def test_unknown_query_returns_no_results(engine):
    for m in METHODS:
        assert engine.search("zzzqqq xxyyzz", m, 10) == []


def test_query_of_only_stop_words_returns_no_lexical_results(engine):
    assert engine.search("what is the", "bm25") == [] and engine.search("what is the", "tfidf") == []


def test_unusual_but_valid_input_does_not_crash(engine):
    for q in ["<script>alert(1)</script>", "insurance" * 50, "保险 cost?", "' OR 1=1 --", "car\x00insurance"]:
        assert isinstance(engine.search(q, "hybrid", 5), list)


def test_sample_queries_are_real_dataset_questions(engine, dataset):
    real = {q.question for q in dataset.queries["test"]}
    sample = engine.sample_queries(6)
    assert len(sample) == 6 and all(s["question"] in real for s in sample)
    assert len({s["domain"] for s in sample}) == 6


def test_official_relevant_ids(engine, real_question):
    assert set(real_question.gold_ids) <= engine.official_relevant_ids(real_question.question)
    assert engine.official_relevant_ids("not a dataset question") == set()


# ---- missing resources ---------------------------------------------------------------
def test_missing_processed_data(tmp_path):
    with pytest.raises(ResourceMissingError):
        SearchEngine.from_config(make_cfg(PROCESSED_DIR=tmp_path))


def test_missing_indexes(tmp_path):
    with pytest.raises(ResourceMissingError):
        SearchEngine.from_config(make_cfg(INDEX_DIR=tmp_path))


def test_missing_embeddings_disable_semantic_and_hybrid_only(tmp_path):
    eng = SearchEngine.from_config(make_cfg(EMBEDDING_DIR=tmp_path))
    avail = eng.available_methods()
    assert avail["tfidf"]["available"] and avail["bm25"]["available"]
    assert not avail["semantic"]["available"] and not avail["hybrid"]["available"]
    assert "build_embeddings" in avail["semantic"]["reason"]
    assert eng.search("life insurance", "bm25", 3)
    for m in ("semantic", "hybrid"):
        with pytest.raises(ResourceMissingError):
            eng.search("life insurance", m)
    assert list(eng.compare("life insurance", 3)) == ["tfidf", "bm25"]

"""Student 2 - BM25. Formula tests use a tiny hand-written corpus (test fixture, not project data)."""
import math

import numpy as np
import pytest

from src.bm25_retriever import BM25Retriever
from src.retrieval_base import ResourceMissingError

CORPUS = [["tree", "rock", "rock"], ["tree", "sand", "sand", "sand", "sand"], ["moon"]]


def expected_score(tf, dl, df, n, avgdl, k1, b):
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def test_matches_bm25_formula():
    r = BM25Retriever.build(CORPUS, k1=1.2, b=0.75)
    avgdl = (3 + 5 + 1) / 3
    scores = r.score_batch(["rock"])[0]
    assert scores[0] == pytest.approx(expected_score(2, 3, 1, 3, avgdl, 1.2, 0.75))
    assert scores[1] == 0 and scores[2] == 0


def test_b_controls_length_normalisation():
    docs = [["tree", "rock"], ["tree", "sand", "sand", "sand", "sand", "sand"], ["moon"]]
    no_norm = BM25Retriever.build(docs, k1=1.5, b=0.0).score_batch(["tree"])[0]
    full_norm = BM25Retriever.build(docs, k1=1.5, b=1.0).score_batch(["tree"])[0]
    assert no_norm[0] == pytest.approx(no_norm[1])        # same tf, length ignored
    assert full_norm[0] > full_norm[1]                    # shorter document wins


def test_k1_controls_term_frequency_saturation():
    docs = [["rock"] * 10 + ["sand"], ["rock", "sand"], ["moon"]]
    low = BM25Retriever.build(docs, k1=0.0, b=0.0).score_batch(["rock"])[0]
    assert low[0] == pytest.approx(low[1])                # k1 = 0 -> tf is ignored
    high = BM25Retriever.build(docs, k1=3.0, b=0.0).score_batch(["rock"])[0]
    assert high[0] > high[1]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Retriever.build(CORPUS, k1=-1)
    with pytest.raises(ValueError):
        BM25Retriever.build(CORPUS, b=1.5)


def test_with_params_reweights_without_rebuilding():
    r = BM25Retriever.build(CORPUS, 1.5, 0.75)
    r2 = r.with_params(0.5, 0.2)
    assert (r2.k1, r2.b) == (0.5, 0.2)
    assert not np.allclose(r.score_batch(["tree"]), r2.score_batch(["tree"]))


def test_ranking_unknown_and_empty_queries():
    r = BM25Retriever.build(CORPUS)
    assert list(r.search("rock", 5).doc_idx) == [0]
    assert r.search("zzzz", 5).doc_idx.size == 0 and r.search("", 5).doc_idx.size == 0


def test_save_and_load_roundtrip(tmp_path):
    r = BM25Retriever.build(CORPUS)
    r.save(tmp_path / "b.pkl")
    again = BM25Retriever.load(tmp_path / "b.pkl", 1.5, 0.75)
    assert np.allclose(again.score_batch(["rock"]), r.score_batch(["rock"]))


def test_missing_index_file(tmp_path):
    with pytest.raises(ResourceMissingError):
        BM25Retriever.load(tmp_path / "nope.pkl")


def test_real_index_returns_ranked_real_answers(engine, real_question):
    ranked = engine.retrievers["bm25"].search(real_question.question, 10)
    assert 0 < len(ranked) <= 10 and np.all(np.diff(ranked.scores) <= 1e-12)

"""Student 2 - TF-IDF. Formula tests use a tiny hand-written corpus (test fixture, not project data)."""
import math

import numpy as np
import pytest

from src.retrieval_base import ResourceMissingError
from src.tfidf_retriever import TfidfRetriever

CORPUS = [["tree", "rock", "rock"], ["tree", "sand"], ["moon"]]


@pytest.fixture()
def tf():
    return TfidfRetriever.build(CORPUS)


def test_weights_follow_documented_formula(tf):
    n = 3
    idf_tree, idf_rock = math.log10(n / 2), math.log10(n / 1)
    w_tree, w_rock = 1 * idf_tree, (1 + math.log10(2)) * idf_rock
    norm = math.sqrt(w_tree ** 2 + w_rock ** 2)
    row = tf.doc_matrix.getrow(0).toarray().ravel()
    assert row[tf.vocab["tree"]] == pytest.approx(w_tree / norm)
    assert row[tf.vocab["rock"]] == pytest.approx(w_rock / norm)
    assert np.linalg.norm(row) == pytest.approx(1.0)


def test_cosine_similarity_score(tf):
    # query "rock" -> unit vector on 'rock'; cosine with doc0 = its normalised rock weight
    n = 3
    w_tree, w_rock = math.log10(n / 2), (1 + math.log10(2)) * math.log10(n)
    expected = w_rock / math.sqrt(w_tree ** 2 + w_rock ** 2)
    scores = tf.score_batch(["rock"])[0]
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0 and scores[2] == 0


def test_ranking_and_unknown_terms(tf):
    ranked = tf.search("tree rock", 10)
    assert list(ranked.doc_idx) == [0, 1]                 # doc2 shares no term -> not returned
    assert tf.search("zzzz qqqq", 10).doc_idx.size == 0
    assert tf.search("", 10).doc_idx.size == 0


def test_term_in_every_document_has_zero_idf():
    r = TfidfRetriever.build([["tree", "rock"], ["tree", "sand"]])
    assert r.search("tree", 5).doc_idx.size == 0


def test_save_and_load_roundtrip(tf, tmp_path):
    tf.save(tmp_path / "t.pkl")
    again = TfidfRetriever.load(tmp_path / "t.pkl")
    assert np.allclose(again.score_batch(["rock"]), tf.score_batch(["rock"]))


def test_missing_index_file(tmp_path):
    with pytest.raises(ResourceMissingError):
        TfidfRetriever.load(tmp_path / "nope.pkl")


def test_real_index_returns_ranked_real_answers(engine, dataset, real_question):
    ranked = engine.retrievers["tfidf"].search(real_question.question, 10)
    assert 0 < len(ranked) <= 10
    assert np.all(np.diff(ranked.scores) <= 1e-12) and np.all(ranked.scores > 0) and np.all(ranked.scores <= 1.0 + 1e-9)

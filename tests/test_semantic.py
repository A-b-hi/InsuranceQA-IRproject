"""Student 3 - Sentence Transformer retrieval (real model + real cached embeddings)."""
import numpy as np
import pytest

import config
from src.retrieval_base import ResourceMissingError
from src.semantic_retriever import SemanticRetriever, doc_ids_fingerprint, resolve_device


def test_embeddings_are_cached_and_normalised(engine, dataset):
    emb = engine.retrievers["semantic"].doc_embeddings
    assert emb.shape == (len(dataset), 384)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


def test_search_returns_ranked_results(engine):
    ranked = engine.retrievers["semantic"].search("What does term life insurance cost?", 10)
    assert len(ranked) == 10 and np.all(np.diff(ranked.scores) <= 1e-9)
    assert ranked.scores[0] > 0.6


def test_semantic_matches_meaning_not_just_words(engine, dataset):
    ranked = engine.retrievers["semantic"].search("Will my insurer pay if a tree falls on my house?", 10)
    domains = [dataset.documents[i].domain for i in ranked.doc_idx]
    assert sum("home" in d or "renters" in d for d in domains) >= 5


def test_query_encoding_is_deterministic_and_batched(engine):
    s = engine.retrievers["semantic"]
    a, b = s.encode_queries(["medicare coverage"]), s.encode_queries(["medicare coverage", "car insurance"])
    assert np.allclose(a[0], b[0], atol=1e-5) and b.shape == (2, 384)


def test_nonsense_query_is_below_the_relevance_floor(engine):
    assert len(engine.retrievers["semantic"].search("zzzqqq xxyyzz", 10)) == 0


def test_device_resolution():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_missing_embeddings(tmp_path):
    with pytest.raises(ResourceMissingError):
        SemanticRetriever.load(tmp_path, config.MODEL_DIR)


def test_missing_model(tmp_path, dataset):
    with pytest.raises(ResourceMissingError):
        SemanticRetriever.load(config.EMBEDDING_DIR, tmp_path, dataset.doc_ids)


def test_embeddings_that_do_not_match_documents_are_rejected(dataset):
    with pytest.raises(ResourceMissingError):
        SemanticRetriever.load(config.EMBEDDING_DIR, config.MODEL_DIR, dataset.doc_ids[:-1])
    assert doc_ids_fingerprint([1, 2]) != doc_ids_fingerprint([2, 1])

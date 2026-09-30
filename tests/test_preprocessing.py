"""Student 1 - preprocessing."""
import pytest

from src.preprocessing import (STOP_WORDS, clean_for_embedding, clean_text, preprocess,
                               remove_stop_words, stem, tokenize)


def test_clean_text_collapses_whitespace_and_control_chars():
    assert clean_text("  a \t b\n\nc\x07d ") == "a b c d"


def test_clean_text_rejects_non_strings():
    with pytest.raises(TypeError):
        clean_text(None)


def test_tokenize_lowercases_and_handles_punctuation():
    assert tokenize("Don't pay $5,000!") == ["dont", "pay", "5", "000"]
    assert tokenize("") == [] and tokenize("?!  ...") == []


def test_stop_words_removed():
    assert remove_stop_words(["what", "is", "the", "premium"]) == ["premium"]
    assert "the" in STOP_WORDS and "premium" not in STOP_WORDS


def test_stemming_merges_word_forms():
    assert stem("insurance") == stem("insurances")
    assert preprocess("Is my Term Life Insurance covered?") == ["term", "life", "insur", "cover"]


def test_preprocess_options():
    assert preprocess("The running policies", use_stemming=False) == ["running", "policies"]
    assert preprocess("The running", drop_stop_words=False, use_stemming=False) == ["the", "running"]


def test_embedding_text_keeps_original_wording():
    assert clean_for_embedding("  What IS   term life? ") == "What IS term life?"


def test_stored_document_tokens_match_the_pipeline(dataset):
    for doc in dataset.documents[:200]:
        assert doc.tokens == preprocess(doc.answer)

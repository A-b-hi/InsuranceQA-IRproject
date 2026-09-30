"""Student 1 - text preprocessing shared by indexing and querying.

Two representations are produced from the same original text:
  * `preprocess()`         -> stemmed, stop-word-free tokens for TF-IDF / BM25
  * `clean_for_embedding()` -> lightly cleaned natural text for the Sentence Transformer
The original text is never modified; it is kept separately for display.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from nltk.stem import PorterStemmer

# Standard English stop-word list (NLTK's 179 words, apostrophes removed because
# `tokenize` strips them). Written out here so no corpus download is needed.
_STOP = """
i me my myself we our ours ourselves you youre youve youll youd your yours yourself yourselves
he him his himself she shes her hers herself it its itself they them their theirs themselves
what which who whom this that thatll these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will just don dont should
shouldve now d ll m o re ve y ain aren arent couldn couldnt didn didnt doesn doesnt hadn hadnt
hasn hasnt haven havent isn isnt ma mightn mightnt mustn mustnt needn neednt shan shant
shouldn shouldnt wasn wasnt weren werent won wont wouldn wouldnt
""".split()
STOP_WORDS = frozenset(_STOP)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")
_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    return _stemmer.stem(word)


def clean_text(text: str) -> str:
    """Unicode-normalise, drop control characters, collapse whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lower-case and split into alphanumeric tokens (apostrophes are dropped: don't -> dont)."""
    text = clean_text(text).lower().replace("'", "").replace("\u2019", "")
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 1 or t.isdigit()]


def remove_stop_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def preprocess(text: str, use_stemming: bool = True, drop_stop_words: bool = True) -> list[str]:
    """Full pipeline: clean -> lowercase -> tokenize -> stop words -> Porter stem."""
    tokens = tokenize(text)
    if drop_stop_words:
        tokens = remove_stop_words(tokens)
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def clean_for_embedding(text: str) -> str:
    """Natural text for the Sentence Transformer: cleaned whitespace, original wording kept."""
    return clean_text(text)

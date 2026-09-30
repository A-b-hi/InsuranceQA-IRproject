"""Student 2 - TF-IDF retrieval implemented from scratch (no search library).

Weighting scheme (documented, log-scaled):
    TF(t, d)     = 1 + log10(tf)              (0 when the term is absent)
    IDF(t)       = log10(N / df)
    weight(t, d) = TF * IDF
Documents and queries are L2-normalised, so the dot product is the cosine similarity used for ranking.
Query terms that never occur in the corpus are ignored.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .preprocessing import preprocess
from .retrieval_base import BaseRetriever, ResourceMissingError

SCHEME = "TF = 1 + log10(tf); IDF = log10(N / df); cosine similarity"


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, vocab: dict[str, int], idf: np.ndarray, doc_matrix: sparse.csr_matrix):
        self.vocab = vocab
        self.idf = idf
        self.doc_matrix = doc_matrix                     # N x V, rows L2-normalised
        self._doc_matrix_t = doc_matrix.T.tocsr()        # V x N, for fast query x corpus products

    @classmethod
    def build(cls, token_lists: list[list[str]]) -> "TfidfRetriever":
        n_docs = len(token_lists)
        vocab: dict[str, int] = {}
        indptr, indices, tfs = [0], [], []
        for tokens in token_lists:
            for term, count in Counter(tokens).items():
                indices.append(vocab.setdefault(term, len(vocab)))
                tfs.append(count)
            indptr.append(len(indices))
        tf = sparse.csr_matrix((np.array(tfs, dtype=np.float64), np.array(indices, dtype=np.int32),
                                np.array(indptr, dtype=np.int64)), shape=(n_docs, len(vocab)))
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log10(n_docs / df)
        weights = tf.copy()
        weights.data = (1.0 + np.log10(weights.data)) * idf[weights.indices]
        norms = np.sqrt(np.asarray(weights.multiply(weights).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        weights = sparse.diags(1.0 / norms) @ weights
        return cls(vocab, idf, weights.tocsr())

    # ---- persistence -------------------------------------------------------------------
    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"vocab": self.vocab, "idf": self.idf, "doc_matrix": self.doc_matrix,
                         "scheme": SCHEME}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        path = Path(path)
        if not path.is_file():
            raise ResourceMissingError(f"TF-IDF index not found at {path}. Run: python scripts/build_indexes.py")
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(data["vocab"], data["idf"], data["doc_matrix"])

    # ---- querying ----------------------------------------------------------------------
    def query_matrix(self, queries: list[str]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for i, query in enumerate(queries):
            for term, count in Counter(preprocess(query)).items():
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append((1.0 + np.log10(count)) * self.idf[j])
        q = sparse.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)))
        norms = np.sqrt(np.asarray(q.multiply(q).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return (sparse.diags(1.0 / norms) @ q).tocsr()

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return (self.query_matrix(queries) @ self._doc_matrix_t).toarray()

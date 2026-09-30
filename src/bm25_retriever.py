"""Student 2 - BM25 (Okapi) retrieval implemented from scratch, independent of the TF-IDF code.

    score(q, d) = sum over query terms t of  IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))      (never negative)
`k1` (term-frequency saturation) and `b` (length normalisation) are configurable.
A term repeated in the query counts once per occurrence.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .preprocessing import preprocess
from .retrieval_base import BaseRetriever, ResourceMissingError


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, vocab: dict[str, int], counts: sparse.csr_matrix, doc_len: np.ndarray,
                 k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self.vocab = vocab
        self.counts = counts              # N x V raw term counts
        self.doc_len = doc_len
        self.k1, self.b = float(k1), float(b)
        n_docs = counts.shape[0]
        self.avgdl = float(doc_len.mean()) if n_docs else 0.0
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        self._weights_t = self._term_weights().T.tocsr()      # V x N

    def _term_weights(self) -> sparse.csr_matrix:
        w = self.counts.astype(np.float64)
        row_of_entry = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[row_of_entry] / self.avgdl)
        tf = w.data
        w.data = self.idf[w.indices] * tf * (self.k1 + 1.0) / (tf + norm)
        return w

    def with_params(self, k1: float, b: float) -> "BM25Retriever":
        return BM25Retriever(self.vocab, self.counts, self.doc_len, k1, b)

    @classmethod
    def build(cls, token_lists: list[list[str]], k1: float = 1.5, b: float = 0.75) -> "BM25Retriever":
        vocab: dict[str, int] = {}
        indptr, indices, tfs = [0], [], []
        for tokens in token_lists:
            for term, count in Counter(tokens).items():
                indices.append(vocab.setdefault(term, len(vocab)))
                tfs.append(count)
            indptr.append(len(indices))
        counts = sparse.csr_matrix((np.array(tfs, dtype=np.float32), np.array(indices, dtype=np.int32),
                                    np.array(indptr, dtype=np.int64)), shape=(len(token_lists), len(vocab)))
        doc_len = np.array([len(t) for t in token_lists], dtype=np.float64)
        return cls(vocab, counts, doc_len, k1, b)

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"vocab": self.vocab, "counts": self.counts, "doc_len": self.doc_len},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, k1: float = 1.5, b: float = 0.75) -> "BM25Retriever":
        path = Path(path)
        if not path.is_file():
            raise ResourceMissingError(f"BM25 index not found at {path}. Run: python scripts/build_indexes.py")
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(data["vocab"], data["counts"], data["doc_len"], k1, b)

    def query_matrix(self, queries: list[str]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for i, query in enumerate(queries):
            for term, count in Counter(preprocess(query)).items():
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(count))
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)))

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return (self.query_matrix(queries) @ self._weights_t).toarray()

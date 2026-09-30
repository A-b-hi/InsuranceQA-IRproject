"""Central search layer - one interface over TF-IDF, BM25, semantic and hybrid (RRF) retrieval.

Integration module shared by all three students. Flask routes and evaluation scripts only talk to
`SearchEngine`; the retrieval algorithms live in their own modules.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import config as default_config
from .bm25_retriever import BM25Retriever
from .dataset import InsuranceQADataset
from .preprocessing import clean_text
from .retrieval_base import RankedList, ResourceMissingError
from .rrf import FusedItem, reciprocal_rank_fusion
from .semantic_retriever import SemanticRetriever
from .tfidf_retriever import TfidfRetriever

METHODS = ("hybrid", "tfidf", "bm25", "semantic")
BASE_METHODS = ("tfidf", "bm25", "semantic")

METHOD_INFO = {
    "hybrid": {"label": "Hybrid RRF", "tagline": "Combines all three",
               "description": "Merges the TF-IDF, BM25 and semantic rankings by rank position, "
                              "so an answer that several methods agree on rises to the top."},
    "tfidf": {"label": "TF-IDF", "tagline": "Keyword weighting",
              "description": "Weights the words you share with an answer by how rare they are, "
                             "then ranks by cosine similarity."},
    "bm25": {"label": "BM25", "tagline": "Keyword ranking",
             "description": "Keyword matching that accounts for answer length and repeated terms."},
    "semantic": {"label": "Semantic", "tagline": "Meaning match",
                 "description": "Compares meaning with a Sentence Transformer, so an answer can match "
                                "even when it uses different words."},
}


class InvalidQueryError(ValueError):
    pass


class InvalidMethodError(ValueError):
    pass


@dataclass
class SearchResult:
    document_id: int
    question: str
    answer: str
    score: float
    rank: int
    method: str
    domain: str = ""
    components: dict | None = None      # hybrid only: rank of this answer in each contributing list

    def to_dict(self) -> dict:
        return asdict(self)


class SearchEngine:
    def __init__(self, dataset: InsuranceQADataset, retrievers: dict, unavailable: dict | None = None,
                 rrf_k: float = 60, rrf_depth: int = 1000, cfg=default_config):
        self.dataset = dataset
        self.retrievers = retrievers
        self.unavailable = dict(unavailable or {})
        self.rrf_k = rrf_k
        self.rrf_depth = rrf_depth
        self.cfg = cfg
        if not any(m in retrievers for m in BASE_METHODS):
            raise ResourceMissingError("No retrieval method is available.")
        if not all(m in retrievers for m in BASE_METHODS):
            missing = [m for m in BASE_METHODS if m not in retrievers]
            self.unavailable.setdefault("hybrid", "Hybrid RRF needs all three rankings; missing: " + ", ".join(missing))

    # ---- construction -----------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg=default_config) -> "SearchEngine":
        try:
            dataset = InsuranceQADataset.load(cfg.PROCESSED_DIR)
        except Exception as exc:
            raise ResourceMissingError(str(exc)) from exc
        retrievers = {
            "tfidf": TfidfRetriever.load(Path(cfg.INDEX_DIR) / "tfidf.pkl"),
            "bm25": BM25Retriever.load(Path(cfg.INDEX_DIR) / "bm25.pkl", cfg.BM25_K1, cfg.BM25_B),
        }
        unavailable = {}
        try:
            retrievers["semantic"] = SemanticRetriever.load(
                cfg.EMBEDDING_DIR, cfg.MODEL_DIR, dataset.doc_ids, cfg.DEVICE,
                cfg.SEMANTIC_MAX_SEQ_LENGTH, cfg.SEMANTIC_BATCH_SIZE, cfg.SEMANTIC_MIN_SCORE)
        except ResourceMissingError as exc:
            unavailable["semantic"] = str(exc)
            unavailable["hybrid"] = "Hybrid RRF needs the semantic ranking. " + str(exc)
        return cls(dataset, retrievers, unavailable, cfg.RRF_K, cfg.RRF_DEPTH, cfg)

    # ---- validation -------------------------------------------------------------------
    def validate_query(self, query) -> str:
        if not isinstance(query, str):
            raise InvalidQueryError("The question must be text.")
        query = clean_text(query)
        if not query:
            raise InvalidQueryError("Enter a question to search.")
        if len(query) > self.cfg.MAX_QUERY_CHARS:
            raise InvalidQueryError(f"Keep the question under {self.cfg.MAX_QUERY_CHARS} characters.")
        return query

    def validate_method(self, method) -> str:
        if method not in METHODS:
            raise InvalidMethodError(f"Unknown method '{method}'. Choose one of: {', '.join(METHODS)}.")
        if method in self.unavailable:
            raise ResourceMissingError(self.unavailable[method])
        return method

    def available_methods(self) -> dict:
        return {m: {"available": m not in self.unavailable, "reason": self.unavailable.get(m, "")} for m in METHODS}

    # ---- ranking ----------------------------------------------------------------------
    def _base_lists(self, queries: list[str], depth: int) -> dict[str, list[RankedList]]:
        return {m: self.retrievers[m].search_batch(queries, depth) for m in BASE_METHODS}

    def _fuse(self, per_method: dict[str, RankedList], top_k: int | None) -> list[FusedItem]:
        return reciprocal_rank_fusion({m: rl.doc_idx.tolist() for m, rl in per_method.items()},
                                      k=self.rrf_k, top_k=top_k)

    def rank_batch(self, method: str, queries: list[str], depth: int) -> list[RankedList]:
        """Ranked lists (corpus positions) for many queries; used by the evaluation script."""
        self.validate_method(method)
        if method != "hybrid":
            return self.retrievers[method].search_batch(queries, depth)
        lists = self._base_lists(queries, self.rrf_depth)
        out = []
        for i in range(len(queries)):
            fused = self._fuse({m: lists[m][i] for m in BASE_METHODS}, depth)
            out.append(RankedList(np.array([f.doc for f in fused], dtype=np.int64),
                                  np.array([f.score for f in fused])))
        return out

    def _result(self, method: str, idx: int, score: float, rank: int, components=None) -> SearchResult:
        doc = self.dataset.documents[idx]
        return SearchResult(doc.doc_id, doc.question, doc.answer, float(score), rank, method, doc.domain, components)

    def search(self, query: str, method: str = "hybrid", top_k: int = 10) -> list[SearchResult]:
        query = self.validate_query(query)
        method = self.validate_method(method)
        top_k = max(1, min(int(top_k), self.cfg.MAX_TOP_K))
        if method == "hybrid":
            lists = {m: self.retrievers[m].search(query, self.rrf_depth) for m in BASE_METHODS}
            return [self._result("hybrid", f.doc, f.score, r, f.ranks)
                    for r, f in enumerate(self._fuse(lists, top_k), 1)]
        ranked = self.retrievers[method].search(query, top_k)
        return [self._result(method, int(i), s, r) for r, (i, s) in enumerate(zip(ranked.doc_idx, ranked.scores), 1)]

    def compare(self, query: str, top_k: int | None = None) -> dict[str, list[SearchResult]]:
        """All available methods for one query. The three base rankings are computed once and shared."""
        query = self.validate_query(query)
        top_k = max(1, min(int(top_k or self.cfg.COMPARE_TOP_K), self.cfg.MAX_TOP_K))
        lists = {m: self.retrievers[m].search(query, self.rrf_depth) for m in BASE_METHODS if m in self.retrievers}
        out = {}
        for m in METHODS:
            if m in self.unavailable:
                continue
            if m == "hybrid":
                out[m] = [self._result(m, f.doc, f.score, r, f.ranks) for r, f in enumerate(self._fuse(lists, top_k), 1)]
            else:
                rl = lists[m]
                out[m] = [self._result(m, int(i), s, r) for r, (i, s) in
                          enumerate(zip(rl.doc_idx[:top_k], rl.scores[:top_k]), 1)]
        return out

    # ---- dataset helpers ---------------------------------------------------------------
    def official_relevant_ids(self, query: str) -> set[int]:
        return self.dataset.official_relevant_ids(query)

    def sample_queries(self, n: int = 6, split: str = "test", seed: int = 7) -> list[dict]:
        """Real questions from the dataset (one per domain first) for the example chips."""
        pool = [q for q in self.dataset.queries.get(split, []) if 4 <= len(q.question.split()) <= 12]
        rng = random.Random(seed)
        rng.shuffle(pool)
        picked, seen = [], set()
        for q in pool:
            if q.domain not in seen:
                seen.add(q.domain)
                picked.append({"question": q.question, "domain": q.domain})
            if len(picked) == n:
                break
        return picked

    def stats(self) -> dict:
        return {"answers": len(self.dataset),
                "questions": sum(len(v) for v in self.dataset.queries.values()),
                "domains": len({d.domain for d in self.dataset.documents if d.domain})}

    def params(self, method: str) -> str:
        if method == "tfidf":
            return "TF = 1 + log10(tf), IDF = log10(N/df), cosine similarity"
        if method == "bm25":
            r = self.retrievers.get("bm25")
            return f"k1 = {r.k1:g}, b = {r.b:g}" if r else ""
        if method == "semantic":
            return "MiniLM-L6 Sentence Transformer, cosine similarity"
        return f"k = {self.rrf_k:g}, top {self.rrf_depth} answers from each ranking"

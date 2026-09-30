"""Student 2 - build and persist the TF-IDF and BM25 indexes.

    python scripts/build_indexes.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.bm25_retriever import BM25Retriever
from src.dataset import InsuranceQADataset
from src.tfidf_retriever import TfidfRetriever


def main() -> int:
    try:
        dataset = InsuranceQADataset.load(config.PROCESSED_DIR)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    tokens = [d.tokens for d in dataset.documents]
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    t = time.time()
    tfidf = TfidfRetriever.build(tokens)
    tfidf.save(config.INDEX_DIR / "tfidf.pkl")
    print(f"TF-IDF: {tfidf.doc_matrix.shape[0]} docs x {tfidf.doc_matrix.shape[1]} terms ({time.time() - t:.1f}s)")
    t = time.time()
    bm25 = BM25Retriever.build(tokens, config.BM25_K1, config.BM25_B)
    bm25.save(config.INDEX_DIR / "bm25.pkl")
    print(f"BM25:   avg doc length {bm25.avgdl:.1f} terms, k1={bm25.k1}, b={bm25.b} ({time.time() - t:.1f}s)")
    print(f"Indexes written to {config.INDEX_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

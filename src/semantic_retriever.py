"""Student 3 - Sentence Transformer semantic retrieval.

Document embeddings are computed once by scripts/build_embeddings.py and cached on disk;
a search only encodes the query and takes a dot product (embeddings are L2-normalised, so
the dot product is cosine similarity). Uses CUDA / Apple MPS when available, otherwise CPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .retrieval_base import BaseRetriever, ResourceMissingError

EMBEDDINGS_FILE = "doc_embeddings.npy"
META_FILE = "embedding_meta.json"


def resolve_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def doc_ids_fingerprint(doc_ids) -> str:
    return hashlib.sha256(",".join(map(str, doc_ids)).encode()).hexdigest()


def load_model(model_dir: Path, device: str = "auto", max_seq_length: int = 256):
    model_dir = Path(model_dir)
    if not (model_dir / "modules.json").is_file():
        raise ResourceMissingError(
            f"Sentence Transformer model not found in {model_dir}. "
            "Place the model folder there or set INSURANCEQA_MODEL_DIR.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_dir), device=resolve_device(device))
    model.max_seq_length = max_seq_length
    return model


class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(self, doc_embeddings: np.ndarray, model, batch_size: int = 64, min_score: float = 0.30):
        self.min_score = min_score      # cosine floor: below it nothing is a meaningful match
        self.doc_embeddings = np.ascontiguousarray(doc_embeddings, dtype=np.float32)
        self.model = model
        self.batch_size = batch_size

    @classmethod
    def load(cls, embedding_dir: Path, model_dir: Path, doc_ids=None, device: str = "auto",
             max_seq_length: int = 256, batch_size: int = 64, min_score: float = 0.30) -> "SemanticRetriever":
        embedding_dir = Path(embedding_dir)
        emb_path, meta_path = embedding_dir / EMBEDDINGS_FILE, embedding_dir / META_FILE
        if not emb_path.is_file() or not meta_path.is_file():
            raise ResourceMissingError(
                f"Document embeddings not found in {embedding_dir}. Run: python scripts/build_embeddings.py")
        embeddings = np.load(emb_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if doc_ids is not None:
            if embeddings.shape[0] != len(doc_ids) or meta.get("doc_ids_sha256") != doc_ids_fingerprint(doc_ids):
                raise ResourceMissingError(
                    "Embeddings do not match the processed documents. Re-run: python scripts/build_embeddings.py")
        return cls(embeddings, load_model(model_dir, device, max_seq_length), batch_size, min_score)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(queries, batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return self.encode_queries(queries) @ self.doc_embeddings.T

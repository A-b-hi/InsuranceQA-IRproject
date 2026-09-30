"""Student 3 - encode every InsuranceQA answer once and cache the embeddings.

    python scripts/build_embeddings.py [--device auto|cpu|cuda] [--batch-size 64]

Progress is checkpointed in chunks, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.dataset import InsuranceQADataset
from src.preprocessing import clean_for_embedding
from src.retrieval_base import ResourceMissingError
from src.semantic_retriever import (EMBEDDINGS_FILE, META_FILE, doc_ids_fingerprint,
                                    load_model, resolve_device)

CHUNK = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--batch-size", type=int, default=config.SEMANTIC_BATCH_SIZE)
    args = ap.parse_args()
    try:
        dataset = InsuranceQADataset.load(config.PROCESSED_DIR)
        device = resolve_device(args.device)
        model = load_model(config.MODEL_DIR, device, config.SEMANTIC_MAX_SEQ_LENGTH)
    except Exception as exc:  # missing data / model
        print(f"ERROR: {exc}")
        return 1

    out = config.EMBEDDING_DIR
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "chunks"
    tmp.mkdir(exist_ok=True)
    texts = [clean_for_embedding(d.answer) for d in dataset.documents]
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    print(f"Encoding {len(texts)} answers on {device} (max_seq_length={model.max_seq_length}) ...")
    start = time.time()
    for i in range(n_chunks):
        path = tmp / f"chunk_{i:04d}.npy"
        if path.is_file():
            continue
        emb = model.encode(texts[i * CHUNK:(i + 1) * CHUNK], batch_size=args.batch_size,
                           convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        np.save(path, emb.astype(np.float32))
        done = min((i + 1) * CHUNK, len(texts))
        elapsed = time.time() - start
        print(f"  {done}/{len(texts)}  ({elapsed:.0f}s elapsed)", flush=True)

    embeddings = np.concatenate([np.load(tmp / f"chunk_{i:04d}.npy") for i in range(n_chunks)])
    assert embeddings.shape[0] == len(texts)
    np.save(out / EMBEDDINGS_FILE, embeddings)
    (out / META_FILE).write_text(json.dumps({
        "model_dir": config.MODEL_DIR.name, "documents": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]), "device": device,
        "max_seq_length": config.SEMANTIC_MAX_SEQ_LENGTH, "normalized": True,
        "doc_ids_sha256": doc_ids_fingerprint(dataset.doc_ids),
    }, indent=2), encoding="utf-8")
    for f in tmp.glob("chunk_*.npy"):
        f.unlink()
    tmp.rmdir()
    print(f"Saved {embeddings.shape} to {out / EMBEDDINGS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

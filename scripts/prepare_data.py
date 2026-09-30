"""Student 1 - decode, validate and preprocess the raw InsuranceQA files.

    python scripts/prepare_data.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data_loader import DatasetError
from src.dataset import build_processed


def main() -> int:
    start = time.time()
    try:
        meta = build_processed(config.RAW_DIR, config.PROCESSED_DIR, config)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(meta["validation"], indent=2))
    print(f"Processed data written to {config.PROCESSED_DIR} in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

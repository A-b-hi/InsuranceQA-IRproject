"""Shared fixtures. Integration tests use the REAL InsuranceQA data, indexes and embeddings;
if a prepared resource is missing the dependent tests are skipped with the command to create it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.dataset import InsuranceQADataset
from src.retrieval_base import ResourceMissingError
from src.search_engine import SearchEngine


@pytest.fixture(scope="session")
def dataset():
    try:
        return InsuranceQADataset.load(config.PROCESSED_DIR)
    except Exception as exc:
        pytest.skip(f"processed data missing ({exc}); run scripts/prepare_data.py")


@pytest.fixture(scope="session")
def engine():
    try:
        return SearchEngine.from_config(config)
    except ResourceMissingError as exc:
        pytest.skip(f"resources missing ({exc})")


@pytest.fixture(scope="session")
def real_question(dataset):
    return dataset.queries["test"][0]

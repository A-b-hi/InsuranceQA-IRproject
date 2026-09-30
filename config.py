"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = Path(os.getenv("INSURANCEQA_RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"
INDEX_DIR = BASE_DIR / "indexes"
EMBEDDING_DIR = BASE_DIR / "models" / "embeddings"
MODEL_DIR = Path(os.getenv("INSURANCEQA_MODEL_DIR", BASE_DIR / "models" / "InsuranceQA_Model"))

# --- InsuranceQA V2 files used by the project (github.com/shuzi/insuranceQA/tree/master/V2) ---
ANSWERS_FILE = "InsuranceQA.label2answer.raw.encoded.gz"
VOCABULARY_FILE = "vocabulary"
QUESTION_FILES = {
    "train": "InsuranceQA.question.anslabel.raw.100.pool.solr.train.encoded.gz",
    "valid": "InsuranceQA.question.anslabel.raw.100.pool.solr.valid.encoded.gz",
    "test": "InsuranceQA.question.anslabel.raw.100.pool.solr.test.encoded.gz",
}
COMBINED_QUESTIONS_FILE = "InsuranceQA.question.anslabel.raw.encoded.gz"  # used only for validation

# --- Retrieval parameters ---
BM25_K1 = float(os.getenv("BM25_K1", 1.5))
BM25_B = float(os.getenv("BM25_B", 0.75))
RRF_K = float(os.getenv("RRF_K", 60))
RRF_DEPTH = int(os.getenv("RRF_DEPTH", 1000))     # how deep each ranking list is before fusion
SEMANTIC_MAX_SEQ_LENGTH = 256
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", 0.30))   # cosine floor; real questions score >= 0.46
SEMANTIC_BATCH_SIZE = int(os.getenv("SEMANTIC_BATCH_SIZE", 64))
DEVICE = os.getenv("INSURANCEQA_DEVICE", "auto")   # auto | cpu | cuda | mps

# --- Application ---
DEFAULT_METHOD = "hybrid"
DEFAULT_TOP_K = 10
MAX_TOP_K = 50
COMPARE_TOP_K = 5
MAX_QUERY_CHARS = 500

# --- Evaluation ---
EVAL_SPLIT = os.getenv("EVAL_SPLIT", "test")
EVAL_DEPTH = int(os.getenv("EVAL_DEPTH", 1000))    # ranking depth used for MAP / recall / PR curve
EVAL_KS = (10, 100)

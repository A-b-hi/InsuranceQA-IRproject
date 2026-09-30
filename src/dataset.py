"""Student 1 - normalised dataset, processed-data pipeline and access layer.

`build_processed()` turns the raw InsuranceQA files into:
  data/processed/documents.jsonl     one record per answer (the retrieval corpus)
  data/processed/queries_<split>.jsonl  real questions with their official relevant answer ids
  data/processed/meta.json           counts, validation report, source-file checksums
`InsuranceQADataset.load()` reads them back for the search engine and evaluation.

Corpus design: the retrieval corpus is the ANSWER text (this is the standard InsuranceQA answer-
retrieval task). Each answer keeps the real question(s) it answers for display only; question text
is never indexed, so evaluating on a question can never leak its own text into the index.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import data_loader as dl
from .preprocessing import preprocess

SPLIT_ORDER = ("train", "valid", "test")


@dataclass
class Document:
    doc_id: int                       # official InsuranceQA answer label
    answer: str                       # original answer text (display)
    question: str = ""                # a real question this answer belongs to ("" if none is linked)
    questions: list[str] = field(default_factory=list)
    domain: str = ""                  # e.g. "life-insurance"
    tokens: list[str] = field(default_factory=list)   # processed tokens for TF-IDF / BM25


@dataclass
class Query:
    qid: str
    split: str
    domain: str
    question: str
    gold_ids: list[int]
    tokens: list[str] = field(default_factory=list)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_processed(raw_dir: Path, processed_dir: Path, config) -> dict:
    """Load + validate the real raw data, preprocess it and write the processed files."""
    names = [config.ANSWERS_FILE, config.VOCABULARY_FILE, config.COMBINED_QUESTIONS_FILE,
             *config.QUESTION_FILES.values()]
    files = dl.check_raw_files(raw_dir, names)
    decoder = dl.Decoder(files[config.VOCABULARY_FILE])
    answers = dl.load_answers(files[config.ANSWERS_FILE], decoder)
    splits = {s: dl.load_questions(files[f], s, decoder) for s, f in config.QUESTION_FILES.items()}
    combined = dl.load_questions(files[config.COMBINED_QUESTIONS_FILE], "all", decoder)
    report = dl.validate_dataset(answers, splits, combined, decoder)

    # Link every answer to the real question(s) it answers (display only).
    linked: dict[int, list[tuple[str, str]]] = {}
    for split in SPLIT_ORDER:
        for rec in splits[split]:
            for gold in rec.gold_ids:
                linked.setdefault(gold, []).append((rec.question, rec.domain))

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    with open(processed_dir / "documents.jsonl", "w", encoding="utf-8") as fh:
        for doc_id in sorted(answers):
            links = linked.get(doc_id, [])
            questions = list(dict.fromkeys(q for q, _ in links))[:5]
            n_linked += bool(questions)
            doc = Document(doc_id=doc_id, answer=answers[doc_id],
                           question=questions[0] if questions else "", questions=questions,
                           domain=links[0][1] if links else "",
                           tokens=preprocess(answers[doc_id]))
            fh.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
    for split, records in splits.items():
        with open(processed_dir / f"queries_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in records:
                q = Query(r.qid, r.split, r.domain, r.question, r.gold_ids, preprocess(r.question))
                fh.write(json.dumps(q.__dict__, ensure_ascii=False) + "\n")

    report["answers_linked_to_a_question"] = n_linked
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "InsuranceQA V2 (github.com/shuzi/insuranceQA), raw text decoded with the official vocabulary",
        "validation": report,
        "preprocessing": "clean -> lowercase -> alphanumeric tokens -> NLTK stop words -> Porter stemmer",
        "source_files": {n: {"file": p.name, "sha256": dl.sha256_of(p)} for n, p in files.items()},
    }
    (processed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class InsuranceQADataset:
    """Read access to the processed data."""

    def __init__(self, documents: list[Document], queries: dict[str, list[Query]], meta: dict):
        self.documents = documents
        self.queries = queries
        self.meta = meta
        self.doc_ids = [d.doc_id for d in documents]
        self.id_to_index = {d.doc_id: i for i, d in enumerate(documents)}
        self._question_index: dict[str, set[int]] = {}
        for split_queries in queries.values():
            for q in split_queries:
                self._question_index.setdefault(normalize_question(q.question), set()).update(q.gold_ids)

    def __len__(self) -> int:
        return len(self.documents)

    @classmethod
    def load(cls, processed_dir: Path) -> "InsuranceQADataset":
        processed_dir = Path(processed_dir)
        doc_path = processed_dir / "documents.jsonl"
        if not doc_path.is_file():
            raise dl.DatasetMissingError(
                f"Processed data not found in {processed_dir}. Run: python scripts/prepare_data.py")
        with open(doc_path, encoding="utf-8") as fh:
            documents = [Document(**json.loads(line)) for line in fh if line.strip()]
        queries = {}
        for split in SPLIT_ORDER:
            path = processed_dir / f"queries_{split}.jsonl"
            if path.is_file():
                with open(path, encoding="utf-8") as fh:
                    queries[split] = [Query(**json.loads(line)) for line in fh if line.strip()]
        meta_path = processed_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        return cls(documents, queries, meta)

    def official_relevant_ids(self, question: str) -> set[int]:
        """Official gold answer ids if `question` is one of the dataset's real questions."""
        return set(self._question_index.get(normalize_question(question), ()))

    def get(self, doc_id: int) -> Document | None:
        i = self.id_to_index.get(doc_id)
        return self.documents[i] if i is not None else None

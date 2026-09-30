"""Student 1 - InsuranceQA V2 loader and validator.

Real file layout (see data/raw/README.md):
  label2answer : <answer label> TAB <answer text>
  question file: <domain> TAB <question> TAB <ground-truth answer labels> [TAB <candidate pool>]
Text is stored as `idx_N` tokens; the official `vocabulary` file (idx_N TAB word) decodes it.
"""
from __future__ import annotations

import gzip
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

GITHUB_URL = "https://github.com/shuzi/insuranceQA/tree/master/V2"
_IDX_RE = re.compile(r"\bidx_\d+\b")


class DatasetError(Exception):
    """Base class for dataset problems."""


class DatasetMissingError(DatasetError):
    """Raised when required InsuranceQA files are not present."""


class DatasetFormatError(DatasetError):
    """Raised when a file does not follow the documented InsuranceQA format."""


@dataclass
class QuestionRecord:
    qid: str                 # e.g. "test-0000"
    split: str
    domain: str
    question: str
    gold_ids: list[int] = field(default_factory=list)   # official relevant answer labels


def resolve_file(raw_dir: Path, name: str) -> Path | None:
    """Return the file path, accepting either the .gz name or the decompressed name."""
    raw_dir = Path(raw_dir)
    for candidate in (name, name[:-3] if name.endswith(".gz") else name):
        if (raw_dir / candidate).is_file():
            return raw_dir / candidate
    return None


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def check_raw_files(raw_dir: Path, names: list[str]) -> dict[str, Path]:
    found, missing = {}, []
    for name in names:
        path = resolve_file(raw_dir, name)
        if path:
            found[name] = path
        else:
            missing.append(name)
    if missing:
        raise DatasetMissingError(
            f"InsuranceQA files missing from {raw_dir}:\n  - " + "\n  - ".join(missing)
            + f"\nDownload them from {GITHUB_URL} and place them in that folder."
        )
    return found


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class Decoder:
    """Maps `idx_N` tokens to words using the official vocabulary file."""

    def __init__(self, vocab_path: Path):
        self.vocab: dict[str, str] = {}
        with _open(Path(vocab_path)) as fh:
            for n, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                key, sep, word = line.partition("\t")
                if not sep:
                    raise DatasetFormatError(f"{vocab_path}:{n}: expected 'idx_N<TAB>word'")
                self.vocab[key] = word
        self.unknown_tokens = 0

    def decode(self, encoded: str) -> str:
        out = []
        for tok in encoded.split():
            word = self.vocab.get(tok)
            if word is None:
                self.unknown_tokens += 1
                word = tok
            out.append(word)
        return " ".join(out)


def load_answers(path: Path, decoder: Decoder) -> dict[int, str]:
    answers: dict[int, str] = {}
    with _open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            label, sep, text = line.partition("\t")
            if not sep or not label.isdigit():
                raise DatasetFormatError(f"{path}:{n}: expected '<label><TAB><answer>'")
            if int(label) in answers:
                raise DatasetFormatError(f"{path}:{n}: duplicate answer label {label}")
            answers[int(label)] = decoder.decode(text)
    return answers


def load_questions(path: Path, split: str, decoder: Decoder) -> list[QuestionRecord]:
    records = []
    with _open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                raise DatasetFormatError(f"{path}:{n}: expected at least 3 tab-separated columns")
            try:
                gold = [int(x) for x in cols[2].split()]
            except ValueError as exc:
                raise DatasetFormatError(f"{path}:{n}: ground-truth column is not numeric") from exc
            if not gold:
                raise DatasetFormatError(f"{path}:{n}: no ground-truth answer label")
            records.append(QuestionRecord(
                qid=f"{split}-{len(records):04d}", split=split, domain=cols[0],
                question=decoder.decode(cols[1]), gold_ids=gold,
            ))
    return records


def validate_dataset(answers: dict[int, str], splits: dict[str, list[QuestionRecord]],
                     combined: list[QuestionRecord] | None, decoder: Decoder) -> dict:
    """Consistency checks. Raises DatasetError on any real problem; returns a report."""
    problems = []
    if decoder.unknown_tokens:
        problems.append(f"{decoder.unknown_tokens} tokens were not found in the vocabulary")
    empty_answers = [k for k, v in answers.items() if not v.strip()]
    if empty_answers:
        problems.append(f"{len(empty_answers)} empty answers")
    if any(_IDX_RE.search(v) for v in answers.values()):
        problems.append("undecoded idx_ tokens remain in answers")
    n_gold = 0
    for split, records in splits.items():
        for r in records:
            if not r.question.strip():
                problems.append(f"empty question {r.qid}")
            missing = [g for g in r.gold_ids if g not in answers]
            if missing:
                problems.append(f"{r.qid}: gold ids not in answer file: {missing[:3]}")
            n_gold += len(r.gold_ids)
    if combined is not None:
        # The combined file lists the same records in a different order, so compare as multisets.
        key = lambda r: (r.domain, r.question, tuple(r.gold_ids))
        flat = Counter(key(r) for s in ("train", "valid", "test") for r in splits.get(s, []))
        whole = Counter(key(r) for r in combined)
        if flat != whole:
            problems.append(f"split files and combined file differ in {sum(((flat - whole) + (whole - flat)).values())} questions")
    if problems:
        raise DatasetError("Dataset validation failed: " + "; ".join(problems[:8]))
    return {
        "answers": len(answers),
        "questions": {s: len(r) for s, r in splits.items()},
        "gold_links": n_gold,
        "unknown_tokens": decoder.unknown_tokens,
        "cross_checked_with_combined_file": combined is not None,
    }

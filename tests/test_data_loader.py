"""Student 1 - dataset loading, validation, question-answer mapping (real InsuranceQA data)."""
import gzip
import json
import shutil

import pytest

import config
from src import data_loader as dl
from src.dataset import InsuranceQADataset, build_processed

RAW_NAMES = [config.ANSWERS_FILE, config.VOCABULARY_FILE, config.COMBINED_QUESTIONS_FILE, *config.QUESTION_FILES.values()]


@pytest.fixture(scope="module")
def raw():
    files = dl.check_raw_files(config.RAW_DIR, RAW_NAMES)
    dec = dl.Decoder(files[config.VOCABULARY_FILE])
    answers = dl.load_answers(files[config.ANSWERS_FILE], dec)
    splits = {s: dl.load_questions(files[f], s, dec) for s, f in config.QUESTION_FILES.items()}
    return files, dec, answers, splits


def test_real_dataset_sizes(raw):
    _, _, answers, splits = raw
    assert len(answers) == 27413
    assert {s: len(r) for s, r in splits.items()} == {"train": 12889, "valid": 2000, "test": 2000}


def test_text_is_decoded_to_real_words(raw):
    _, dec, answers, splits = raw
    assert dec.unknown_tokens == 0
    assert not any("idx_" in a for a in answers.values())
    assert not any("idx_" in r.question for rs in splits.values() for r in rs)
    assert all(a.strip() for a in answers.values())


def test_gold_labels_map_to_real_answers(raw):
    _, _, answers, splits = raw
    for records in splits.values():
        for r in records:
            assert r.gold_ids and all(g in answers for g in r.gold_ids)


def test_validation_report(raw):
    files, dec, answers, splits = raw
    combined = dl.load_questions(files[config.COMBINED_QUESTIONS_FILE], "all", dec)
    report = dl.validate_dataset(answers, splits, combined, dec)
    assert report["answers"] == 27413 and report["cross_checked_with_combined_file"]


def test_validation_detects_unknown_gold_id(raw):
    files, dec, answers, splits = raw
    bad = {"test": [dl.QuestionRecord("test-x", "test", "d", "a question", [-5])]}
    with pytest.raises(dl.DatasetError):
        dl.validate_dataset(answers, bad, None, dec)


def test_missing_files_are_reported_by_name(tmp_path):
    with pytest.raises(dl.DatasetMissingError) as exc:
        dl.check_raw_files(tmp_path, RAW_NAMES)
    assert config.ANSWERS_FILE in str(exc.value) and "github.com/shuzi/insuranceQA" in str(exc.value)


def test_malformed_files_raise_format_errors(tmp_path):
    shutil.copy(config.RAW_DIR / config.VOCABULARY_FILE, tmp_path / "vocabulary")
    dec = dl.Decoder(tmp_path / "vocabulary")
    bad_answers = tmp_path / "answers.txt"
    bad_answers.write_text("12\tidx_1\nnot-a-number\toops\n", encoding="utf-8")
    with pytest.raises(dl.DatasetFormatError):
        dl.load_answers(bad_answers, dec)
    bad_q = tmp_path / "q.txt"
    bad_q.write_text("life-insurance\tidx_1\n", encoding="utf-8")            # only 2 columns
    with pytest.raises(dl.DatasetFormatError):
        dl.load_questions(bad_q, "test", dec)
    bad_gold = tmp_path / "g.txt"
    bad_gold.write_text("life-insurance\tidx_1\tabc\n", encoding="utf-8")    # non-numeric gold
    with pytest.raises(dl.DatasetFormatError):
        dl.load_questions(bad_gold, "test", dec)


def test_gz_and_plain_files_are_both_accepted(tmp_path):
    src = config.RAW_DIR / config.ANSWERS_FILE
    with gzip.open(src, "rb") as fin, open(tmp_path / config.ANSWERS_FILE[:-3], "wb") as fout:
        fout.write(fin.read())
    assert dl.resolve_file(tmp_path, config.ANSWERS_FILE).name == config.ANSWERS_FILE[:-3]


def test_processed_documents_keep_original_text_and_question_mapping(dataset):
    assert len(dataset) == 27413
    doc = dataset.get(dataset.doc_ids[0])
    assert doc.answer and doc.tokens and doc.question
    # every answer is linked to a real question of the dataset, and the mapping is consistent
    q = dataset.queries["test"][0]
    for gold in q.gold_ids:
        assert q.question in dataset.get(gold).questions
    assert set(dataset.queries) == {"train", "valid", "test"}


def test_official_relevance_lookup(dataset, real_question):
    assert dataset.official_relevant_ids(real_question.question) >= set(real_question.gold_ids)
    assert dataset.official_relevant_ids("  " + real_question.question.upper() + "  ") >= set(real_question.gold_ids)
    assert dataset.official_relevant_ids("a question that is not in the dataset at all") == set()


def test_build_processed_end_to_end(tmp_path):
    meta = build_processed(config.RAW_DIR, tmp_path, config)
    assert meta["validation"]["answers"] == 27413
    assert meta["validation"]["answers_linked_to_a_question"] == 27413
    loaded = InsuranceQADataset.load(tmp_path)
    assert len(loaded) == 27413 and len(loaded.queries["test"]) == 2000
    assert all(len(v["sha256"]) == 64 for v in meta["source_files"].values())
    assert json.loads((tmp_path / "meta.json").read_text())["preprocessing"]


def test_loading_missing_processed_data(tmp_path):
    with pytest.raises(dl.DatasetMissingError):
        InsuranceQADataset.load(tmp_path)

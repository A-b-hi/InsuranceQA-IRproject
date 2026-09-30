# PolicyLens - FAQ Retrieval using InsuranceQA

An Information Retrieval system (not a chatbot). Given an insurance question it returns the most relevant
**existing expert answers** from the InsuranceQA V2 corpus (27,413 answers, 16,889 real questions).
Nothing is generated and no external API is used.

Methods compared: **TF-IDF**, **BM25**, **Sentence Transformer** (semantic) and **Hybrid RRF**
(Reciprocal Rank Fusion of the other three). Metrics: Precision@10, Recall, MAP, nDCG and a Precision-Recall graph.

## Dataset setup
The real dataset files used are already in `data/raw/` (InsuranceQA V2, github.com/shuzi/insuranceQA, research use,
cite Feng et al. 2015). If you ever need to restore them, put these files from the repo's `V2/` folder in `data/raw/`:
`InsuranceQA.label2answer.raw.encoded.gz`, `vocabulary`, `InsuranceQA.question.anslabel.raw.encoded.gz` and
`InsuranceQA.question.anslabel.raw.100.pool.solr.{train,valid,test}.encoded.gz`.
The Sentence Transformer model is in `models/InsuranceQA_Model/` (MiniLM-L6, 384-d; a general-purpose model, not fine-tuned on InsuranceQA).

## Installation
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare data, indexes and embeddings
Prepared outputs are already included, so you can skip straight to running the app. To rebuild them:
```
python scripts/prepare_data.py        # decode, validate and preprocess the raw files
python scripts/build_indexes.py       # TF-IDF and BM25 indexes
python scripts/build_embeddings.py    # Sentence Transformer embeddings (one-time; uses GPU if available)
```

## Evaluate
```
python scripts/evaluate.py            # writes data/evaluation/results.json, comparison.csv, pr_curve.png
```
Every method ranks all 27,413 answers for each of the 2,000 test questions; the official answer labels define relevance.

## Run the application
```
python app.py                         # http://127.0.0.1:5000
```
Tests: `pytest`

## Who owns what
| Student | Modules |
|---|---|
| 1 | `src/data_loader.py`, `src/preprocessing.py`, `src/dataset.py`, `scripts/prepare_data.py` |
| 2 | `src/tfidf_retriever.py`, `src/bm25_retriever.py`, `scripts/build_indexes.py` |
| 3 | `src/semantic_retriever.py`, `src/rrf.py`, `src/evaluation.py`, `scripts/build_embeddings.py`, `scripts/evaluate.py` |
| All | `src/search_engine.py`, `src/retrieval_base.py`, `app.py`, `templates/`, `static/`, `tests/` |

## Design notes
- The retrieval corpus is the answer text. Each answer is shown with the real question it belongs to; question text is never indexed, so a test question cannot match itself.
- Answers longer than 256 tokens are truncated for the Sentence Transformer; TF-IDF and BM25 use the full text.
- Semantic results below cosine 0.30 are dropped (`SEMANTIC_MIN_SCORE`), so nonsense queries return an empty state.
- Fonts load from Google Fonts; offline, the page falls back to system fonts.

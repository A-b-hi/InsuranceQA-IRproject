"""Integration - Flask application (real engine, real data)."""
import json
import re

import pytest

import config
from app import create_app
from helpers import make_cfg


@pytest.fixture(scope="module")
def client(engine):
    return create_app(engine).test_client()


def count_cards(html):
    return len(re.findall(r'<article class="card answer', html))


def test_home_page(client, engine):
    r = client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "PolicyLens" in html and "27,413" in html
    assert all(f'value="{m}"' in html for m in ("hybrid", "tfidf", "bm25", "semantic"))
    assert 'value="hybrid" checked' in html                                   # default method
    real = {q.question for q in engine.dataset.queries["test"]}
    chips = re.findall(r'data-example="([^"]+)"', html)
    assert chips and all(c.replace("&#39;", "'") in real for c in chips)      # examples are real questions


@pytest.mark.parametrize("method", ["hybrid", "tfidf", "bm25", "semantic"])
def test_search_page_shows_ranked_real_answers(client, engine, real_question, method):
    r = client.get("/search", query_string={"q": real_question.question, "method": method})
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and count_cards(html) == 10
    top = engine.search(real_question.question, method, 1)[0]
    assert f"Answer {top.document_id}" in html


def test_search_defaults_to_hybrid_with_rank_breakdown(client, real_question):
    html = client.get("/search", query_string={"q": real_question.question}).get_data(as_text=True)
    assert "Hybrid RRF results for" in html and "Rank in each list" in html


def test_official_answer_badge_for_dataset_questions(client, dataset):
    for q in dataset.queries["test"][:30]:
        html = client.get("/search", query_string={"q": q.question, "method": "semantic"}).get_data(as_text=True)
        if "Official answer" in html:
            return
    pytest.fail("no official-answer badge shown for 30 real questions")


def test_compare_view(client, real_question):
    html = client.get("/search", query_string={"q": real_question.question, "compare": "1"}).get_data(as_text=True)
    assert html.count('class="compare-col') == 4 and count_cards(html) == 20
    redirect = client.get("/compare", query_string={"q": "x"})
    assert redirect.status_code == 302 and "compare=1" in redirect.headers["Location"]


def test_empty_query_shows_validation_message(client):
    for q in ("", "   "):
        r = client.get("/search", query_string={"q": q})
        assert r.status_code == 400 and "Enter a question to search." in r.get_data(as_text=True)
    assert client.get("/search").status_code == 400


def test_overlong_and_unknown_method(client):
    assert client.get("/search", query_string={"q": "a " * 400}).status_code == 400
    r = client.get("/search", query_string={"q": "life insurance", "method": "nope"})
    assert r.status_code == 400 and "Unknown method" in r.get_data(as_text=True)


def test_malformed_k_falls_back_and_is_clamped(client):
    ok = client.get("/search", query_string={"q": "life insurance", "method": "bm25", "k": "abc"})
    assert count_cards(ok.get_data(as_text=True)) == 10
    big = client.get("/search", query_string={"q": "life insurance", "method": "bm25", "k": "9999"})
    assert count_cards(big.get_data(as_text=True)) == config.MAX_TOP_K


def test_no_results_empty_state(client):
    r = client.get("/search", query_string={"q": "zzzqqq xxyyzz", "method": "hybrid"})
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "No matching answers" in html and count_cards(html) == 0


def test_user_input_is_escaped(client):
    html = client.get("/search", query_string={"q": "<script>alert(1)</script> insurance"}).get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_api_search_and_errors(client, real_question):
    r = client.get("/api/search", query_string={"q": real_question.question, "method": "bm25", "k": 3})
    data = r.get_json()
    assert r.status_code == 200 and data["count"] == 3 and data["method"] == "bm25"
    assert set(data["results"][0]) >= {"document_id", "question", "answer", "score", "rank", "method"}
    assert client.get("/api/search?q=").status_code == 400
    assert client.get("/api/search?q=x&method=bad").status_code == 400
    assert set(client.get("/api/compare", query_string={"q": real_question.question}).get_json()["results"]) == \
        {"hybrid", "tfidf", "bm25", "semantic"}


def test_health(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok" and data["answers"] == 27413


def test_evaluation_dashboard_shows_real_saved_results(client):
    path = config.EVAL_DIR / "results.json"
    if not path.is_file():
        pytest.skip("run scripts/evaluate.py first")
    saved = json.loads(path.read_text())
    html = client.get("/evaluation").get_data(as_text=True)
    assert "Comparison table" in html and "Precision-Recall curve" in html and 'id="pr-chart"' in html
    sem = saved["methods"]["semantic"]["metrics"]
    assert f"{sem['MAP']:.4f}" in html and f"{sem['P@10']:.4f}" in html
    assert client.get("/evaluation/pr_curve.png").mimetype == "image/png"
    assert client.get("/api/evaluation").get_json()["config"]["queries"] == 2000


def test_evaluation_empty_state(engine, tmp_path):
    c = create_app(engine, make_cfg(EVAL_DIR=tmp_path)).test_client()
    assert "No evaluation results yet" in c.get("/evaluation").get_data(as_text=True)
    assert c.get("/api/evaluation").status_code == 404 and c.get("/evaluation/pr_curve.png").status_code == 404


def test_unknown_page_is_404(client):
    assert client.get("/nope").status_code == 404


def test_app_survives_missing_resources(tmp_path):
    app = create_app(cfg=make_cfg(PROCESSED_DIR=tmp_path))
    c = app.test_client()
    assert app.config["ENGINE_ERROR"]
    home = c.get("/")
    assert home.status_code == 200 and "Search is not ready" in home.get_data(as_text=True)
    page = c.get("/search", query_string={"q": "life insurance"})
    assert page.status_code == 503 and "scripts/prepare_data.py" in page.get_data(as_text=True)
    assert c.get("/api/search?q=x").status_code == 503 and c.get("/api/health").status_code == 503


def test_missing_embeddings_disable_methods_in_ui_and_return_503(tmp_path):
    c = create_app(cfg=make_cfg(EMBEDDING_DIR=tmp_path)).test_client()
    assert re.search(r'value="semantic"\s+disabled', c.get("/").get_data(as_text=True))
    r = c.get("/search", query_string={"q": "life insurance", "method": "hybrid"})
    assert r.status_code == 503 and "build_embeddings" in r.get_data(as_text=True)
    assert c.get("/search", query_string={"q": "life insurance", "method": "bm25"}).status_code == 200

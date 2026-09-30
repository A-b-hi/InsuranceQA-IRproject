"""Flask application (integration of all three students' modules).

    python app.py            # http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

import config
from src.retrieval_base import ResourceMissingError
from src.search_engine import (METHOD_INFO, METHODS, InvalidMethodError, InvalidQueryError,
                               SearchEngine)

EVAL_METRIC_ORDER = ["P@10", "Recall@10", "Recall@100", "MAP", "nDCG@10"]


def create_app(engine: SearchEngine | None = None, cfg=config) -> Flask:
    app = Flask(__name__)
    app.config["ENGINE_ERROR"] = None
    app.config["ENGINE"] = engine
    if engine is None:
        try:
            app.config["ENGINE"] = SearchEngine.from_config(cfg)
        except ResourceMissingError as exc:
            app.config["ENGINE_ERROR"] = str(exc)

    # ---- template helpers -------------------------------------------------------------
    @app.template_filter("domain_label")
    def domain_label(value: str) -> str:
        value = (value or "").replace("-", " ").strip()
        return value[:1].upper() + value[1:]

    @app.template_filter("score_fmt")
    def score_fmt(score: float, method: str) -> str:
        return f"{score:.2f}" if method == "bm25" else f"{score:.4f}" if method == "hybrid" else f"{score:.3f}"

    @app.template_filter("pct")
    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    @app.context_processor
    def inject_globals():
        eng = app.config["ENGINE"]
        return {"method_info": METHOD_INFO, "methods": METHODS,
                "availability": eng.available_methods() if eng else {},
                "stats": eng.stats() if eng else None,
                "engine_error": app.config["ENGINE_ERROR"]}

    def engine_or_none():
        return app.config["ENGINE"]

    def page_error(kind: str, message: str, status: int, **ctx):
        return render_template("results.html", error={"kind": kind, "message": message}, results=None,
                               compare=None, **ctx), status

    def form_context(query: str, method: str, compare: bool) -> dict:
        eng = engine_or_none()
        return {"query": query, "method": method if method in METHODS else config.DEFAULT_METHOD,
                "compare_mode": compare, "examples": eng.sample_queries() if eng else []}

    # ---- pages ------------------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template("index.html", **form_context("", config.DEFAULT_METHOD, False))

    @app.get("/search")
    def search():
        query = request.args.get("q", "")
        method = request.args.get("method", config.DEFAULT_METHOD)
        compare = request.args.get("compare") == "1"
        try:
            k = int(request.args.get("k", config.DEFAULT_TOP_K))
        except ValueError:
            k = config.DEFAULT_TOP_K
        k = max(1, min(k, config.MAX_TOP_K))
        ctx = form_context(query, method, compare)
        eng = engine_or_none()
        if eng is None:
            return page_error("unavailable", app.config["ENGINE_ERROR"], 503, mode="single", **ctx)
        if method not in METHODS:
            return page_error("invalid", f"Unknown method '{method}'. Choose one of: {', '.join(METHODS)}.",
                              400, mode="single", **ctx)
        try:
            clean = eng.validate_query(query)
        except InvalidQueryError as exc:
            return render_template("index.html", form_error=str(exc), **ctx), 400
        ctx["query"] = clean
        try:
            started = time.perf_counter()
            if compare:
                compared = eng.compare(clean)
                took = (time.perf_counter() - started) * 1000
                official = eng.official_relevant_ids(clean)
                shown = any(r.document_id in official for rs in compared.values() for r in rs)
                return render_template("results.html", error=None, results=None, compare=compared,
                                       mode="compare", took_ms=took, k=config.COMPARE_TOP_K,
                                       official_ids=official, official_shown=shown, **ctx)
            results = eng.search(clean, ctx["method"], k)
            took = (time.perf_counter() - started) * 1000
        except InvalidMethodError as exc:
            return page_error("invalid", str(exc), 400, mode="single", **ctx)
        except ResourceMissingError as exc:
            return page_error("unavailable", str(exc), 503, mode="single", **ctx)
        official = eng.official_relevant_ids(clean)
        return render_template("results.html", error=None, results=results, compare=None, mode="single",
                               took_ms=took, k=k, official_ids=official,
                               official_shown=any(r.document_id in official for r in results),
                               params=eng.params(ctx["method"]), **ctx)

    @app.get("/compare")
    def compare_redirect():
        return redirect(url_for("search", q=request.args.get("q", ""), compare=1))

    @app.get("/evaluation")
    def evaluation():
        data = load_evaluation(cfg)
        return render_template("evaluation.html", evaluation=data, metric_order=EVAL_METRIC_ORDER,
                               has_png=(Path(cfg.EVAL_DIR) / "pr_curve.png").is_file())

    @app.get("/evaluation/pr_curve.png")
    def pr_curve_png():
        path = Path(cfg.EVAL_DIR) / "pr_curve.png"
        if not path.is_file():
            return jsonify(error="Precision-Recall graph not generated yet. Run scripts/evaluate.py."), 404
        return send_file(path, mimetype="image/png", download_name="precision_recall_curve.png")

    # ---- JSON API ---------------------------------------------------------------------
    def api_guard(fn):
        def wrapper():
            eng = engine_or_none()
            if eng is None:
                return jsonify(error=app.config["ENGINE_ERROR"]), 503
            try:
                return fn(eng)
            except (InvalidQueryError, InvalidMethodError) as exc:
                return jsonify(error=str(exc)), 400
            except ResourceMissingError as exc:
                return jsonify(error=str(exc)), 503
        wrapper.__name__ = fn.__name__
        return wrapper

    @app.get("/api/search")
    @api_guard
    def api_search(eng):
        method = request.args.get("method", config.DEFAULT_METHOD)
        try:
            k = int(request.args.get("k", config.DEFAULT_TOP_K))
        except ValueError:
            k = config.DEFAULT_TOP_K
        started = time.perf_counter()
        results = eng.search(request.args.get("q", ""), method, k)
        return jsonify(query=request.args.get("q", "").strip(), method=method, count=len(results),
                       took_ms=round((time.perf_counter() - started) * 1000, 1),
                       results=[r.to_dict() for r in results])

    @app.get("/api/compare")
    @api_guard
    def api_compare(eng):
        out = eng.compare(request.args.get("q", ""))
        return jsonify(query=request.args.get("q", "").strip(),
                       results={m: [r.to_dict() for r in rs] for m, rs in out.items()})

    @app.get("/api/evaluation")
    def api_evaluation():
        data = load_evaluation(cfg)
        if data is None:
            return jsonify(error="No evaluation results yet. Run: python scripts/evaluate.py"), 404
        return jsonify(data)

    @app.get("/api/health")
    def api_health():
        eng = engine_or_none()
        if eng is None:
            return jsonify(status="unavailable", error=app.config["ENGINE_ERROR"]), 503
        return jsonify(status="ok", methods=eng.available_methods(), **eng.stats())

    @app.errorhandler(404)
    def not_found(_):
        return render_template("index.html", form_error="That page does not exist.",
                               **form_context("", config.DEFAULT_METHOD, False)), 404

    return app


def load_evaluation(cfg=config) -> dict | None:
    path = Path(cfg.EVAL_DIR) / "results.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)

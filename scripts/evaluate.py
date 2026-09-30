"""Student 3 - evaluate all four methods on the real InsuranceQA questions.

    python scripts/evaluate.py [--split test] [--limit N]

Every method ranks the whole answer corpus for each question; the dataset's official gold answer
labels define relevance. Writes results.json, comparison.csv and pr_curve.png to data/evaluation/.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.evaluation import evaluate_ranking
from src.retrieval_base import ResourceMissingError
from src.search_engine import METHOD_INFO, METHODS, SearchEngine

ORDER = ["tfidf", "bm25", "semantic", "hybrid"]
COLORS = {"hybrid": "#2456E8", "semantic": "#7A5AF8", "bm25": "#0E9AA7", "tfidf": "#E08A00"}


def plot_pr_curve(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    for m in ORDER:
        pts = results["methods"][m]["pr_curve"]
        ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], marker="o", ms=4,
                lw=3 if m == "hybrid" else 2, color=COLORS[m], label=results["methods"][m]["label"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.0); ax.set_ylim(bottom=0)
    ax.set_title(f"Precision-Recall curve - InsuranceQA {results['config']['split']} "
                 f"({results['config']['queries']} questions)", loc="left", color="#0A1B3D")
    ax.grid(color="#DCE3F0"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=config.EVAL_SPLIT, choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions (quick check)")
    ap.add_argument("--no-save", action="store_true", help="print only, do not overwrite saved results")
    args = ap.parse_args()
    try:
        engine = SearchEngine.from_config(config)
    except ResourceMissingError as exc:
        print(f"ERROR: {exc}")
        return 1
    if engine.unavailable:
        print("ERROR: all four methods are needed for evaluation:", engine.unavailable)
        return 1

    queries = engine.dataset.queries[args.split][:args.limit]
    texts = [q.question for q in queries]
    relevant = [{engine.dataset.id_to_index[g] for g in q.gold_ids} for q in queries]
    print(f"Evaluating {len(queries)} '{args.split}' questions against {len(engine.dataset)} answers ...")

    results = {"methods": {}}
    for m in ORDER:
        start = time.time()
        ranked = [rl.doc_idx.tolist() for rl in engine.rank_batch(m, texts, config.EVAL_DEPTH)]
        out = evaluate_ranking(ranked, relevant, ks=config.EVAL_KS)
        out["label"] = METHOD_INFO[m]["label"]
        out["seconds"] = round(time.time() - start, 1)
        results["methods"][m] = out
        mt = out["metrics"]
        print(f"  {out['label']:11s} P@10={mt['P@10']:.4f} R@10={mt['Recall@10']:.4f} R@100={mt['Recall@100']:.4f} "
              f"MAP={mt['MAP']:.4f} nDCG@10={mt['nDCG@10']:.4f}  ({out['seconds']}s)")

    keys = list(results["methods"]["tfidf"]["metrics"])
    results["best"] = {k: max(ORDER, key=lambda m: results["methods"][m]["metrics"][k]) for k in keys}
    results["config"] = {
        "split": args.split, "queries": len(queries), "corpus_size": len(engine.dataset),
        "depth": config.EVAL_DEPTH, "ks": list(config.EVAL_KS),
        "avg_relevant": sum(len(r) for r in relevant) / len(relevant),
        "bm25": {"k1": engine.retrievers["bm25"].k1, "b": engine.retrievers["bm25"].b},
        "rrf": {"k": engine.rrf_k, "depth": engine.rrf_depth},
        "model": config.MODEL_DIR.name + " (MiniLM-L6, 384-d)",
        "device": str(engine.retrievers["semantic"].model.device),
        "semantic_min_score": engine.retrievers["semantic"].min_score,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.no_save:
        return 0
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(config.EVAL_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", *keys])
        for m in ORDER:
            w.writerow([results["methods"][m]["label"], *[f"{results['methods'][m]['metrics'][k]:.4f}" for k in keys]])
    plot_pr_curve(results, config.EVAL_DIR / "pr_curve.png")
    print(f"Saved results.json, comparison.csv and pr_curve.png to {config.EVAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

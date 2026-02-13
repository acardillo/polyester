#!/usr/bin/env python3
"""
Run benchmarks across all stores and generate report.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress non-critical warnings and HF Hub noise (must be before store imports)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

from src.adapters import PythonDocsAdapter
from src.stores import VectorStore, GraphStore, BM25Store, HybridStore

DATA_PATH = Path("data/python_docs.json")
BENCHMARKS_PATH = Path("data/python_benchmarks.json")


@dataclass
class QueryResult:
    """Result of one query against one store."""
    query_id: str
    store_name: str
    retrieval_time_ms: float
    retrieved_ids: list[str]
    relevant_ids: list[str]
    hits: int  # Number of relevant ids retrieved
    precision_at_5: float
    recall_at_5: float


@dataclass
class StoreMetrics:
    """Aggregate metrics for one store."""
    store_name: str
    avg_retrieval_time_ms: float = 0.0
    median_retrieval_time_ms: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_recall_at_5: float = 0.0
    semantic_precision: float = 0.0
    semantic_recall: float = 0.0
    semantic_avg_time_ms: float = 0.0
    keyword_precision: float = 0.0
    keyword_recall: float = 0.0
    keyword_avg_time_ms: float = 0.0
    structural_precision: float = 0.0
    structural_recall: float = 0.0
    structural_avg_time_ms: float = 0.0


def load_documents() -> list:
    """Load documents via PythonDocsAdapter."""
    adapter = PythonDocsAdapter(data_path=str(DATA_PATH))
    return adapter.load_documents()


def load_benchmarks() -> dict[str, Any]:
    """Load benchmark queries from JSON."""
    with open(BENCHMARKS_PATH) as f:
        return json.load(f)


def run_one_query(store, query_text: str, relevant_ids: list[str]) -> tuple[list[str], float]:
    """Run a single query, return (retrieved_ids, time_ms)."""
    start = time.perf_counter()
    results = store.query(query_text, n_results=5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = [doc.id for doc in results]
    return retrieved_ids, elapsed_ms


def compute_precision_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> tuple[float, float, int]:
    """Compute precision@k, recall@k, and hit count."""
    retrieved_set = set(retrieved_ids[:5])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)
    precision = hits / 5 if 5 else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return precision, recall, hits


def run_benchmarks(stores: dict[str, Any], queries: list[dict]) -> dict[str, list[QueryResult]]:
    """Run all queries against all stores; return results[store_name] = [QueryResult, ...]."""
    results_by_store: dict[str, list[QueryResult]] = {name: [] for name in stores}

    for q in queries:
        query_id = q["id"]
        query_text = q["query"]
        relevant_ids = q.get("relevant_ids", [])

        for store_name, store in stores.items():
            retrieved_ids, time_ms = run_one_query(store, query_text, relevant_ids)
            precision, recall, hits = compute_precision_recall(retrieved_ids, relevant_ids)
            results_by_store[store_name].append(
                QueryResult(
                    query_id=query_id,
                    store_name=store_name,
                    retrieval_time_ms=time_ms,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    hits=hits,
                    precision_at_5=precision,
                    recall_at_5=recall,
                )
            )

    return results_by_store


def compute_store_metrics(
    store_name: str,
    query_results: list[QueryResult],
    queries_by_id: dict[str, dict],
) -> StoreMetrics:
    """Aggregate QueryResults and category info into StoreMetrics."""
    if not query_results:
        return StoreMetrics(store_name=store_name)

    times = [r.retrieval_time_ms for r in query_results]
    times_sorted = sorted(times)
    n = len(times_sorted)
    median_ms = times_sorted[n // 2] if n else 0.0

    avg_time = sum(times) / len(times)
    avg_precision = sum(r.precision_at_5 for r in query_results) / len(query_results)
    avg_recall = sum(r.recall_at_5 for r in query_results) / len(query_results)

    by_category: dict[str, list[QueryResult]] = {"semantic": [], "keyword": [], "structural": []}
    for r in query_results:
        cat = queries_by_id.get(r.query_id, {}).get("category", "semantic")
        if cat in by_category:
            by_category[cat].append(r)

    def cat_metrics(cat: str) -> tuple[float, float, float]:
        lst = by_category.get(cat, [])
        if not lst:
            return 0.0, 0.0, 0.0
        prec = sum(r.precision_at_5 for r in lst) / len(lst)
        rec = sum(r.recall_at_5 for r in lst) / len(lst)
        t = sum(r.retrieval_time_ms for r in lst) / len(lst)
        return prec, rec, t

    sem_prec, sem_rec, sem_time = cat_metrics("semantic")
    kw_prec, kw_rec, kw_time = cat_metrics("keyword")
    struct_prec, struct_rec, struct_time = cat_metrics("structural")

    return StoreMetrics(
        store_name=store_name,
        avg_retrieval_time_ms=avg_time,
        median_retrieval_time_ms=median_ms,
        avg_precision_at_5=avg_precision,
        avg_recall_at_5=avg_recall,
        semantic_precision=sem_prec,
        semantic_recall=sem_rec,
        semantic_avg_time_ms=sem_time,
        keyword_precision=kw_prec,
        keyword_recall=kw_rec,
        keyword_avg_time_ms=kw_time,
        structural_precision=struct_prec,
        structural_recall=struct_rec,
        structural_avg_time_ms=struct_time,
    )


def generate_comparison_report(
    metrics_by_store: dict[str, StoreMetrics],
    results_by_store: dict[str, list[QueryResult]],
    queries: list[dict],
) -> dict[str, Any]:
    """Build comparison report: fastest, most accurate, best per category."""
    if not metrics_by_store:
        return {"summary": {}, "category_analysis": {}}

    # Fastest = lowest avg retrieval time
    fastest = min(metrics_by_store.keys(), key=lambda s: metrics_by_store[s].avg_retrieval_time_ms)
    # Most accurate = highest avg precision@5
    most_accurate = max(metrics_by_store.keys(), key=lambda s: metrics_by_store[s].avg_precision_at_5)

    # Per-query winner (which store had best precision for that query)
    query_id_to_category = {q["id"]: q.get("category", "semantic") for q in queries}
    category_counts: dict[str, dict[str, int]] = {
        "semantic": {s: 0 for s in metrics_by_store},
        "keyword": {s: 0 for s in metrics_by_store},
        "structural": {s: 0 for s in metrics_by_store},
    }
    n_queries = len(queries)
    for i in range(n_queries):
        qid = queries[i]["id"]
        cat = query_id_to_category.get(qid, "semantic")
        best_store = max(
            metrics_by_store.keys(),
            key=lambda s: results_by_store[s][i].precision_at_5,
        )
        category_counts[cat][best_store] = category_counts[cat].get(best_store, 0) + 1

    category_analysis = {}
    for cat in ["semantic", "keyword", "structural"]:
        counts = category_counts[cat]
        best = max(counts.keys(), key=lambda s: counts[s])
        category_analysis[cat] = {"best_store": best, "counts": counts}

    return {
        "summary": {
            "fastest_store": fastest,
            "most_accurate_store": most_accurate,
        },
        "category_analysis": category_analysis,
    }


def write_results_markdown(
    metrics_by_store: dict[str, StoreMetrics],
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write all results to RESULTS.md: key findings first, then separate metric tables."""
    lines = ["# Polystore Benchmark Results", ""]

    # Key findings first
    lines.extend(["## Key findings", ""])
    lines.append("| Finding | Store |")
    lines.append("| --- | --- |")
    lines.append(f"| Fastest | {report['summary']['fastest_store']} |")
    lines.append(f"| Most accurate | {report['summary']['most_accurate_store']} |")
    for category, data in report["category_analysis"].items():
        best = data["best_store"]
        total = sum(data["counts"].values())
        wins = data["counts"][best]
        lines.append(f"| {category.capitalize()} queries (wins) | {best} ({wins}/{total}) |")

    # Average (overall)
    lines.extend(["", "## Average", ""])
    lines.append("| Store | Avg (ms) | Median (ms) | Precision@5 | Recall@5 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for store_name, m in metrics_by_store.items():
        row = [
            store_name,
            f"{m.avg_retrieval_time_ms:.2f}",
            f"{m.median_retrieval_time_ms:.2f}",
            f"{m.avg_precision_at_5:.2%}",
            f"{m.avg_recall_at_5:.2%}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Semantic
    lines.extend(["", "## Semantic", ""])
    lines.append("| Store | Precision | Recall | Avg (ms) |")
    lines.append("| --- | --- | --- | --- |")
    for store_name, m in metrics_by_store.items():
        row = [
            store_name,
            f"{m.semantic_precision:.2%}",
            f"{m.semantic_recall:.2%}",
            f"{m.semantic_avg_time_ms:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Keyword
    lines.extend(["", "## Keyword", ""])
    lines.append("| Store | Precision | Recall | Avg (ms) |")
    lines.append("| --- | --- | --- | --- |")
    for store_name, m in metrics_by_store.items():
        row = [
            store_name,
            f"{m.keyword_precision:.2%}",
            f"{m.keyword_recall:.2%}",
            f"{m.keyword_avg_time_ms:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Structural
    lines.extend(["", "## Structural", ""])
    lines.append("| Store | Precision | Recall | Avg (ms) |")
    lines.append("| --- | --- | --- | --- |")
    for store_name, m in metrics_by_store.items():
        row = [
            store_name,
            f"{m.structural_precision:.2%}",
            f"{m.structural_recall:.2%}",
            f"{m.structural_avg_time_ms:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("# POLYESTER BENCHMARK SUITE")

    print("\nLoading documents...")
    documents = load_documents()
    print(f"  ✓ {len(documents)} documents loaded from {DATA_PATH.name}")

    print("\nLoading benchmark queries...")
    benchmarks = load_benchmarks()
    queries = benchmarks.get("queries", [])
    print(f"  ✓ {len(queries)} queries loaded from {BENCHMARKS_PATH.name}")

    print("\nInitializing and indexing stores...")
    stores = {
        "vector": VectorStore(collection_name="benchmark_vector"),
        "graph": GraphStore(),
        "bm25": BM25Store(),
        "hybrid": HybridStore(collection_name="benchmark_hybrid"),
    }
    for name, store in stores.items():
        store.index(documents)
        print(f"  ✓ {name.capitalize()} Store indexed: {store.size()} documents")

    print("\nRunning benchmarks... ", end="", flush=True)
    results_by_store = run_benchmarks(stores, queries)
    print("Done.")

    queries_by_id = {q["id"]: q for q in queries}
    metrics_by_store = {
        name: compute_store_metrics(name, results, queries_by_id)
        for name, results in results_by_store.items()
    }

    for store_name in stores:
        m = metrics_by_store[store_name]
        print(f"\n{store_name.upper()} STORE:")
        print(f"  Speed:     {m.avg_retrieval_time_ms:.2f}ms avg, {m.median_retrieval_time_ms:.2f}ms median")
        print(f"  Accuracy:  {m.avg_precision_at_5:.2%} precision, {m.avg_recall_at_5:.2%} recall")
        print("  By Category:")
        print(f"    Semantic:    {m.semantic_precision:.2%} precision, {m.semantic_recall:.2%} recall, {m.semantic_avg_time_ms:.2f}ms")
        print(f"    Keyword:     {m.keyword_precision:.2%} precision, {m.keyword_recall:.2%} recall, {m.keyword_avg_time_ms:.2f}ms")
        print(f"    Structural:  {m.structural_precision:.2%} precision, {m.structural_recall:.2%} recall, {m.structural_avg_time_ms:.2f}ms")

    report = generate_comparison_report(metrics_by_store, results_by_store, queries)

    print("\nKEY FINDINGS:")
    print(f"  Fastest store:      {report['summary']['fastest_store']}")
    print(f"  Most accurate:      {report['summary']['most_accurate_store']}")
    for category, data in report["category_analysis"].items():
        best = data["best_store"]
        counts = data["counts"]
        total = sum(counts.values())
        print(f"  {category.capitalize()} queries: {best} wins ({counts[best]}/{total})")

    project_root = Path(__file__).resolve().parent.parent
    md_path = project_root / "RESULTS.md"
    write_results_markdown(metrics_by_store, report, md_path)

    print(f"\nResults table saved to: {md_path}")


if __name__ == "__main__":
    main()

"""``hybrid-rag-bench``: download a BEIR dataset, index it, run every system, write the report.

uv run --group bench hybrid-rag-bench                 # full SciFact run
uv run --group bench hybrid-rag-bench --report-only   # regenerate docs from the saved run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from hybrid_rag.config import Settings
from hybrid_rag.evaluation.beir import BeirDataset, download_beir, load_beir
from hybrid_rag.evaluation.report import render_chart, render_markdown, summarize
from hybrid_rag.evaluation.runner import (
    DEFAULT_SYSTEMS,
    BenchmarkResult,
    SystemSpec,
    doc_id_map,
    index_corpus,
    run_system,
)
from hybrid_rag.indexing import Indexer, rebuild_sparse
from hybrid_rag.ingestion.pipeline import IngestionPipeline
from hybrid_rag.retrieval.dense import QdrantDenseIndex
from hybrid_rag.retrieval.embeddings import FastEmbedEmbedder
from hybrid_rag.retrieval.hybrid import HybridRetriever
from hybrid_rag.retrieval.rerank import FlashRankReranker, Reranker
from hybrid_rag.retrieval.sparse import BM25Index

FETCH_K = 20  # chunks requested per query; collapsed to TOP documents
TOP = 10


def _progress(label: str) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            print(f"  {label}: {done}/{total}", file=sys.stderr, flush=True)

    return report


def _run(args: argparse.Namespace) -> BenchmarkResult:
    settings = Settings()
    data_dir, workdir = Path(args.data_dir), Path(args.workdir) / args.dataset

    print(f"dataset {args.dataset}: checking {data_dir}", file=sys.stderr)
    _, sha = download_beir(args.dataset, data_dir)
    dataset: BeirDataset = load_beir(data_dir, args.dataset)
    if args.limit_queries:
        dataset = dataset.limit_queries(args.limit_queries)
    print(f"{len(dataset.corpus):,} documents, {len(dataset.queries)} queries", file=sys.stderr)

    specs = [s for s in DEFAULT_SYSTEMS if not args.systems or s.name in args.systems]
    if args.rebuild:
        shutil.rmtree(workdir, ignore_errors=True)
    marker = workdir / "indexed.ok"
    if not marker.exists():  # an interrupted indexing run must not be reused
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    embedder = FastEmbedEmbedder(
        settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        cache_dir=settings.embedding.cache_dir,
    )
    client = QdrantClient(path=str(workdir / "qdrant"))
    try:
        dense = QdrantDenseIndex(client, "bench", embedder.dimension)
        sparse = BM25Index(k1=settings.bm25.k1, b=settings.bm25.b)
        if marker.exists():
            print(f"reusing index ({rebuild_sparse(dense, sparse):,} chunks)", file=sys.stderr)
        else:
            pipeline = IngestionPipeline.default(embedder, settings.chunking)
            chunks = index_corpus(
                dataset,
                pipeline,
                Indexer(embedder, dense, sparse, batch_size=settings.embedding.batch_size),
                progress=_progress("indexing"),
            )
            marker.write_text(str(chunks), encoding="utf-8")
            print(f"indexed {chunks:,} chunks", file=sys.stderr)

        rerankers: dict[str, Reranker] = {}

        def retriever_for(spec: SystemSpec) -> HybridRetriever:
            reranker = None
            if spec.reranker:
                reranker = rerankers.setdefault(
                    spec.reranker,
                    FlashRankReranker(spec.reranker, cache_dir=settings.rerank.cache_dir),
                )
            return HybridRetriever(embedder, dense, sparse, reranker, settings.retrieval)

        mapping = doc_id_map(dataset)
        results = []
        for spec in specs:
            print(f"running {spec.label}", file=sys.stderr)
            results.append(
                run_system(
                    dataset,
                    retriever_for(spec),
                    spec,
                    mapping,
                    fetch_k=FETCH_K,
                    top=TOP,
                    progress=_progress(spec.name),
                )
            )
    finally:
        client.close()

    return BenchmarkResult(
        dataset=args.dataset,
        n_docs=len(dataset.corpus),
        n_queries=len(dataset.queries),
        embedding_model=settings.embedding.model,
        dataset_sha256=sha,
        top_k=TOP,
        qrels=dataset.qrels,
        systems=results,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hybrid-rag-bench", description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", default="data/beir", help="where BEIR archives are unpacked")
    parser.add_argument("--workdir", default="data/bench", help="index cache (gitignored)")
    parser.add_argument("--out-dir", default="docs", help="where benchmark.{json,md,png} go")
    parser.add_argument("--limit-queries", type=int, default=0, help="smoke test: first N queries")
    parser.add_argument("--systems", nargs="*", help="subset of system names to run")
    parser.add_argument("--baseline", default="dense", help="system the paired deltas compare to")
    parser.add_argument("--rebuild", action="store_true", help="discard the cached index")
    parser.add_argument("--report-only", action="store_true", help="skip the run; reuse the JSON")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "benchmark.json"
    if args.report_only:
        result = BenchmarkResult.model_validate_json(results_path.read_text(encoding="utf-8"))
    else:
        result = _run(args)
        results_path.write_text(result.model_dump_json(), encoding="utf-8")

    summary = summarize(result, baseline=args.baseline)
    (out / "benchmark.md").write_text(render_markdown(result, summary), encoding="utf-8")
    try:
        render_chart(result, summary, out / "benchmark.png")
    except ImportError as exc:
        print(f"chart skipped: {exc}", file=sys.stderr)
    print(f"wrote {out / 'benchmark.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

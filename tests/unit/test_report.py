from pathlib import Path

import pytest

from hybrid_rag.evaluation.report import render_chart, render_markdown, summarize
from hybrid_rag.evaluation.runner import BenchmarkResult, SystemResult, SystemSpec

QRELS = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 1}, "q4": {"d": 1}}


def system(name: str, ranking: dict[str, list[str]], latency: float, **kw: str) -> SystemResult:
    spec = SystemSpec(name=name, label=kw.get("label", name), mode="dense")
    return SystemResult(spec=spec, ranking=ranking, latency_ms=dict.fromkeys(QRELS, latency))


PERFECT = {"q1": ["a"], "q2": ["b"], "q3": ["c"], "q4": ["d"]}
WEAK = {"q1": ["x", "a"], "q2": ["x", "b"], "q3": ["c"], "q4": ["x"]}
RESULT = BenchmarkResult(
    dataset="mini",
    n_docs=10,
    n_queries=4,
    embedding_model="fake-embedder",
    dataset_sha256="abc123",
    top_k=10,
    qrels=QRELS,
    systems=[
        system("dense", WEAK, 10.0, label="Naive RAG"),
        system("hybrid", PERFECT, 20.0, label="Hybrid"),
    ],
)


def test_summary_computes_metrics_confidence_intervals_and_latency_percentiles() -> None:
    summary = summarize(RESULT, baseline="dense", n_resamples=200)

    naive, hybrid = summary.systems
    assert naive.metrics["MRR@10"] == pytest.approx((0.5 + 0.5 + 1 + 0) / 4)
    assert hybrid.metrics["MRR@10"] == 1.0
    assert naive.ci["MRR@10"][0] <= naive.metrics["MRR@10"] <= naive.ci["MRR@10"][1]
    assert hybrid.ci["nDCG@10"] == pytest.approx((1.0, 1.0))
    assert (hybrid.latency_p50_ms, hybrid.latency_p95_ms) == (20.0, 20.0)


def test_paired_deltas_are_reported_against_the_baseline_only() -> None:
    summary = summarize(RESULT, baseline="dense", n_resamples=200)

    [delta] = summary.deltas
    assert delta.system == "hybrid"
    assert delta.metrics["MRR@10"][0] == pytest.approx(0.5)
    lo, hi = delta.metrics["MRR@10"][1:]
    assert lo <= 0.5 <= hi


def test_unknown_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="baseline"):
        summarize(RESULT, baseline="nope")


def test_markdown_has_one_row_per_system_with_the_best_value_in_bold() -> None:
    summary = summarize(RESULT, baseline="dense", n_resamples=200)

    md = render_markdown(RESULT, summary)

    assert "| Naive RAG |" in md
    assert "| Hybrid |" in md
    assert "**1.000**" in md  # hybrid wins MRR@10
    [naive_row] = [line for line in md.splitlines() if line.startswith("| Naive RAG |")]
    assert "| 0.500 |" in naive_row  # naive MRR@10 = (0.5 + 0.5 + 1 + 0) / 4
    assert "mini" in md
    assert "abc123" in md  # provenance of the dataset archive
    assert "Paired difference" in md


def test_chart_is_written_as_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "bench.png"

    render_chart(RESULT, summarize(RESULT, baseline="dense", n_resamples=200), out)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

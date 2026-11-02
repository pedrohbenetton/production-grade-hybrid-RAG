from pathlib import Path

import pytest

from hybrid_rag.evaluation.cli import main
from hybrid_rag.evaluation.runner import BenchmarkResult, SystemResult, SystemSpec

QRELS = {"q1": {"a": 1}, "q2": {"b": 1}}


def write_result(out: Path) -> None:
    def system(name: str, ranking: dict[str, list[str]]) -> SystemResult:
        return SystemResult(
            spec=SystemSpec(name=name, label=name, mode="dense"),
            ranking=ranking,
            latency_ms={"q1": 5.0, "q2": 7.0},
        )

    result = BenchmarkResult(
        dataset="mini",
        n_docs=4,
        n_queries=2,
        embedding_model="fake",
        dataset_sha256=None,
        top_k=10,
        qrels=QRELS,
        systems=[
            system("dense", {"q1": ["x", "a"], "q2": ["b"]}),
            system("hybrid", {"q1": ["a"], "q2": ["b"]}),
        ],
    )
    (out / "benchmark.json").write_text(result.model_dump_json(), encoding="utf-8")


def test_report_only_regenerates_the_markdown_from_the_saved_run(tmp_path: Path) -> None:
    write_result(tmp_path)

    assert main(["--report-only", "--out-dir", str(tmp_path)]) == 0

    markdown = (tmp_path / "benchmark.md").read_text(encoding="utf-8")
    assert "| hybrid |" in markdown
    assert "Paired difference vs dense" in markdown


def test_report_only_also_draws_the_chart_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    write_result(tmp_path)

    main(["--report-only", "--out-dir", str(tmp_path)])

    assert (tmp_path / "benchmark.png").read_bytes()[:4] == b"\x89PNG"


def test_report_only_without_a_saved_run_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["--report-only", "--out-dir", str(tmp_path)])


def test_unknown_baseline_is_rejected(tmp_path: Path) -> None:
    write_result(tmp_path)
    with pytest.raises(ValueError, match="baseline"):
        main(["--report-only", "--out-dir", str(tmp_path), "--baseline", "nope"])


def test_custom_dataset_prefix(tmp_path: Path) -> None:
    write_result(tmp_path)
    (tmp_path / "benchmark.json").rename(tmp_path / "benchmark_nfcorpus.json")
    assert (
        main(
            [
                "--report-only",
                "--out-dir",
                str(tmp_path),
                "--dataset",
                "nfcorpus",
            ]
        )
        == 0
    )
    assert (tmp_path / "benchmark_nfcorpus.md").exists()

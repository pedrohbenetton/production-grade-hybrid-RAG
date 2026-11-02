from pathlib import Path

import pytest
from conftest import FakeEmbedder

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion.pipeline import IngestionPipeline, diff_chunks
from hybrid_rag.ingestion.sources import UnsupportedFormatError, load_file

pytestmark = pytest.mark.integration


@pytest.fixture
def pipeline(fake_embedder: FakeEmbedder) -> IngestionPipeline:
    return IngestionPipeline.default(
        embedder=fake_embedder, settings=ChunkingSettings(min_tokens=5, max_tokens=60)
    )


def test_load_file_detects_mime_and_uri(fixtures_dir: Path) -> None:
    raw = load_file(fixtures_dir / "sample.html")
    assert raw.mime_type == "text/html"
    assert raw.source_uri.startswith("file:")


def test_load_file_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedFormatError):
        load_file(path)


def test_markdown_end_to_end(pipeline: IngestionPipeline, fixtures_dir: Path) -> None:
    result = pipeline.process(load_file(fixtures_dir / "sample.md"))

    assert result.tree.title == "Hybrid Search Guide"
    assert all(c.embed_text.startswith("Hybrid Search Guide") for c in result.chunks)
    [code] = [c for c in result.chunks if "def cosine" in c.text]
    assert code.section_path == ("Dense Retrieval",)


def test_html_end_to_end(pipeline: IngestionPipeline, fixtures_dir: Path) -> None:
    result = pipeline.process(load_file(fixtures_dir / "sample.html"))

    joined = "\n".join(c.text for c in result.chunks)
    assert "alert(1)" not in joined
    assert "Copyright" not in joined
    [bench] = [c for c in result.chunks if "Qdrant" in c.text]
    assert bench.section_path == ("Benchmarks",)


def test_pdf_end_to_end(pipeline: IngestionPipeline, sample_pdf: Path) -> None:
    result = pipeline.process(load_file(sample_pdf))

    assert result.tree.title == "Retrieval Systems Handbook"
    assert [s.heading for s in result.tree.root.children] == [
        "1. Dense Retrieval",
        "2. Sparse Retrieval",
    ]
    joined = "\n".join(c.text for c in result.chunks)
    assert "Confidential draft" not in joined
    [bm25] = [c for c in result.chunks if "BM25" in c.text]
    assert bm25.page == 2
    assert bm25.section_path == ("2. Sparse Retrieval",)


def test_reprocessing_is_deterministic_and_diff_is_noop(
    pipeline: IngestionPipeline, fixtures_dir: Path
) -> None:
    raw = load_file(fixtures_dir / "sample.md")
    first = pipeline.process(raw)
    second = pipeline.process(raw)

    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert diff_chunks({c.chunk_id for c in first.chunks}, second.chunks).is_noop

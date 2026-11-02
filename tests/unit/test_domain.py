import hashlib

import pytest
from pydantic import ValidationError

from hybrid_rag.config import ChunkingSettings, Settings
from hybrid_rag.domain.ids import make_chunk_id, make_doc_id
from hybrid_rag.domain.models import Block, RawDocument


def test_raw_document_sha256_is_derived_from_content() -> None:
    raw = RawDocument(source_uri="file:///a.md", mime_type="text/markdown", content=b"hello")
    assert raw.sha256 == hashlib.sha256(b"hello").hexdigest()


def test_models_are_immutable() -> None:
    block = Block(kind="paragraph", text="Some text.")
    with pytest.raises(ValidationError):
        block.text = "changed"  # type: ignore[misc]


def test_block_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        Block(kind="paragraph", text="   ")


def test_raw_document_rejects_unknown_mime() -> None:
    with pytest.raises(ValidationError):
        RawDocument(source_uri="file:///a.doc", mime_type="application/msword", content=b"")


def test_doc_id_is_stable_and_uri_scoped() -> None:
    assert make_doc_id("file:///a.md") == make_doc_id("file:///a.md")
    assert make_doc_id("file:///a.md") != make_doc_id("file:///b.md")


def test_chunk_id_depends_on_text_path_and_occurrence() -> None:
    base = make_chunk_id("doc", ("A",), "text", 0)
    assert base == make_chunk_id("doc", ("A",), "text", 0)
    assert base != make_chunk_id("doc", ("A",), "other", 0)
    assert base != make_chunk_id("doc", ("B",), "text", 0)
    assert base != make_chunk_id("doc", ("A",), "text", 1)
    assert base != make_chunk_id("doc2", ("A",), "text", 0)


def test_chunk_id_path_is_unambiguous() -> None:
    # ("a b",) and ("a", "b") must not collide through naive joining.
    assert make_chunk_id("d", ("a b",), "t", 0) != make_chunk_id("d", ("a", "b"), "t", 0)


def test_chunking_settings_reject_min_not_below_max() -> None:
    with pytest.raises(ValidationError):
        ChunkingSettings(min_tokens=100, max_tokens=100)


def test_settings_read_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNKING__MAX_TOKENS", "256")
    monkeypatch.setenv("RAG_CONTEXTUAL_LLM", "true")
    settings = Settings(_env_file=None)
    assert settings.chunking.max_tokens == 256
    assert settings.contextual_llm is True

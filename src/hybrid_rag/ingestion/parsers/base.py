from __future__ import annotations

from pathlib import PurePosixPath
from typing import Protocol
from urllib.parse import unquote, urlparse

from hybrid_rag.domain.models import DocumentTree, RawDocument


class Parser(Protocol):
    def parse(self, raw: RawDocument) -> DocumentTree: ...


def decode_text(raw: RawDocument) -> str:
    return raw.content.decode("utf-8-sig", errors="replace")


def fallback_title(source_uri: str) -> str:
    """File stem from the URI, used when a document declares no title."""
    return PurePosixPath(unquote(urlparse(source_uri).path)).stem or source_uri

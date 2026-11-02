from __future__ import annotations

from collections.abc import Callable

from hybrid_rag.domain.models import MimeType
from hybrid_rag.ingestion.parsers.base import Parser
from hybrid_rag.ingestion.parsers.html import HtmlParser
from hybrid_rag.ingestion.parsers.markdown import MarkdownParser
from hybrid_rag.ingestion.parsers.pdf import PdfParser

__all__ = ["HtmlParser", "MarkdownParser", "Parser", "PdfParser", "get_parser"]

_REGISTRY: dict[MimeType, Callable[[], Parser]] = {
    "text/markdown": MarkdownParser,
    "text/html": HtmlParser,
    "application/pdf": PdfParser,
}


def get_parser(mime_type: MimeType) -> Parser:
    return _REGISTRY[mime_type]()

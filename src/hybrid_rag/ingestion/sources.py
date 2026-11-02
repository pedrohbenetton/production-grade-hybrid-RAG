from __future__ import annotations

from pathlib import Path

from hybrid_rag.domain.models import MimeType, RawDocument

SUFFIX_TO_MIME: dict[str, MimeType] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
}


class UnsupportedFormatError(ValueError):
    pass


def mime_for(filename: str) -> MimeType:
    suffix = Path(filename).suffix.lower()
    try:
        return SUFFIX_TO_MIME[suffix]
    except KeyError:
        supported = ", ".join(sorted(SUFFIX_TO_MIME))
        raise UnsupportedFormatError(
            f"{suffix or filename!r} not supported ({supported})"
        ) from None


def load_file(path: Path) -> RawDocument:
    resolved = path.resolve()
    return RawDocument(
        source_uri=resolved.as_uri(), mime_type=mime_for(path.name), content=resolved.read_bytes()
    )

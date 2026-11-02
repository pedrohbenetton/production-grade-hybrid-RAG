"""Deterministic, content-addressed identifiers.

Parts are length-prefixed before hashing so different tuples can never
serialize to the same byte string (``("a b",)`` vs ``("a", "b")``).
"""

from __future__ import annotations

import hashlib

from hybrid_rag.domain.models import SectionPath


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        data = part.encode()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def make_doc_id(source_uri: str) -> str:
    return _digest("doc", source_uri)


def make_chunk_id(doc_id: str, section_path: SectionPath, text: str, occurrence: int) -> str:
    """``occurrence`` disambiguates identical text repeated under the same path.

    Position is deliberately excluded: inserting a paragraph must not change the
    IDs of every chunk after it, or re-ingestion would rewrite the whole document.
    """
    return _digest("chunk", doc_id, str(len(section_path)), *section_path, text, str(occurrence))

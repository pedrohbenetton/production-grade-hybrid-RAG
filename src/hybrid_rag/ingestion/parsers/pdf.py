"""PDF → DocumentTree using font-size statistics to recover heading structure."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

from hybrid_rag.domain.models import DocumentTree, RawDocument
from hybrid_rag.ingestion.parsers.base import fallback_title
from hybrid_rag.ingestion.tree_builder import TreeBuilder

HEADING_SIZE_RATIO = 1.15  # a line is a heading candidate if >= 1.15x body size
MARGIN_BAND = 0.10  # top/bottom fraction of the page scanned for running headers
MAX_HEADING_CHARS = 120


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    size: float
    page: int
    block: int
    in_margin: bool

    @property
    def signature(self) -> str:
        """Page-number-insensitive key for detecting running headers/footers."""
        return re.sub(r"\d+", "#", self.text.lower())


def _extract_lines(doc: pymupdf.Document) -> list[_Line]:
    lines: list[_Line] = []
    for index in range(doc.page_count):
        page = doc.load_page(index)
        page_no = index + 1
        height = float(page.rect.height)
        for block_no, block in enumerate(page.get_text("dict")["blocks"]):
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                y0, y1 = line["bbox"][1], line["bbox"][3]
                lines.append(
                    _Line(
                        text=" ".join("".join(s["text"] for s in spans).split()),
                        size=round(max(float(s["size"]) for s in spans), 1),
                        page=page_no,
                        block=block_no,
                        in_margin=y1 < height * MARGIN_BAND or y0 > height * (1 - MARGIN_BAND),
                    )
                )
    return lines


def _drop_running_furniture(lines: list[_Line], n_pages: int) -> list[_Line]:
    if n_pages < 2:
        return lines
    pages_by_sig: dict[str, set[int]] = {}
    for line in lines:
        if line.in_margin:
            pages_by_sig.setdefault(line.signature, set()).add(line.page)
    threshold = max(2, math.ceil(n_pages / 2))
    furniture = {sig for sig, pages in pages_by_sig.items() if len(pages) >= threshold}
    return [ln for ln in lines if not (ln.in_margin and ln.signature in furniture)]


def _heading_levels(lines: list[_Line]) -> dict[float, int]:
    weights: Counter[float] = Counter()
    for line in lines:
        weights[line.size] += len(line.text)
    if not weights:
        return {}
    body = weights.most_common(1)[0][0]
    sizes = sorted((s for s in weights if s >= body * HEADING_SIZE_RATIO), reverse=True)
    return {size: min(i + 1, 6) for i, size in enumerate(sizes)}


def _join_lines(parts: list[str]) -> str:
    text = parts[0]
    for part in parts[1:]:
        # De-hyphenate words broken across lines ("retrie-" + "val").
        hyphenated = text.endswith("-") and part[:1].islower()
        text = text[:-1] + part if hyphenated else f"{text} {part}"
    return text


class PdfParser:
    def parse(self, raw: RawDocument) -> DocumentTree:
        with pymupdf.open(stream=raw.content, filetype="pdf") as doc:
            meta_title = str((doc.metadata or {}).get("title") or "").strip()
            lines = _drop_running_furniture(_extract_lines(doc), doc.page_count)

        levels = _heading_levels(lines)
        builder = TreeBuilder(meta_title or fallback_title(raw.source_uri))
        body: list[_Line] = []

        def flush() -> None:
            if body:
                builder.block("paragraph", _join_lines([ln.text for ln in body]), body[0].page)
                body.clear()

        heading: list[_Line] = []
        for line in lines:
            is_heading = line.size in levels and len(line.text) <= MAX_HEADING_CHARS
            if heading and not (is_heading and line.size == heading[0].size):
                builder.heading(_join_lines([h.text for h in heading]), levels[heading[0].size])
                heading.clear()
            if is_heading:
                flush()
                heading.append(line)
                continue
            if body and (line.block != body[-1].block or line.page != body[-1].page):
                flush()
            body.append(line)
        if heading:
            builder.heading(_join_lines([h.text for h in heading]), levels[heading[0].size])
        flush()
        return builder.build(raw.source_uri)

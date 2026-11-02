"""Pack reranked chunks into a numbered, token-budgeted prompt context (§4.5)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from hybrid_rag.domain.models import ContextualChunk, ScoredChunk
from hybrid_rag.ports import TokenCounter


class ContextPassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    chunk: ContextualChunk

    @property
    def source(self) -> str:
        """Human-readable provenance: the breadcrumb, else the section path."""
        prefix = self.chunk.context_prefix.strip()
        label = prefix.splitlines()[0] if prefix else " > ".join(self.chunk.section_path)
        return label.replace('"', "'")


class Context(BaseModel):
    model_config = ConfigDict(frozen=True)

    passages: tuple[ContextPassage, ...]

    def render(self) -> str:
        return "\n\n".join(self._render(p) for p in self.passages)

    @staticmethod
    def _render(passage: ContextPassage) -> str:
        page = f' page="{passage.chunk.page}"' if passage.chunk.page is not None else ""
        # Passages are untrusted text: stop them from closing their own tag.
        body = passage.chunk.text.replace("</passage", "&lt;/passage")
        return f'<passage id="{passage.n}" source="{passage.source}"{page}>\n{body}\n</passage>'


def build_context(
    hits: Sequence[ScoredChunk], *, token_budget: int, counter: TokenCounter
) -> Context:
    """Take hits in rank order until the budget is spent; the top hit is always kept."""
    passages: list[ContextPassage] = []
    used = 0
    for hit in hits:
        cost = counter.count(hit.chunk.text)
        if passages and used + cost > token_budget:
            break
        passages.append(ContextPassage(n=len(passages) + 1, chunk=hit.chunk))
        used += cost
    return Context(passages=tuple(passages))

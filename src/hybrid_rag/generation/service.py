"""Retrieve → generate → verify → apply policy (§5)."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from hybrid_rag.config import GuardrailSettings, LLMSettings
from hybrid_rag.generation.context import Context, ContextPassage, build_context
from hybrid_rag.generation.llm import LLMClient
from hybrid_rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from hybrid_rag.guardrails.claims import extract_claims, is_abstention
from hybrid_rag.guardrails.models import Claim, ClaimVerdict, FaithfulnessReport
from hybrid_rag.guardrails.policy import AbstainReason, GuardrailPolicy, apply_policy
from hybrid_rag.ingestion.tokens import RegexTokenCounter
from hybrid_rag.ports import TokenCounter
from hybrid_rag.retrieval.hybrid import RetrievalMode, RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query: str, *, mode: RetrievalMode, k: int) -> RetrievalResult: ...


class Verifier(Protocol):
    def verify(
        self, claims: Sequence[Claim], passages: Sequence[ContextPassage]
    ) -> list[ClaimVerdict]: ...


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    chunk_id: str
    doc_id: str
    source: str
    section_path: tuple[str, ...]
    page: int | None
    text: str


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    mode: RetrievalMode
    answer: str | None
    abstained: bool
    abstain_reason: AbstainReason | None
    faithfulness: float | None
    claims: tuple[ClaimVerdict, ...]  # every claim, including those the policy dropped
    invalid_citations: tuple[int, ...]
    citations: tuple[Citation, ...]  # passages cited by the claims that were kept
    timings_ms: dict[str, float]


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        verifier: Verifier,
        *,
        settings: LLMSettings,
        guardrails: GuardrailSettings,
        counter: TokenCounter | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._verifier = verifier
        self._settings = settings
        self._guardrails = guardrails
        self._counter = counter or RegexTokenCounter()

    def answer(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid_rerank",
        k: int = 5,
        policy: GuardrailPolicy | None = None,
    ) -> AnswerResult:
        policy = policy or GuardrailPolicy(
            mode=self._guardrails.policy, min_faithfulness=self._guardrails.min_faithfulness
        )
        start = perf_counter()
        retrieval = self._retriever.retrieve(query, mode=mode, k=k)
        timings = {
            ("retrieval" if stage == "total" else stage): ms
            for stage, ms in retrieval.timings_ms.items()
        }

        def done(result: AnswerResult) -> AnswerResult:
            timings["total"] = (perf_counter() - start) * 1000
            return result.model_copy(update={"timings_ms": timings})

        def refuse(reason: AbstainReason, report: FaithfulnessReport | None = None) -> AnswerResult:
            report = report or FaithfulnessReport(claims=())
            return done(self._result(query, mode, None, reason, report, Context(passages=())))

        # The cheapest hallucination is the one never generated.
        floor = self._guardrails.min_relevance
        if not retrieval.hits or (floor is not None and retrieval.hits[0].score < floor):
            return refuse("no_relevant_context")

        context = build_context(
            retrieval.hits, token_budget=self._settings.context_token_budget, counter=self._counter
        )
        t0 = perf_counter()
        draft = self._llm.complete(
            system=SYSTEM_PROMPT,
            prompt=build_user_prompt(query, context),
            max_tokens=self._settings.max_tokens,
        )
        timings["generate"] = (perf_counter() - t0) * 1000
        if is_abstention(draft):
            return refuse("model_abstained")

        t0 = perf_counter()
        claims, invalid = extract_claims(draft, len(context.passages))
        report = FaithfulnessReport(
            claims=tuple(self._verifier.verify(claims, context.passages)),
            invalid_citations=invalid,
        )
        timings["verify"] = (perf_counter() - t0) * 1000

        decision = apply_policy(draft, report, policy)
        if decision.abstained:
            return refuse(decision.abstain_reason or "no_supported_claims", report)
        return done(
            self._result(query, mode, decision.answer, None, report, context, decision.kept)
        )

    @staticmethod
    def _result(
        query: str,
        mode: RetrievalMode,
        answer: str | None,
        reason: AbstainReason | None,
        report: FaithfulnessReport,
        context: Context,
        kept: Sequence[ClaimVerdict] = (),
    ) -> AnswerResult:
        cited = {n for verdict in kept for n in verdict.claim.citations}
        return AnswerResult(
            query=query,
            mode=mode,
            answer=answer,
            abstained=answer is None,
            abstain_reason=reason,
            faithfulness=report.faithfulness,
            claims=report.claims,
            invalid_citations=report.invalid_citations,
            citations=tuple(
                Citation(
                    n=p.n,
                    chunk_id=p.chunk.chunk_id,
                    doc_id=p.chunk.doc_id,
                    source=p.source,
                    section_path=p.chunk.section_path,
                    page=p.chunk.page,
                    text=p.chunk.text,
                )
                for p in context.passages
                if p.n in cited
            ),
            timings_ms={},
        )

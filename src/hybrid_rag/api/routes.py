from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from hybrid_rag.api.container import Container
from hybrid_rag.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from hybrid_rag.generation.llm import LLMError
from hybrid_rag.guardrails.policy import GuardrailPolicy
from hybrid_rag.retrieval.hybrid import RerankerUnavailableError

router = APIRouter()


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


ContainerDep = Annotated[Container, Depends(get_container)]


@router.get("/health", tags=["ops"], summary="Liveness probe")
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", tags=["ops"], summary="Readiness probe (indexes loaded)")
def ready(container: ContainerDep) -> ReadyResponse:
    return ReadyResponse(chunks=container.dense.count())


# Sync handlers on purpose: retrieval is CPU-bound (ONNX, BM25), so FastAPI runs
# them in its threadpool instead of blocking the event loop.
@router.post(
    "/v1/retrieve",
    tags=["retrieval"],
    summary="Retrieve passages (no generation)",
    responses={status.HTTP_400_BAD_REQUEST: {"description": "Mode unavailable in this deployment"}},
)
def retrieve(request: RetrieveRequest, container: ContainerDep) -> RetrieveResponse:
    try:
        result = container.retriever.retrieve(request.query, mode=request.mode, k=request.k)
    except RerankerUnavailableError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return RetrieveResponse.from_result(result)


@router.post(
    "/v1/query",
    tags=["generation"],
    summary="Answer a question from the indexed documents, with faithfulness checks",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Generation or mode unavailable"},
        status.HTTP_502_BAD_GATEWAY: {"description": "LLM provider failed"},
    },
)
def query(request: QueryRequest, container: ContainerDep) -> QueryResponse:
    if container.service is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "generation is not configured: set ANTHROPIC_API_KEY (or RAG_LLM__API_KEY)",
        )
    policy = None
    if request.policy is not None:
        guardrails = container.settings.guardrails
        policy = GuardrailPolicy(mode=request.policy, min_faithfulness=guardrails.min_faithfulness)
    try:
        result = container.service.answer(
            request.query, mode=request.mode, k=request.k, policy=policy
        )
    except RerankerUnavailableError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return QueryResponse.from_result(result)

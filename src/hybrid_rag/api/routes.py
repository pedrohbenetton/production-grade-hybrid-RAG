from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from hybrid_rag.api.container import Container
from hybrid_rag.api.schemas import HealthResponse, ReadyResponse, RetrieveRequest, RetrieveResponse
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

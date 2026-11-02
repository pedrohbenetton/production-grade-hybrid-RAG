import base64
from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from hybrid_rag.api.container import Container
from hybrid_rag.api.schemas import (
    DeleteDocumentResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from hybrid_rag.domain.ids import make_doc_id
from hybrid_rag.domain.models import RawDocument
from hybrid_rag.generation.llm import LLMError
from hybrid_rag.guardrails.policy import GuardrailPolicy
from hybrid_rag.ingestion.sources import UnsupportedFormatError, mime_for
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


@router.post(
    "/v1/ingest",
    tags=["ingestion"],
    summary="Ingest a document from text or base64 into dense and sparse indexes",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported format, blank content, or invalid document"
        }
    },
)
def ingest(request: IngestRequest, container: ContainerDep) -> IngestResponse:
    mime = request.mime_type
    if mime is None:
        try:
            mime = mime_for(request.source_uri)
        except UnsupportedFormatError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if mime == "application/pdf":
        try:
            content_bytes = base64.b64decode(request.content)
        except Exception as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"invalid base64 content for PDF: {exc}"
            ) from exc
    else:
        content_bytes = request.content.encode("utf-8")

    if not content_bytes.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "document content must not be blank")

    raw = RawDocument(source_uri=request.source_uri, mime_type=mime, content=content_bytes)
    try:
        processed = container.pipeline.process(raw)
        report = container.indexer.index(processed)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"failed to ingest document: {exc}"
        ) from exc
    return IngestResponse.from_report(report)


@router.post(
    "/v1/ingest/file",
    tags=["ingestion"],
    summary="Ingest an uploaded file (Markdown, HTML, or PDF) into dense and sparse indexes",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported format, empty file, or invalid document"
        }
    },
)
def ingest_file(
    file: Annotated[UploadFile, File(description="File to index (Markdown, HTML, or PDF)")],
    container: ContainerDep,
    source_uri: Annotated[
        str | None,
        Form(description="Optional custom source URI. Defaults to file.filename."),
    ] = None,
) -> IngestResponse:
    filename = file.filename or "unknown"
    uri = source_uri.strip() if source_uri and source_uri.strip() else filename
    try:
        mime = mime_for(filename)
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    content_bytes = file.file.read()
    if not content_bytes.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file content must not be empty")

    raw = RawDocument(source_uri=uri, mime_type=mime, content=content_bytes)
    try:
        processed = container.pipeline.process(raw)
        report = container.indexer.index(processed)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"failed to ingest document: {exc}"
        ) from exc
    return IngestResponse.from_report(report)


@router.delete(
    "/v1/documents/{doc_id:path}",
    tags=["ingestion"],
    summary="Delete a document and its indexed chunks by doc_id or source_uri",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Document not found in the index"}},
)
def delete_document(doc_id: str, container: ContainerDep) -> DeleteDocumentResponse:
    deleted = container.indexer.delete_document(doc_id)
    if deleted > 0:
        return DeleteDocumentResponse(doc_id=doc_id, deleted_chunks=deleted)

    # Check if doc_id was passed as the source_uri
    derived = make_doc_id(doc_id)
    if derived != doc_id:
        deleted = container.indexer.delete_document(derived)
        if deleted > 0:
            return DeleteDocumentResponse(doc_id=derived, deleted_chunks=deleted)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"document {doc_id!r} not found",
    )


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

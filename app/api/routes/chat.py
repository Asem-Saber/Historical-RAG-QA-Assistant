import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, SourceDocument
from app.api.dependencies import get_pipeline, get_stream_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Deep health check — verifies Chroma and Ollama are reachable."""
    from app.core.pipeline import vectorstore, embeddings

    errors = []

    try:
        vectorstore.get(limit=1, include=[])
    except Exception as e:
        errors.append(f"chroma: {e}")

    try:
        embeddings.embed_query("ping")
    except Exception as e:
        errors.append(f"ollama: {e}")

    if errors:
        return {"status": "degraded", "errors": errors}

    return {"status": "healthy"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline=Depends(get_pipeline)):
    """Send a question about Ancient Egypt and receive a RAG-powered answer with source documents."""
    logger.info("Chat request: %s", request.query[:100])

    try:
        result = pipeline(request.query)
    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Pipeline error: {e}")

    source_documents = [
        SourceDocument(
            citation=i,
            content=doc.page_content,
            metadata=doc.metadata,
        )
        for i, doc in enumerate(result.get("source_documents", []), 1)
    ]

    return ChatResponse(
        answer=result["answer"],
        source_documents=source_documents,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, stream_pipeline=Depends(get_stream_pipeline)):
    """Stream a RAG-powered answer token by token via Server-Sent Events."""
    logger.info("Stream request: %s", request.query[:100])
    return StreamingResponse(
        stream_pipeline(request.query),
        media_type="text/event-stream",
    )


@router.get("/metrics")
async def get_metrics():
    """Return running averages for latency and token consumption."""
    from app.core.pipeline import metrics
    return metrics.snapshot()

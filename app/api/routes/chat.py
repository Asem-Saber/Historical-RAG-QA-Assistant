from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.schemas import ChatRequest, ChatResponse, SourceDocument
from app.api.dependencies import get_pipeline, get_stream_pipeline


router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline=Depends(get_pipeline)):
    """Send a question about Ancient Egypt and receive a RAG-powered answer with source documents."""
    result = pipeline(request.query)

    source_documents = [
        SourceDocument(
            content=doc.page_content,
            metadata=doc.metadata,
        )
        for doc in result.get("source_documents", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        source_documents=source_documents,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, stream_pipeline=Depends(get_stream_pipeline)):
    """Stream a RAG-powered answer token by token via Server-Sent Events."""
    return StreamingResponse(
        stream_pipeline(request.query),
        media_type="text/event-stream",
    )

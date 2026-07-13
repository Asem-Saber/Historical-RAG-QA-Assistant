from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""
    query: str = Field(..., min_length=1, max_length=1000)


class SourceDocument(BaseModel):
    """A single retrieved source document with citation index."""
    citation: int
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    """Chat response with answer and sources."""
    answer: str
    source_documents: list[SourceDocument]
    thread_id: str = ""

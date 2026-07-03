from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Incoming chat request."""
    query: str


class SourceDocument(BaseModel):
    """A single retrieved source document."""
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    """Chat response with answer and sources."""
    answer: str
    source_documents: list[SourceDocument]

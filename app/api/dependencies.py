from app.core.pipeline import AncientEgyptRAG, AncientEgyptRAGStream


def get_pipeline():
    """FastAPI dependency that provides the RAG pipeline function."""
    return AncientEgyptRAG


def get_stream_pipeline():
    """FastAPI dependency that provides the streaming RAG pipeline function."""
    return AncientEgyptRAGStream
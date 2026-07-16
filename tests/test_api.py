"""Integration tests for the FastAPI API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def client():
    """Create a test client with mocked pipeline dependencies."""
    with patch("app.core.pipeline.OllamaEmbeddings"), \
         patch("app.core.pipeline.Chroma"), \
         patch("app.core.pipeline.ChatOpenAI"), \
         patch("app.core.pipeline.build_hybrid_retriever"), \
         patch("app.core.pipeline.CrossEncoder"), \
         patch("app.core.pipeline.create_retriever_tool"), \
         patch("app.core.pipeline.build_graph"):

        from fastapi.testclient import TestClient
        from app.api.main import app

        with TestClient(app) as tc:
            yield tc


@pytest.fixture
def mock_pipeline():
    """Mock that returns a valid pipeline response."""
    return MagicMock(return_value={
        "answer": "The Great Pyramid was built by Pharaoh Khufu.",
        "source_documents": [
            Document(page_content="Khufu built the pyramid.", metadata={"source": "ch1"}),
        ],
        "thread_id": "abc123",
    })


@pytest.fixture
def mock_stream_pipeline():
    """Mock that yields SSE chunks."""
    import json

    def stream(query):
        yield f"data: {json.dumps({'type': 'chunk', 'content': 'The Great'})}\n\n"
        yield f"data: {json.dumps({'type': 'chunk', 'content': ' Pyramid'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'thread_id': 'abc123'})}\n\n"

    return stream


# =============================================================================
# Health endpoint
# =============================================================================


class TestHealthEndpoint:
    def test_healthy_response(self, client):
        with patch("app.core.pipeline.vectorstore") as mock_vs, \
             patch("app.core.pipeline.embeddings") as mock_emb:
            mock_vs.get.return_value = {}
            mock_emb.embed_query.return_value = [0.1, 0.2]

            response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_degraded_when_chroma_fails(self, client):
        with patch("app.core.pipeline.vectorstore") as mock_vs, \
             patch("app.core.pipeline.embeddings") as mock_emb:
            mock_vs.get.side_effect = Exception("Chroma unreachable")
            mock_emb.embed_query.return_value = [0.1]

            response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert any("chroma" in e for e in data["errors"])

    def test_degraded_when_ollama_fails(self, client):
        with patch("app.core.pipeline.vectorstore") as mock_vs, \
             patch("app.core.pipeline.embeddings") as mock_emb:
            mock_vs.get.return_value = {}
            mock_emb.embed_query.side_effect = Exception("Ollama down")

            response = client.get("/api/health")

        data = response.json()
        assert data["status"] == "degraded"
        assert any("ollama" in e for e in data["errors"])


# =============================================================================
# Chat endpoint
# =============================================================================


class TestChatEndpoint:
    def test_valid_chat_request(self, client, mock_pipeline):
        with patch("app.api.dependencies.get_pipeline", return_value=lambda: mock_pipeline):
            with patch("app.api.routes.chat.get_pipeline", return_value=mock_pipeline):
                response = client.post("/api/chat", json={"query": "Who built the pyramids?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "source_documents" in data
        assert "thread_id" in data

    def test_empty_query_rejected(self, client):
        response = client.post("/api/chat", json={"query": ""})
        assert response.status_code == 422

    def test_query_too_long_rejected(self, client):
        response = client.post("/api/chat", json={"query": "x" * 1001})
        assert response.status_code == 422

    def test_missing_query_field(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_pipeline_error_returns_502(self, client):
        from app.api.main import app
        from app.api.dependencies import get_pipeline

        def failing():
            raise RuntimeError("LLM timeout")

        app.dependency_overrides[get_pipeline] = lambda: failing
        try:
            response = client.post("/api/chat", json={"query": "test question"})
            assert response.status_code == 502
        finally:
            app.dependency_overrides.pop(get_pipeline, None)


# =============================================================================
# Stream endpoint
# =============================================================================


class TestStreamEndpoint:
    def test_stream_returns_sse(self, client, mock_stream_pipeline):
        with patch("app.api.routes.chat.get_stream_pipeline", return_value=mock_stream_pipeline):
            response = client.post("/api/chat/stream", json={"query": "Tell me about pyramids"})

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_empty_query_rejected(self, client):
        response = client.post("/api/chat/stream", json={"query": ""})
        assert response.status_code == 422


# =============================================================================
# Metrics endpoint
# =============================================================================


class TestMetricsEndpoint:
    def test_returns_metrics(self, client):
        with patch("app.core.pipeline.metrics") as mock_metrics:
            mock_metrics.snapshot.return_value = {
                "requests": 5,
                "avg_latency": 1.2,
                "total_input_tokens": 500,
                "total_output_tokens": 200,
                "avg_input_tokens": 100,
                "avg_output_tokens": 40,
            }
            response = client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "requests" in data

    def test_zero_requests(self, client):
        with patch("app.core.pipeline.metrics") as mock_metrics:
            mock_metrics.snapshot.return_value = {"requests": 0}
            response = client.get("/api/metrics")

        assert response.status_code == 200


# =============================================================================
# State endpoint
# =============================================================================


class TestStateEndpoint:
    def test_not_found_thread(self, client):
        with patch("app.core.graph.get_state", return_value=None):
            response = client.get("/api/state/nonexistent")
        assert response.status_code == 404

    def test_valid_thread_state(self, client):
        from app.core.models import GuardrailScoring, GradingResult

        mock_state = {
            "channel_values": {
                "original_query": "test query",
                "rewritten_query": None,
                "sub_queries": ["sub1"],
                "retrieval_attempts": 1,
                "guardrail_result": GuardrailScoring(score=85, reason="Egypt topic"),
                "routing_decision": "generate_answer",
                "documents": [
                    Document(page_content="Pyramids content here", metadata={"source": "ch1"}),
                ],
                "grading_results": [
                    GradingResult(document_id="batch", is_relevant=True, score=1.0, reasoning="Relevant"),
                ],
                "sources": [],
                "metadata": {},
                "messages": [],
            }
        }

        with patch("app.core.graph.get_state", return_value=mock_state):
            response = client.get("/api/state/thread123")

        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "thread123"
        assert data["original_query"] == "test query"
        assert data["document_count"] == 1


# =============================================================================
# Root / frontend
# =============================================================================


class TestRootEndpoint:
    def test_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

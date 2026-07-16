"""Shared test fixtures — mocks for LLM, embeddings, vectorstore, and reranker."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


# --- Fake LLM ---


class FakeLLM:
    """Mock LLM that returns configurable responses."""

    def __init__(self, response_content="Test answer"):
        self.response_content = response_content
        self._structured_schema = None

    def invoke(self, prompt, **kwargs):
        return AIMessage(content=self.response_content)

    def with_structured_output(self, schema):
        structured = FakeStructuredLLM(schema, self.response_content)
        return structured


class FakeStructuredLLM:
    """Mock for llm.with_structured_output() calls."""

    def __init__(self, schema, fallback_content=""):
        self.schema = schema
        self.fallback_content = fallback_content
        self._response = None

    def set_response(self, response):
        self._response = response

    def invoke(self, prompt, **kwargs):
        if self._response is not None:
            return self._response
        return self._default_response()

    def _default_response(self):
        from app.core.models import (
            AnswerRelevanceCheck,
            GradeDocuments,
            GuardrailScoring,
            HallucinationCheck,
        )

        if self.schema == GuardrailScoring:
            return GuardrailScoring(score=85, reason="About Ancient Egypt")
        elif self.schema == GradeDocuments:
            return GradeDocuments(binary_score="yes", reasoning="Relevant content")
        elif self.schema == HallucinationCheck:
            return HallucinationCheck(grounded="yes", reasoning="Grounded in sources")
        elif self.schema == AnswerRelevanceCheck:
            return AnswerRelevanceCheck(relevant="yes", reasoning="Addresses the question")
        return MagicMock()


# --- Fixtures ---


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="[Subject: Pyramids]\n\nThe Great Pyramid of Giza was built by Pharaoh Khufu.",
            metadata={"Major_Entry": "Pyramids", "Minor_Entry": "Great Pyramid"},
        ),
        Document(
            page_content="[Subject: Mummification]\n\nMummification was practiced to preserve the body for the afterlife.",
            metadata={"Major_Entry": "Mummification"},
        ),
        Document(
            page_content="[Subject: Pharaohs]\n\nRamesses II ruled for 66 years during the New Kingdom.",
            metadata={"Major_Entry": "Pharaohs", "Minor_Entry": "Ramesses II"},
        ),
    ]


@pytest.fixture
def base_state(sample_documents):
    return {
        "messages": [HumanMessage(content="Who built the Great Pyramid?")],
        "original_query": None,
        "rewritten_query": None,
        "sub_queries": [],
        "retrieval_attempts": 0,
        "guardrail_result": None,
        "routing_decision": None,
        "documents": sample_documents,
        "grading_results": [],
        "sources": [],
        "metadata": {},
    }


@pytest.fixture
def empty_state():
    return {
        "messages": [HumanMessage(content="Who built the Great Pyramid?")],
        "original_query": None,
        "rewritten_query": None,
        "sub_queries": [],
        "retrieval_attempts": 0,
        "guardrail_result": None,
        "routing_decision": None,
        "documents": [],
        "grading_results": [],
        "sources": [],
        "metadata": {},
    }

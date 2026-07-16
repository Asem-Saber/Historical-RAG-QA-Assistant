"""Tests for app.nodes — all graph pipeline steps."""

import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.models import (
    AnswerRelevanceCheck,
    GradeDocuments,
    GuardrailScoring,
    HallucinationCheck,
)
from app.nodes.decompose import decompose_query_step
from app.nodes.generate import generate_answer_step
from app.nodes.grade_documents import grade_documents_step
from app.nodes.grade_generation import check_generation_quality, grade_generation_step
from app.nodes.guardrail import GUARDRAIL_THRESHOLD, continue_after_guardrail, guardrail_step
from app.nodes.out_of_scope import out_of_scope_step
from app.nodes.process_retrieval import process_retrieval_step
from app.nodes.retrieve import MAX_RETRIEVAL_ATTEMPTS, retrieve_step
from app.nodes.rewrite_query import rewrite_query_step
from app.nodes.utils import get_latest_query


# =============================================================================
# utils
# =============================================================================


class TestGetLatestQuery:
    def test_returns_last_human_message(self):
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="response"),
            HumanMessage(content="second"),
        ]
        assert get_latest_query(messages) == "second"

    def test_returns_empty_when_no_human_message(self):
        messages = [AIMessage(content="hello")]
        assert get_latest_query(messages) == ""

    def test_returns_empty_for_empty_list(self):
        assert get_latest_query([]) == ""


# =============================================================================
# guardrail
# =============================================================================


class TestGuardrailStep:
    def test_returns_guardrail_result(self, base_state, fake_llm):
        result = guardrail_step(base_state, fake_llm)
        assert "guardrail_result" in result
        assert isinstance(result["guardrail_result"], GuardrailScoring)

    def test_high_score_for_egypt_query(self, base_state, fake_llm):
        result = guardrail_step(base_state, fake_llm)
        assert result["guardrail_result"].score >= GUARDRAIL_THRESHOLD

    def test_fallback_on_exception(self, base_state):
        class FailingLLM:
            def with_structured_output(self, schema):
                return self

            def invoke(self, prompt, **kwargs):
                raise RuntimeError("LLM down")

        result = guardrail_step(base_state, FailingLLM())
        assert result["guardrail_result"].score == 50


class TestContinueAfterGuardrail:
    def test_continue_when_above_threshold(self):
        state = {"guardrail_result": GuardrailScoring(score=80, reason="Egypt")}
        assert continue_after_guardrail(state) == "continue"

    def test_out_of_scope_when_below_threshold(self):
        state = {"guardrail_result": GuardrailScoring(score=20, reason="Not Egypt")}
        assert continue_after_guardrail(state) == "out_of_scope"

    def test_continue_when_no_guardrail_result(self):
        state = {"guardrail_result": None}
        assert continue_after_guardrail(state) == "continue"

    def test_boundary_at_threshold(self):
        state = {"guardrail_result": GuardrailScoring(score=GUARDRAIL_THRESHOLD, reason="Borderline")}
        assert continue_after_guardrail(state) == "continue"

    def test_just_below_threshold(self):
        state = {"guardrail_result": GuardrailScoring(score=GUARDRAIL_THRESHOLD - 1, reason="Below")}
        assert continue_after_guardrail(state) == "out_of_scope"


# =============================================================================
# out_of_scope
# =============================================================================


class TestOutOfScopeStep:
    def test_returns_ai_message(self, base_state):
        result = out_of_scope_step(base_state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_contains_original_query(self, base_state):
        result = out_of_scope_step(base_state)
        assert "Who built the Great Pyramid?" in result["messages"][0].content

    def test_mentions_domain(self, base_state):
        result = out_of_scope_step(base_state)
        assert "Ancient Egypt" in result["messages"][0].content


# =============================================================================
# decompose
# =============================================================================


class TestDecomposeQueryStep:
    def test_returns_sub_queries(self, base_state):
        class DecomposeLLM:
            def __or__(self, other):
                return self

            def invoke(self, input, **kwargs):
                return AIMessage(content='{"sub_queries": ["What is the Great Pyramid?", "Who was Khufu?"]}')

        from unittest.mock import patch
        from langchain_core.output_parsers import JsonOutputParser

        class FakeChain:
            def invoke(self, input, **kwargs):
                return {"sub_queries": ["What is the Great Pyramid?", "Who was Khufu?"]}

        with patch("app.nodes.decompose.prompt_decomposition") as mock_prompt:
            mock_prompt.__or__ = lambda self, other: FakeChain()

            # Use a simpler approach: patch the chain invocation
            result = decompose_query_step(base_state, DecomposeLLM())

        assert "sub_queries" in result
        assert len(result["sub_queries"]) >= 1

    def test_fallback_on_exception(self, base_state):
        """When LLM chain fails, decompose falls back to the original query."""
        from unittest.mock import patch, MagicMock

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM failure")

        with patch("app.nodes.decompose.prompt_decomposition.__or__", return_value=mock_chain):
            with patch.object(mock_chain, "__or__", return_value=mock_chain):
                result = decompose_query_step(base_state, MagicMock())

        assert result["sub_queries"] == ["Who built the Great Pyramid?"]


# =============================================================================
# retrieve
# =============================================================================


class TestRetrieveStep:
    def test_emits_tool_calls(self, base_state):
        base_state["sub_queries"] = ["What is the Great Pyramid?"]
        result = retrieve_step(base_state)
        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0].tool_calls is not None
        assert len(messages[0].tool_calls) == 1
        assert messages[0].tool_calls[0]["name"] == "retrieve_documents"

    def test_sets_original_query_on_first_call(self, base_state):
        base_state["sub_queries"] = ["sub1"]
        result = retrieve_step(base_state)
        assert result.get("original_query") == "Who built the Great Pyramid?"

    def test_does_not_overwrite_original_query(self, base_state):
        base_state["original_query"] = "original question"
        base_state["sub_queries"] = ["sub1"]
        result = retrieve_step(base_state)
        assert "original_query" not in result

    def test_increments_retrieval_attempts(self, base_state):
        base_state["sub_queries"] = ["sub1"]
        result = retrieve_step(base_state)
        assert result["retrieval_attempts"] == 1

    def test_max_attempts_returns_fallback(self, base_state):
        base_state["retrieval_attempts"] = MAX_RETRIEVAL_ATTEMPTS
        base_state["sub_queries"] = ["sub1"]
        result = retrieve_step(base_state)
        assert isinstance(result["messages"][0], AIMessage)
        assert "couldn't find" in result["messages"][0].content
        assert result["documents"] == []

    def test_multiple_sub_queries_emit_multiple_tool_calls(self, base_state):
        base_state["sub_queries"] = ["q1", "q2", "q3"]
        result = retrieve_step(base_state)
        assert len(result["messages"][0].tool_calls) == 3

    def test_uses_rewritten_query_if_present(self, base_state):
        base_state["rewritten_query"] = "rewritten version"
        base_state["sub_queries"] = ["rewritten version"]
        result = retrieve_step(base_state)
        tool_calls = result["messages"][0].tool_calls
        assert tool_calls[0]["args"]["query"] == "rewritten version"


# =============================================================================
# process_retrieval
# =============================================================================


class TestProcessRetrievalStep:
    def test_extracts_documents_from_tool_messages(self):
        docs_json = json.dumps([
            {"page_content": "Pyramids were tombs.", "metadata": {"source": "ch1"}},
            {"page_content": "Pharaohs ruled Egypt.", "metadata": {"source": "ch2"}},
        ])
        state = {
            "messages": [
                HumanMessage(content="query"),
                AIMessage(content="", tool_calls=[{"id": "t1", "name": "retrieve_documents", "args": {"query": "q"}}]),
                ToolMessage(content=docs_json, tool_call_id="t1"),
            ],
        }
        result = process_retrieval_step(state)
        assert len(result["documents"]) == 2
        assert result["documents"][0].page_content == "Pyramids were tombs."

    def test_deduplicates_documents(self):
        doc = {"page_content": "Same content repeated for dedup testing which is more than 120 chars to test the key logic properly.", "metadata": {}}
        docs_json = json.dumps([doc, doc])
        state = {
            "messages": [
                ToolMessage(content=docs_json, tool_call_id="t1"),
            ],
        }
        result = process_retrieval_step(state)
        assert len(result["documents"]) == 1

    def test_handles_invalid_json(self):
        state = {
            "messages": [
                ToolMessage(content="not valid json", tool_call_id="t1"),
            ],
        }
        result = process_retrieval_step(state)
        assert result["documents"] == []

    def test_stops_at_non_tool_message(self):
        docs_json = json.dumps([{"page_content": "Recent doc", "metadata": {}}])
        old_docs_json = json.dumps([{"page_content": "Old doc", "metadata": {}}])
        state = {
            "messages": [
                ToolMessage(content=old_docs_json, tool_call_id="t0"),
                AIMessage(content="separator"),
                ToolMessage(content=docs_json, tool_call_id="t1"),
            ],
        }
        result = process_retrieval_step(state)
        # Only picks up the last contiguous ToolMessages from the end
        assert len(result["documents"]) == 1
        assert result["documents"][0].page_content == "Recent doc"


# =============================================================================
# grade_documents
# =============================================================================


class TestGradeDocumentsStep:
    def test_routes_to_generate_when_relevant(self, base_state, fake_llm):
        result = grade_documents_step(base_state, fake_llm)
        assert result["routing_decision"] == "generate_answer"
        assert len(result["grading_results"]) == 1
        assert result["grading_results"][0].is_relevant is True

    def test_routes_to_rewrite_when_irrelevant(self, base_state):
        from tests.conftest import FakeStructuredLLM
        from app.core.models import GradeDocuments

        class IrrelevantLLM:
            def with_structured_output(self, schema):
                llm = FakeStructuredLLM(schema)
                llm.set_response(GradeDocuments(binary_score="no", reasoning="Not relevant"))
                return llm

        result = grade_documents_step(base_state, IrrelevantLLM())
        assert result["routing_decision"] == "rewrite_query"

    def test_routes_to_rewrite_when_no_documents(self, empty_state, fake_llm):
        result = grade_documents_step(empty_state, fake_llm)
        assert result["routing_decision"] == "rewrite_query"
        assert result["grading_results"] == []

    def test_fallback_on_llm_failure(self, base_state):
        class FailingLLM:
            def with_structured_output(self, schema):
                return self

            def invoke(self, prompt, **kwargs):
                raise RuntimeError("LLM down")

        result = grade_documents_step(base_state, FailingLLM())
        # Fallback heuristic: docs exist -> relevant
        assert result["routing_decision"] == "generate_answer"
        assert "Fallback" in result["grading_results"][0].reasoning


# =============================================================================
# rewrite_query
# =============================================================================


class TestRewriteQueryStep:
    def test_returns_rewritten_query(self, base_state):
        from tests.conftest import FakeLLM

        llm = FakeLLM(response_content="What pharaoh commissioned the Great Pyramid of Giza?")
        result = rewrite_query_step(base_state, llm)
        assert result["rewritten_query"] == "What pharaoh commissioned the Great Pyramid of Giza?"
        assert isinstance(result["messages"][0], HumanMessage)

    def test_fallback_on_empty_rewrite(self, base_state):
        from tests.conftest import FakeLLM

        llm = FakeLLM(response_content="")
        result = rewrite_query_step(base_state, llm)
        assert "ancient egypt" in result["rewritten_query"].lower()

    def test_fallback_on_exception(self, base_state):
        class FailingLLM:
            def invoke(self, prompt, **kwargs):
                raise RuntimeError("LLM down")

        result = rewrite_query_step(base_state, FailingLLM())
        assert "ancient egypt" in result["rewritten_query"].lower()
        assert "Who built the Great Pyramid?" in result["rewritten_query"]


# =============================================================================
# generate_answer
# =============================================================================


class TestGenerateAnswerStep:
    def test_generates_answer_with_docs(self, base_state):
        from tests.conftest import FakeLLM

        llm = FakeLLM(response_content="The Great Pyramid was built by Pharaoh Khufu [1].")
        result = generate_answer_step(base_state, llm)
        assert isinstance(result["messages"][0], AIMessage)
        assert "Khufu" in result["messages"][0].content
        assert len(result["sources"]) == len(base_state["documents"])

    def test_no_docs_returns_insufficient_info(self, empty_state, fake_llm):
        result = generate_answer_step(empty_state, fake_llm)
        assert "don't have enough information" in result["messages"][0].content
        assert result["sources"] == []

    def test_exception_returns_error_message(self, base_state):
        class FailingLLM:
            def invoke(self, prompt, **kwargs):
                raise RuntimeError("Generation failed")

        result = generate_answer_step(base_state, FailingLLM())
        assert "error" in result["messages"][0].content.lower()


# =============================================================================
# grade_generation
# =============================================================================


class TestGradeGenerationStep:
    def test_accepts_grounded_relevant_answer(self, base_state, fake_llm):
        base_state["messages"].append(AIMessage(content="Khufu built it."))
        result = grade_generation_step(base_state, fake_llm)
        assert result["metadata"]["generation_decision"] == "accept"
        assert result["metadata"]["hallucination_grounded"] is True
        assert result["metadata"]["answer_relevant"] is True

    def test_retries_on_hallucination(self, base_state):
        base_state["messages"].append(AIMessage(content="Fabricated answer."))
        base_state["retrieval_attempts"] = 1

        class HallucinatingLLM:
            def with_structured_output(self, schema):
                if schema == HallucinationCheck:

                    class Resp:
                        def invoke(self, prompt, **kwargs):
                            return HallucinationCheck(grounded="no", reasoning="Not grounded")

                    return Resp()
                from tests.conftest import FakeStructuredLLM
                return FakeStructuredLLM(schema)

        result = grade_generation_step(base_state, HallucinatingLLM())
        assert result["metadata"]["generation_decision"] == "retry"

    def test_accepts_hallucination_at_max_retries(self, base_state):
        base_state["messages"].append(AIMessage(content="Fabricated answer."))
        base_state["retrieval_attempts"] = MAX_RETRIEVAL_ATTEMPTS

        class HallucinatingLLM:
            def with_structured_output(self, schema):
                if schema == HallucinationCheck:

                    class Resp:
                        def invoke(self, prompt, **kwargs):
                            return HallucinationCheck(grounded="no", reasoning="Not grounded")

                    return Resp()
                from tests.conftest import FakeStructuredLLM
                return FakeStructuredLLM(schema)

        result = grade_generation_step(base_state, HallucinatingLLM())
        assert result["metadata"]["generation_decision"] == "accept"

    def test_retries_on_irrelevant_answer(self, base_state):
        base_state["messages"].append(AIMessage(content="Off topic answer."))
        base_state["retrieval_attempts"] = 1

        class IrrelevantLLM:
            def with_structured_output(self, schema):
                if schema == HallucinationCheck:

                    class Resp:
                        def invoke(self, prompt, **kwargs):
                            return HallucinationCheck(grounded="yes", reasoning="Grounded")

                    return Resp()
                if schema == AnswerRelevanceCheck:

                    class Resp2:
                        def invoke(self, prompt, **kwargs):
                            return AnswerRelevanceCheck(relevant="no", reasoning="Off topic")

                    return Resp2()
                from tests.conftest import FakeStructuredLLM
                return FakeStructuredLLM(schema)

        result = grade_generation_step(base_state, IrrelevantLLM())
        assert result["metadata"]["generation_decision"] == "retry"


class TestCheckGenerationQuality:
    def test_reads_accept_from_metadata(self):
        state = {"metadata": {"generation_decision": "accept"}}
        assert check_generation_quality(state) == "accept"

    def test_reads_retry_from_metadata(self):
        state = {"metadata": {"generation_decision": "retry"}}
        assert check_generation_quality(state) == "retry"

    def test_defaults_to_accept(self):
        state = {"metadata": {}}
        assert check_generation_quality(state) == "accept"

    def test_no_metadata_defaults_to_accept(self):
        state = {}
        assert check_generation_quality(state) == "accept"

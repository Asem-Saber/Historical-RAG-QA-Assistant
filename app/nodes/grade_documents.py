import logging

from langchain_openai import ChatOpenAI

from app.core.models import GradeDocuments, GradingResult
from app.core.prompts import GRADE_DOCUMENTS_PROMPT
from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)


def grade_documents_step(state: AgentState, llm: ChatOpenAI) -> dict:
    """Grade retrieved documents for relevance using LLM."""
    logger.info("NODE: grade_documents")

    question = get_latest_query(state["messages"])
    docs = state.get("documents", [])

    if not docs:
        logger.warning("No documents to grade, routing to rewrite_query")
        return {"routing_decision": "rewrite_query", "grading_results": []}

    context_text = "\n\n".join(
        f"[{i}] {doc.page_content[:1500]}" for i, doc in enumerate(docs, 1)
    )

    try:
        prompt = GRADE_DOCUMENTS_PROMPT.format(context=context_text, question=question)
        structured_llm = llm.with_structured_output(GradeDocuments)
        grading_response = structured_llm.invoke(prompt)

        is_relevant = grading_response.binary_score == "yes"
        logger.info(f"Grading: {grading_response.binary_score} — {grading_response.reasoning}")

        grading_result = GradingResult(
            document_id="retrieved_batch",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=grading_response.reasoning,
        )

    except Exception as e:
        logger.error(f"LLM grading failed: {e}, falling back to heuristic")
        is_relevant = len(docs) > 0
        grading_result = GradingResult(
            document_id="retrieved_batch",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=f"Fallback heuristic: {len(docs)} docs retrieved",
        )

    route = "generate_answer" if is_relevant else "rewrite_query"
    logger.info(f"Routing to: {route}")

    return {"routing_decision": route, "grading_results": [grading_result]}

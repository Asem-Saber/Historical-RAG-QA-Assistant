import logging
from typing import Literal

from langchain_openai import ChatOpenAI

from app.core.models import AnswerRelevanceCheck, HallucinationCheck
from app.core.prompts import ANSWER_RELEVANCE_PROMPT, HALLUCINATION_PROMPT
from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 2


def grade_generation_step(state: AgentState, llm: ChatOpenAI) -> dict:
    """Check if the generated answer is grounded and relevant."""
    logger.info("NODE: grade_generation")

    docs = state.get("documents", [])
    messages = state.get("messages", [])
    generation = messages[-1].content if messages else ""
    question = state.get("original_query") or get_latest_query(messages)
    retries = state.get("retrieval_attempts", 0)

    sources_text = "\n\n".join(doc.page_content for doc in docs)

    # Hallucination check
    hallucination_grounded = True
    hallucination_reasoning = ""
    try:
        structured_llm = llm.with_structured_output(HallucinationCheck)
        prompt = HALLUCINATION_PROMPT.format(sources=sources_text, generation=generation)
        result = structured_llm.invoke(prompt)
        hallucination_grounded = result.grounded == "yes"
        hallucination_reasoning = result.reasoning
        logger.info(f"Hallucination check: grounded={hallucination_grounded} — {hallucination_reasoning}")
    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        hallucination_reasoning = f"Check failed: {e}"

    if not hallucination_grounded:
        decision = "accept" if retries >= MAX_RETRIEVAL_ATTEMPTS else "retry"
        if decision == "accept":
            logger.warning("Hallucination detected but max retries reached, accepting")
        else:
            logger.info("Hallucination detected — retrying")

        return {
            "metadata": {
                **state.get("metadata", {}),
                "generation_decision": decision,
                "hallucination_grounded": hallucination_grounded,
                "hallucination_reasoning": hallucination_reasoning,
                "relevance_checked": False,
            },
        }

    # Answer relevance check
    answer_relevant = True
    relevance_reasoning = ""
    try:
        structured_llm = llm.with_structured_output(AnswerRelevanceCheck)
        prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, generation=generation)
        result = structured_llm.invoke(prompt)
        answer_relevant = result.relevant == "yes"
        relevance_reasoning = result.reasoning
        logger.info(f"Relevance check: relevant={answer_relevant} — {relevance_reasoning}")
    except Exception as e:
        logger.error(f"Relevance check failed: {e}")
        relevance_reasoning = f"Check failed: {e}"

    if not answer_relevant:
        decision = "accept" if retries >= MAX_RETRIEVAL_ATTEMPTS else "retry"
        if decision == "accept":
            logger.warning("Answer not relevant but max retries reached, accepting")
        else:
            logger.info("Answer not relevant — retrying")
    else:
        decision = "accept"
        logger.info("Generation passed all quality checks")

    return {
        "metadata": {
            **state.get("metadata", {}),
            "generation_decision": decision,
            "hallucination_grounded": hallucination_grounded,
            "hallucination_reasoning": hallucination_reasoning,
            "answer_relevant": answer_relevant,
            "relevance_reasoning": relevance_reasoning,
        },
    }


def check_generation_quality(state: AgentState) -> Literal["accept", "retry"]:
    """Read the decision stored by grade_generation_step."""
    return state.get("metadata", {}).get("generation_decision", "accept")

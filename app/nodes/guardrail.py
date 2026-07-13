import logging
from typing import Literal

from langchain_openai import ChatOpenAI

from app.core.models import GuardrailScoring
from app.core.prompts import GUARDRAIL_PROMPT
from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

GUARDRAIL_THRESHOLD = 60


def continue_after_guardrail(state: AgentState) -> Literal["continue", "out_of_scope"]:
    """Route based on guardrail score vs threshold."""
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        return "continue"

    logger.info(f"Guardrail score: {guardrail_result.score}, threshold: {GUARDRAIL_THRESHOLD}")
    return "continue" if guardrail_result.score >= GUARDRAIL_THRESHOLD else "out_of_scope"


def guardrail_step(state: AgentState, llm: ChatOpenAI) -> dict:
    """Validate whether the query is about Ancient Egypt."""
    logger.info("NODE: guardrail")

    query = get_latest_query(state["messages"])

    try:
        prompt = GUARDRAIL_PROMPT.format(question=query)
        structured_llm = llm.with_structured_output(GuardrailScoring)
        response = structured_llm.invoke(prompt)
        logger.info(f"Guardrail — Score: {response.score}, Reason: {response.reason}")

    except Exception as e:
        logger.error(f"Guardrail failed: {e}, using default")
        response = GuardrailScoring(score=50, reason=f"Validation failed: {str(e)}")

    return {"guardrail_result": response}

import logging

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.prompts import REWRITE_PROMPT
from app.core.state import AgentState

logger = logging.getLogger(__name__)


def rewrite_query_step(state: AgentState, llm: ChatOpenAI) -> dict:
    """Rewrite the query for better document retrieval."""
    logger.info("NODE: rewrite_query")

    original_question = state.get("original_query") or state["messages"][0].content

    try:
        prompt = REWRITE_PROMPT.format(question=original_question)
        rewritten = llm.invoke(prompt).content.strip()

        if not rewritten:
            raise ValueError("Empty rewrite")

        logger.info(f"Rewritten: '{original_question[:50]}...' -> '{rewritten[:50]}...'")

    except Exception as e:
        logger.error(f"Query rewrite failed: {e}, falling back to keyword expansion")
        rewritten = f"{original_question} ancient egypt pharaoh dynasty"

    return {
        "messages": [HumanMessage(content=rewritten)],
        "rewritten_query": rewritten,
    }

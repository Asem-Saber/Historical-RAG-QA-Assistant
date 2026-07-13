import logging

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.prompts import GENERATE_ANSWER_PROMPT
from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)


def generate_answer_step(state: AgentState, llm: ChatOpenAI) -> dict:
    """Generate a cited answer from retrieved documents."""
    logger.info("NODE: generate_answer")

    question = state.get("original_query") or get_latest_query(state["messages"])
    docs = state.get("documents", [])

    if not docs:
        answer = "I don't have enough information from the sources to answer this question."
        return {"messages": [AIMessage(content=answer)], "sources": []}

    context_text = "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )

    try:
        prompt = GENERATE_ANSWER_PROMPT.format(context=context_text, question=question)
        answer = llm.invoke(prompt).content
        logger.info(f"Generated answer: {len(answer)} characters")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = f"I encountered an error generating the answer: {str(e)}"

    sources = [
        {"citation": i, "content": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(docs, 1)
    ]

    return {"messages": [AIMessage(content=answer)], "sources": sources}

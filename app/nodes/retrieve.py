import logging

from langchain_core.messages import AIMessage

from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 2


def retrieve_step(state: AgentState) -> dict:
    """Emit tool calls for retrieval — ToolNode executes them."""
    logger.info("NODE: retrieve")

    messages = state["messages"]
    query = state.get("rewritten_query") or get_latest_query(messages)
    sub_queries = state.get("sub_queries", [query])
    current_attempts = state.get("retrieval_attempts", 0)

    updates: dict = {}
    if state.get("original_query") is None:
        updates["original_query"] = query

    if current_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        logger.warning(f"Max retrieval attempts ({MAX_RETRIEVAL_ATTEMPTS}) reached")
        fallback_msg = (
            f"I couldn't find relevant information after {MAX_RETRIEVAL_ATTEMPTS} attempts. "
            "Please try rephrasing your question with more specific terms about Ancient Egypt."
        )
        return {**updates, "messages": [AIMessage(content=fallback_msg)], "documents": []}

    new_attempt = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt
    logger.info(f"Retrieval attempt {new_attempt}/{MAX_RETRIEVAL_ATTEMPTS}")

    tool_calls = [
        {
            "id": f"retrieve_{new_attempt}_sq{i}",
            "name": "retrieve_documents",
            "args": {"query": sq},
        }
        for i, sq in enumerate(sub_queries)
    ]

    updates["messages"] = [AIMessage(content="", tool_calls=tool_calls)]
    return updates

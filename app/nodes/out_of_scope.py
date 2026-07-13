import logging
from langchain_core.messages import AIMessage
from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_RESPONSE = (
    "I can only help with questions about Ancient Egyptian history, civilization, "
    "archaeology, and culture — based on the Encyclopedia of Ancient Egypt.\n\n"
    "Your question: '{question}'\n\n"
    "This appears to be outside my domain. If you have a question about Ancient Egypt, "
    "I'd be happy to help!"
)


def out_of_scope_step(state: AgentState) -> dict:
    """Respond to queries outside the Ancient Egypt domain."""
    logger.info("NODE: out_of_scope")

    question = get_latest_query(state["messages"])
    response_text = OUT_OF_SCOPE_RESPONSE.format(question=question)

    return {"messages": [AIMessage(content=response_text)]}

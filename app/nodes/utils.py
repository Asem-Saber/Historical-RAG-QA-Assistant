from langchain_core.messages import AnyMessage, HumanMessage


def get_latest_query(messages: list[AnyMessage]) -> str:
    """Extract the latest user query from messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

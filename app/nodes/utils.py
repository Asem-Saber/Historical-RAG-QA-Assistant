from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage


def get_latest_query(messages: list[AnyMessage]) -> str:
    """Extract the latest user query from messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def get_latest_context(messages: list[AnyMessage]) -> str | None:
    """Extract the latest tool result content from messages."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list):
                return "\n\n".join(
                    item.page_content if hasattr(item, "page_content") else str(item)
                    for item in content
                )
            return str(content)
    return None

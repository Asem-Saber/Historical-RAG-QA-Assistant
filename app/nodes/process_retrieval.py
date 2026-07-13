import json
import logging

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from app.core.state import AgentState

logger = logging.getLogger(__name__)


def process_retrieval_step(state: AgentState) -> dict:
    """Extract documents from ToolMessages into state["documents"]."""
    logger.info("NODE: process_retrieval")

    documents = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            try:
                docs_data = json.loads(msg.content)
                documents.extend(
                    Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                    for d in docs_data
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse tool result: {e}")
        else:
            break

    seen = set()
    unique = []
    for doc in documents:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    logger.info(f"Extracted {len(unique)} unique documents from tool results")
    return {"documents": unique}

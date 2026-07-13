from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.core.models import GradingResult, GuardrailScoring


class AgentState(TypedDict):
    """State for the Agentic RAG workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: Optional[str]
    rewritten_query: Optional[str]
    sub_queries: List[str]
    retrieval_attempts: int
    guardrail_result: Optional[GuardrailScoring]
    routing_decision: Optional[str]
    documents: List[Document]
    grading_results: List[GradingResult]
    sources: List[dict]
    metadata: Dict[str, Any]

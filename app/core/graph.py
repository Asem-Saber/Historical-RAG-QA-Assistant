import logging
import sqlite3
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import settings
from app.core.state import AgentState
from app.nodes import (
    check_generation_quality,
    continue_after_guardrail,
    decompose_query_step,
    generate_answer_step,
    grade_documents_step,
    grade_generation_step,
    guardrail_step,
    out_of_scope_step,
    process_retrieval_step,
    retrieve_step,
    rewrite_query_step,
)

logger = logging.getLogger(__name__)

_INITIAL_STATE = {
    "retrieval_attempts": 0,
    "sub_queries": [],
    "guardrail_result": None,
    "routing_decision": None,
    "documents": [],
    "grading_results": [],
    "sources": [],
    "metadata": {},
    "original_query": None,
    "rewritten_query": None,
}

CHECKPOINT_DB = Path(settings.data_dir) / "checkpoints.sqlite"
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
checkpointer = SqliteSaver(_conn)


def build_graph(llm: ChatOpenAI, retriever_tool):
    """Build and compile the LangGraph workflow with SQLite checkpointer."""
    tools = [retriever_tool]
    workflow = StateGraph(AgentState)

    workflow.add_node("guardrail", lambda s: guardrail_step(s, llm))
    workflow.add_node("out_of_scope", out_of_scope_step)
    workflow.add_node("decompose_query", lambda s: decompose_query_step(s, llm))
    workflow.add_node("retrieve", retrieve_step)
    workflow.add_node("tool_retrieve", ToolNode(tools))
    workflow.add_node("process_retrieval", process_retrieval_step)
    workflow.add_node("grade_documents", lambda s: grade_documents_step(s, llm))
    workflow.add_node("rewrite_query", lambda s: rewrite_query_step(s, llm))
    workflow.add_node("generate_answer", lambda s: generate_answer_step(s, llm))
    workflow.add_node("grade_generation", lambda s: grade_generation_step(s, llm))

    workflow.add_edge(START, "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        continue_after_guardrail,
        {"continue": "decompose_query", "out_of_scope": "out_of_scope"},
    )
    workflow.add_edge("out_of_scope", END)

    workflow.add_edge("decompose_query", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        tools_condition,
        {"tools": "tool_retrieve", END: END},
    )

    workflow.add_edge("tool_retrieve", "process_retrieval")
    workflow.add_edge("process_retrieval", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        lambda s: s.get("routing_decision", "generate_answer"),
        {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
    )

    workflow.add_edge("rewrite_query", "retrieve")

    workflow.add_edge("generate_answer", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation",
        check_generation_quality,
        {"accept": END, "retry": "rewrite_query"},
    )

    return workflow.compile(checkpointer=checkpointer)


def invoke_graph(query: str, graph, thread_id: str | None = None) -> dict:
    """Run a query through the agentic RAG graph and return structured results."""
    if thread_id is None:
        thread_id = uuid.uuid4().hex

    config = {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()

    result = graph.invoke(
        {"messages": [HumanMessage(content=query)], **_INITIAL_STATE},
        config=config,
    )

    elapsed = time.perf_counter() - start
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "No answer generated."

    logger.info(
        "GRAPH query=%r thread=%s total=%.2fs docs=%d attempts=%d guardrail=%s",
        query[:80], thread_id, elapsed, len(result.get("documents", [])),
        result.get("retrieval_attempts", 0),
        result["guardrail_result"].score if result.get("guardrail_result") else "N/A",
    )

    return {
        "answer": answer,
        "documents": result.get("documents", []),
        "sources": result.get("sources", []),
        "sub_queries": result.get("sub_queries", []),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "rewritten_query": result.get("rewritten_query"),
        "guardrail_score": result["guardrail_result"].score if result.get("guardrail_result") else None,
        "execution_time": round(elapsed, 2),
        "thread_id": thread_id,
    }


def get_state(thread_id: str):
    """Retrieve the full saved state for a given thread."""
    config = {"configurable": {"thread_id": thread_id}}
    return checkpointer.get(config)

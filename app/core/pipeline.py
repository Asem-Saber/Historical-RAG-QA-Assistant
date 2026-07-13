import json
import logging
import threading
from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.callbacks import get_openai_callback
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.graph import build_graph, invoke_graph
from app.retrieval.hybrid import build_hybrid_retriever
from app.retrieval.tools import create_retriever_tool

logger = logging.getLogger(__name__)


class _MetricsAccumulator:
    """Thread-safe running totals for latency and token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.total_latency = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record(self, latency: float, input_tokens: int = 0, output_tokens: int = 0):
        with self._lock:
            self.request_count += 1
            self.total_latency += latency
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

    def snapshot(self) -> dict:
        with self._lock:
            n = self.request_count
            if n == 0:
                return {"requests": 0}
            return {
                "requests": n,
                "avg_latency": round(self.total_latency / n, 2),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "avg_input_tokens": round(self.total_input_tokens / n),
                "avg_output_tokens": round(self.total_output_tokens / n),
            }


metrics = _MetricsAccumulator()

# Shared components

embeddings = OllamaEmbeddings(
    model=settings.embedding_model,
    base_url=settings.embedding_endpoint,
)

vectorstore = Chroma(
    persist_directory=settings.chroma_path,
    embedding_function=embeddings,
)

llm = ChatOpenAI(
    model=settings.llm_model_id,
    base_url=settings.llm_endpoint,
    api_key=settings.api_key,
    temperature=settings.llm_temperature,
)

retriever = build_hybrid_retriever(vectorstore, k=settings.retriever_k)
reranker = CrossEncoder(settings.reranker_model_name, device="cpu")


# Agentic RAG graph

retriever_tool = create_retriever_tool(retriever, reranker)
graph = build_graph(llm, retriever_tool)


# Logging


def _log_metrics(query: str, result: dict, token_usage: dict):
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)

    metrics.record(result["execution_time"], input_tokens, output_tokens)

    logger.info(
        "METRIC query=%r total=%.2fs attempts=%d docs=%d sub_queries=%d "
        "guardrail_score=%s rewritten=%s "
        "input_tokens=%d output_tokens=%d total_tokens=%d",
        query[:80],
        result["execution_time"],
        result["retrieval_attempts"],
        len(result["documents"]),
        len(result["sub_queries"]),
        result["guardrail_score"],
        bool(result["rewritten_query"]),
        input_tokens,
        output_tokens,
        total_tokens,
    )

    snap = metrics.snapshot()
    logger.info(
        "METRIC_AVG requests=%d avg_latency=%.2fs avg_input_tokens=%d avg_output_tokens=%d",
        snap["requests"],
        snap.get("avg_latency", 0),
        snap.get("avg_input_tokens", 0),
        snap.get("avg_output_tokens", 0),
    )


def AncientEgyptRAG(query: str) -> dict:
    with get_openai_callback() as cb:
        result = invoke_graph(query, graph)

    token_usage = {
        "input_tokens": cb.prompt_tokens,
        "output_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    }
    _log_metrics(query, result, token_usage)

    return {
        "answer": result["answer"],
        "source_documents": result["documents"],
        "thread_id": result["thread_id"],
    }


def AncientEgyptRAGStream(query: str) -> Generator[str, None, None]:
    try:
        with get_openai_callback() as cb:
            result = invoke_graph(query, graph)

        token_usage = {
            "input_tokens": cb.prompt_tokens,
            "output_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
        }
        _log_metrics(query, result, token_usage)

        answer = result["answer"]
        docs = result["documents"]

        words = answer.split(" ")
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i : i + 3])
            if i > 0:
                chunk = " " + chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        sources = [
            {"citation": i, "content": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(docs, 1)
        ]
        yield f"data: {json.dumps({'type': 'sources', 'documents': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'thread_id': result['thread_id']})}\n\n"

    except Exception as e:
        logger.error("Stream pipeline failed", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

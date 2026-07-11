import json
import logging
import time
import threading
from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable

from app.core.config import settings
from app.retrieval.hybrid import build_hybrid_retriever
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class _MetricsAccumulator:
    """Thread-safe running totals for latency and token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.total_latency = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record(self, latency: float, input_tokens: int, output_tokens: int):
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
reranker = Reranker(settings.reranker_model_name, device="cpu")

# Query Decomposition

decomposition_template = """You are a question analyzer. Decide whether the user's question is \
simple (answerable from a single topic) or multi-hop (needs facts from \
multiple topics to answer).

If SIMPLE  → return: {{"sub_queries": ["<original question>"]}}
If MULTI-HOP → break it into 2-4 independent sub-queries that each \
target one specific fact, and return:
{{"sub_queries": ["sub-query 1", "sub-query 2", ...]}}

Return ONLY valid JSON, nothing else.

Question: {input}"""

prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)


@traceable(name="Decompose Query")
def decompose_query(query: str) -> list[str]:
    try:
        chain = prompt_decomposition | llm | JsonOutputParser()
        result = chain.invoke({"input": query})
        subs = result.get("sub_queries", [query])
        return subs if subs else [query]
    except Exception:
        logger.warning("Query decomposition failed, using original query", exc_info=True)
        return [query]


@traceable(name="Retrieve for Sub-queries", run_type="retriever")
def retrieve_for_subqueries(sub_queries: list[str]):
    seen = set()
    merged = []
    for sq in sub_queries:
        for doc in retriever.invoke(sq):
            doc_id = doc.page_content[:120]
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)
    return merged

# Citation formatting & generation

system_prompt = (
    "You are a helpful historian assistant. Use the numbered source excerpts "
    "below from the Encyclopedia of Ancient Egypt to answer the question.\n\n"
    "RULES:\n"
    "- Cite sources inline using [1], [2], etc. after each claim.\n"
    "- Combine citations when a claim draws from several sources: [1][3].\n"
    "- If no source answers the question, say you don't know.\n"
    "- Do NOT fabricate information beyond what the sources state.\n\n"
    "SOURCES:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


@traceable(name="Format with Citations")
def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )


@traceable(name="Generate Answer", run_type="llm")
def generate_answer(query: str, context: str) -> tuple[str, dict]:
    messages = prompt_template.invoke({"input": query, "context": context})
    response = llm.invoke(messages)
    token_usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage = {
            "input_tokens": response.usage_metadata.get("input_tokens", 0),
            "output_tokens": response.usage_metadata.get("output_tokens", 0),
            "total_tokens": response.usage_metadata.get("total_tokens", 0),
        }
    return response.content, token_usage


def stream_generate_answer(query: str, context: str, token_usage_out: dict | None = None) -> Generator[str, None, None]:
    messages = prompt_template.invoke({"input": query, "context": context})
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
        if token_usage_out is not None and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            token_usage_out["input_tokens"] = chunk.usage_metadata.get("input_tokens", 0)
            token_usage_out["output_tokens"] = chunk.usage_metadata.get("output_tokens", 0)
            token_usage_out["total_tokens"] = chunk.usage_metadata.get("total_tokens", 0)

# Orchestrators


def _log_metrics(query: str, timings: dict, docs_before_rerank: int, docs_after_rerank: int,
                  sub_queries: list[str], token_usage: dict | None = None):
    total = sum(timings.values())
    token_str = ""
    if token_usage:
        token_str = (
            f" input_tokens={token_usage.get('input_tokens', 0)}"
            f" output_tokens={token_usage.get('output_tokens', 0)}"
            f" total_tokens={token_usage.get('total_tokens', 0)}"
        )
    logger.info(
        "METRIC query=%r total=%.2fs decompose=%.2fs retrieve=%.2fs rerank=%.2fs format=%.2fs generate=%.2fs "
        "sub_queries=%d docs_retrieved=%d docs_after_rerank=%d%s",
        query[:80], total,
        timings.get("decompose", 0),
        timings.get("retrieve", 0),
        timings.get("rerank", 0),
        timings.get("format", 0),
        timings.get("generate", 0),
        len(sub_queries), docs_before_rerank, docs_after_rerank, token_str,
    )
    metrics.record(
        latency=total,
        input_tokens=token_usage.get("input_tokens", 0) if token_usage else 0,
        output_tokens=token_usage.get("output_tokens", 0) if token_usage else 0,
    )
    snap = metrics.snapshot()
    logger.info(
        "METRIC_AVG requests=%d avg_latency=%.2fs avg_input_tokens=%d avg_output_tokens=%d",
        snap["requests"], snap.get("avg_latency", 0),
        snap.get("avg_input_tokens", 0), snap.get("avg_output_tokens", 0),
    )


@traceable(name="RAG Pipeline")
def AncientEgyptRAG(query: str) -> dict:
    timings = {}

    t = time.perf_counter()
    sub_queries = decompose_query(query)
    timings["decompose"] = time.perf_counter() - t

    t = time.perf_counter()
    docs = retrieve_for_subqueries(sub_queries)
    timings["retrieve"] = time.perf_counter() - t
    docs_before = len(docs)

    t = time.perf_counter()
    docs = reranker.rerank(query, docs, top_n=5)
    timings["rerank"] = time.perf_counter() - t

    t = time.perf_counter()
    context = format_docs(docs)
    timings["format"] = time.perf_counter() - t

    t = time.perf_counter()
    answer, token_usage = generate_answer(query, context)
    timings["generate"] = time.perf_counter() - t

    _log_metrics(query, timings, docs_before, len(docs), sub_queries, token_usage)

    return {
        "answer": answer,
        "source_documents": docs,
    }


def AncientEgyptRAGStream(query: str) -> Generator[str, None, None]:
    try:
        timings = {}

        t = time.perf_counter()
        sub_queries = decompose_query(query)
        timings["decompose"] = time.perf_counter() - t

        t = time.perf_counter()
        docs = retrieve_for_subqueries(sub_queries)
        timings["retrieve"] = time.perf_counter() - t
        docs_before = len(docs)

        t = time.perf_counter()
        docs = reranker.rerank(query, docs, top_n=5)
        timings["rerank"] = time.perf_counter() - t

        t = time.perf_counter()
        context = format_docs(docs)
        timings["format"] = time.perf_counter() - t

        t = time.perf_counter()
        token_count = 0
        stream_token_usage = {}
        for token in stream_generate_answer(query, context, token_usage_out=stream_token_usage):
            token_count += 1
            yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"
        timings["generate"] = time.perf_counter() - t

        if not stream_token_usage:
            stream_token_usage = {"input_tokens": 0, "output_tokens": token_count, "total_tokens": token_count}

        sources = [
            {"citation": i, "content": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(docs, 1)
        ]
        yield f"data: {json.dumps({'type': 'sources', 'documents': sources})}\n\n"
        yield 'data: {"type": "done"}\n\n'

        _log_metrics(query, timings, docs_before, len(docs), sub_queries, stream_token_usage)
        logger.info("METRIC query=%r tokens_streamed=%d", query[:80], token_count)

    except Exception as e:
        logger.error("Stream pipeline failed", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

import json
from typing import Generator
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langsmith import traceable
from app.core.config import settings

# Shared components

embeddings = OllamaEmbeddings(
    model=settings.embedding_model,
    base_url=settings.embedding_endpoint,
)

vectorstore = Chroma(
    persist_directory=settings.chroma_path,
    embedding_function=embeddings,
)

reranker = CrossEncoder(settings.reranker_model_name, device="cpu")

llm = ChatOpenAI(
    model=settings.llm_model_id,
    base_url=settings.llm_endpoint,
    api_key=settings.api_key,
    temperature=settings.llm_temperature,
)

# hybrid Retrieval 
def hybrid_search(vs: Chroma, k: int= 5) -> EnsembleRetriever: 
    stored = vs.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(docs, k=k)
    vector = vs.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6],
    )

retriever = hybrid_search(vectorstore, k=settings.retriever_k)


# Query Decomposition
decomposition_template= """You are a question analyzer. Decide whether the user's question is \
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
    chain = prompt_decomposition | llm | JsonOutputParser()
    result = chain.invoke({"input": query})
    subs = result.get("sub_queries", [query])
    return subs if subs else [query]

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

# Re-ranking 
@traceable(name="Re-rank Documents", run_type="retriever")
def rerank_documents(query: str, docs, top_n: int = 5):
    if not docs:
        return docs
    scores = reranker.rank(query, [doc.page_content for doc in docs], top_k=top_n)
    return [docs[result["corpus_id"]] for result in scores]


# Citations 
@traceable(name="Format with Citations")
def format_docs(docs) -> str:
    """Extracts text from LangChain Document objects and joins them."""
    return "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )

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

@traceable(name="Generate Answer", run_type="llm")
def generate_answer(query: str, context: str) -> str:
    """Compiles the prompt and calls the LLM."""
    messages = prompt_template.invoke({"input": query, "context": context})
    response = llm.invoke(messages)
    return response.content


@traceable(name="RAG Pipeline")
def AncientEgyptRAG(query: str) -> dict:
    """The main orchestrator function"""
    sub_queries = decompose_query(query)
    docs = retrieve_for_subqueries(sub_queries)
    docs = rerank_documents(query, docs, top_n=5)
    context = format_docs(docs)
    answer = generate_answer(query, context)
    return {
        "answer": answer,
        "source_documents": docs,
    }


def stream_generate_answer(query: str, context: str) -> Generator[str, None, None]:
    """Streams the LLM response token by token."""
    messages = prompt_template.invoke({"input": query, "context": context})
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def AncientEgyptRAGStream(query: str) -> Generator[str, None, None]:
    """Streaming orchestrator — yields SSE events."""
    sub_queries = decompose_query(query)
    docs = retrieve_for_subqueries(sub_queries)
    docs = rerank_documents(query, docs, top_n=5)
    context = format_docs(docs)

    if len(sub_queries) > 1:
        yield f"data: {json.dumps({'type': 'decomposition', 'sub_queries': sub_queries})}\n\n"

    for token in stream_generate_answer(query, context):
        yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"

    sources = [
        {"citation": i, "content": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(docs, 1)
    ]
    yield f"data: {json.dumps({'type': 'sources', 'documents': sources})}\n\n"
    yield 'data: {"type": "done"}\n\n'
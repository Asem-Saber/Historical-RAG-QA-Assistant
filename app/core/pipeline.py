import json
from typing import Generator
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings


embeddings = OllamaEmbeddings(
    model=settings.embedding_model,
    base_url=settings.embedding_endpoint,
)

vectorstore = Chroma(
    persist_directory=settings.chroma_path,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})

llm = ChatOpenAI(
    model=settings.llm_model_id,
    base_url=settings.llm_endpoint,
    api_key=settings.api_key,
    temperature=settings.llm_temperature,
)

system_prompt = (
    "You are a helpful historian assistant. Use the following pieces of retrieved "
    "context from the Encyclopedia of Ancient Egypt to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


@traceable(name="1. Retrieve Context", run_type="retriever")
def retrieve_documents(query: str):
    """Fetches documents from ChromaDB."""
    return retriever.invoke(query)


@traceable(name="2. Format Documents")
def format_docs(docs) -> str:
    """Extracts text from LangChain Document objects and joins them."""
    return "\n\n".join(doc.page_content for doc in docs)


@traceable(name="3. Generate Answer", run_type="llm")
def generate_answer(query: str, context: str) -> str:
    """Compiles the prompt and calls the LLM."""
    messages = prompt_template.invoke({"input": query, "context": context})
    response = llm.invoke(messages)
    return response.content

def stream_generate_answer(query: str, context: str) -> Generator[str, None, None]:
    """Streams the LLM response token by token."""
    messages = prompt_template.invoke({"input": query, "context": context})
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


@traceable(name="Ancient Egypt RAG Pipeline")
def AncientEgyptRAG(query: str) -> dict:
    """The main orchestrator function (non-streaming)."""
    docs = retrieve_documents(query)
    context_string = format_docs(docs)
    answer = generate_answer(query, context_string)
    return {
        "answer": answer,
        "source_documents": docs,
    }


def AncientEgyptRAGStream(query: str) -> Generator[str, None, None]:
    """Streaming orchestrator — yields SSE events."""
    docs = retrieve_documents(query)
    context_string = format_docs(docs)

    for token in stream_generate_answer(query, context_string):
        yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"

    sources = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    yield f"data: {json.dumps({'type': 'sources', 'documents': sources})}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"
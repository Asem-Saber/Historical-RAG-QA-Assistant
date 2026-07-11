from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(vs: Chroma, k: int = 5) -> EnsembleRetriever:
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

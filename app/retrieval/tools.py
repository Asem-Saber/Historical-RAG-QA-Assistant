import json
import logging

from langchain_core.tools import tool
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def create_retriever_tool(
    retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    reranker_top_n: int = 4,
):
    """Create a retriever tool that wraps hybrid search + reranking."""

    @tool
    def retrieve_documents(query: str) -> str:
        """Search the Encyclopedia of Ancient Egypt and return relevant passages.

        Use this tool when the user asks about:
        - Ancient Egyptian history, pharaohs, or dynasties
        - Egyptian monuments, temples, or archaeological sites
        - Egyptian religion, mythology, or gods
        - Daily life, culture, or society in Ancient Egypt
        """
        logger.info(f"Tool: retrieve_documents — query: {query[:100]}...")

        seen = set()
        merged = []
        for doc in retriever.invoke(query):
            doc_id = doc.page_content[:120]
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)

        logger.info(f"Retrieved {len(merged)} unique documents")

        if not merged:
            return json.dumps([])

        scores = reranker.rank(
            query,
            [doc.page_content for doc in merged],
            top_k=min(reranker_top_n, len(merged)),
        )
        reranked = [merged[r["corpus_id"]] for r in scores]
        logger.info(f"Re-ranked to top {len(reranked)}")

        return json.dumps([
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in reranked
        ])

    return retrieve_documents

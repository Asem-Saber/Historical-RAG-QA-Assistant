"""Tests for app.retrieval — hybrid retriever and retriever tool."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# =============================================================================
# hybrid retriever
# =============================================================================


class TestBuildHybridRetriever:
    def test_builds_ensemble_with_correct_weights(self):
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["doc1 content", "doc2 content"],
            "metadatas": [{"source": "a"}, {"source": "b"}],
        }
        mock_vector_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_vector_retriever

        with patch("app.retrieval.hybrid.BM25Retriever") as mock_bm25_cls, \
             patch("app.retrieval.hybrid.EnsembleRetriever") as mock_ensemble_cls:
            mock_bm25 = MagicMock()
            mock_bm25_cls.from_documents.return_value = mock_bm25
            mock_ensemble = MagicMock()
            mock_ensemble_cls.return_value = mock_ensemble

            from app.retrieval.hybrid import build_hybrid_retriever

            result = build_hybrid_retriever(mock_vs, k=3)

        mock_ensemble_cls.assert_called_once_with(
            retrievers=[mock_bm25, mock_vector_retriever],
            weights=[0.4, 0.6],
        )
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})

    def test_passes_k_to_bm25(self):
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["text"],
            "metadatas": [{}],
        }
        mock_vs.as_retriever.return_value = MagicMock()

        with patch("app.retrieval.hybrid.BM25Retriever") as mock_bm25_cls, \
             patch("app.retrieval.hybrid.EnsembleRetriever"):
            mock_bm25_cls.from_documents.return_value = MagicMock()

            from app.retrieval.hybrid import build_hybrid_retriever

            build_hybrid_retriever(mock_vs, k=7)

            call_kwargs = mock_bm25_cls.from_documents.call_args
            assert call_kwargs.kwargs["k"] == 7

    def test_converts_stored_data_to_documents(self):
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["content A", "content B"],
            "metadatas": [{"key": "val"}, None],
        }
        mock_vs.as_retriever.return_value = MagicMock()

        with patch("app.retrieval.hybrid.BM25Retriever") as mock_bm25_cls, \
             patch("app.retrieval.hybrid.EnsembleRetriever"):
            mock_bm25_cls.from_documents.return_value = MagicMock()

            from app.retrieval.hybrid import build_hybrid_retriever

            build_hybrid_retriever(mock_vs, k=2)

            docs_passed = mock_bm25_cls.from_documents.call_args.args[0]
            assert len(docs_passed) == 2
            assert docs_passed[0].page_content == "content A"
            assert docs_passed[0].metadata == {"key": "val"}
            assert docs_passed[1].metadata == {}


# =============================================================================
# retriever tool
# =============================================================================


class TestCreateRetrieverTool:
    def test_retrieves_and_reranks(self):
        docs = [
            Document(page_content="Pyramids were built as tombs for pharaohs.", metadata={"source": "ch1"}),
            Document(page_content="The Nile floods annually.", metadata={"source": "ch2"}),
            Document(page_content="Hieroglyphs were a writing system.", metadata={"source": "ch3"}),
        ]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs

        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [
            {"corpus_id": 0, "score": 0.9},
            {"corpus_id": 2, "score": 0.7},
        ]

        from app.retrieval.tools import create_retriever_tool

        tool = create_retriever_tool(mock_retriever, mock_reranker, reranker_top_n=2)
        result_json = tool.invoke({"query": "pyramids"})
        result = json.loads(result_json)

        assert len(result) == 2
        assert result[0]["page_content"] == "Pyramids were built as tombs for pharaohs."
        assert result[1]["page_content"] == "Hieroglyphs were a writing system."

    def test_deduplicates_before_reranking(self):
        dup_doc = Document(page_content="Same content " * 20, metadata={})
        docs = [dup_doc, dup_doc, dup_doc]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs

        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [{"corpus_id": 0, "score": 0.9}]

        from app.retrieval.tools import create_retriever_tool

        tool = create_retriever_tool(mock_retriever, mock_reranker, reranker_top_n=3)
        result_json = tool.invoke({"query": "test"})
        result = json.loads(result_json)

        assert len(result) == 1
        # Reranker should have been called with only 1 unique document
        reranker_docs = mock_reranker.rank.call_args.args[1]
        assert len(reranker_docs) == 1

    def test_empty_retrieval_returns_empty_list(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        mock_reranker = MagicMock()

        from app.retrieval.tools import create_retriever_tool

        tool = create_retriever_tool(mock_retriever, mock_reranker)
        result_json = tool.invoke({"query": "nothing"})
        result = json.loads(result_json)

        assert result == []
        mock_reranker.rank.assert_not_called()

    def test_respects_reranker_top_n(self):
        docs = [
            Document(page_content=f"Doc {i} with enough content to be unique in dedup", metadata={})
            for i in range(10)
        ]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs

        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [
            {"corpus_id": i, "score": 1.0 - i * 0.1} for i in range(5)
        ]

        from app.retrieval.tools import create_retriever_tool

        tool = create_retriever_tool(mock_retriever, mock_reranker, reranker_top_n=5)
        tool.invoke({"query": "test"})

        _, kwargs = mock_reranker.rank.call_args
        assert kwargs["top_k"] == 5

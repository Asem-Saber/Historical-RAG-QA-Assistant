"""
Build the Chroma vector index from the Ancient Egypt markdown.

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --parse   # also re-parse the PDF first
"""

import argparse
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings
from app.ingestion.chunker import create_chunks


def build_vectorstore(chunked_docs):
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_endpoint,
    )
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=settings.chroma_path,
    )
    print(f"Vector store created at: {settings.chroma_path}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse", action="store_true", help="Re-parse the PDF first")
    args = parser.parse_args()

    if args.parse:
        from app.ingestion.parser import parse_pdf_to_markdown
        md_path = parse_pdf_to_markdown()
    else:
        md_path = Path(settings.data_dir) / "ancient_egypt.md"

    chunks = create_chunks(md_path)
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()

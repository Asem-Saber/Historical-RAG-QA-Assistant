"""
One-off script to create the Chroma vector database from the Ancient Egypt PDF.

Usage:
    python -m scripts.create_vectordb
"""

import re
from pathlib import Path

from llama_cloud import LlamaCloud
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings


def parse_pdf_to_markdown() -> Path:
    """Parse the Ancient Egypt PDF using LlamaParse and save as markdown."""
    llama_parser = LlamaCloud(api_key=settings.llamaparse_api_key)

    pdf_path = Path(settings.data_dir) / "ancient_egypt.pdf"
    md_path = Path(settings.data_dir) / "ancient_egypt.md"

    file_obj = llama_parser.files.create(file=str(pdf_path), purpose="parse")

    result = llama_parser.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        page_ranges={"target_pages": "20-437"},
        expand=["markdown_full"],
    )

    md_path.write_text(result.markdown_full or "")
    print(f"Markdown saved to: {md_path}")
    return md_path


def create_chunks(md_path: Path):
    """Read markdown, clean it, split by headers, then by character size."""
    md_text = md_path.read_text(encoding="utf-8")
    md_text = re.sub(r"ـ", "", md_text)
    md_text_standardized = re.sub(
        r"^\*\*(.*?)\*\*$", r"### \1", md_text, flags=re.MULTILINE
    )

    headers_to_split_on = [
        ("##", "Major_Entry"),
        ("###", "Minor_Entry"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    semantic_docs = markdown_splitter.split_text(md_text_standardized)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "(?<=\\. )", "،", " ", ""],
    )
    chunked_docs = text_splitter.split_documents(semantic_docs)

    for doc in chunked_docs:
        context_parts = []
        if "Major_Entry" in doc.metadata:
            clean_major = doc.metadata["Major_Entry"].replace("*", "").strip()
            context_parts.append(f"Subject: {clean_major}")
        if "Minor_Entry" in doc.metadata:
            clean_minor = doc.metadata["Minor_Entry"].replace("*", "").strip()
            context_parts.append(f"Term: {clean_minor}")
        context_prefix = " | ".join(context_parts)
        if context_prefix:
            doc.page_content = f"[{context_prefix}]\n\n{doc.page_content}"

    print(f"Successfully created {len(chunked_docs)} highly contextualized RAG chunks.")
    return chunked_docs


def build_vectorstore(chunked_docs):
    """Create Chroma vectorstore from chunked documents."""
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
    md_path = parse_pdf_to_markdown()
    chunks = create_chunks(md_path)
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()

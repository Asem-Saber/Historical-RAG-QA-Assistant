import re
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def create_chunks(md_path: Path, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Read markdown, clean it, split by headers, then by character size."""
    md_text = md_path.read_text(encoding="utf-8")
    md_text = re.sub(r"ـ", "", md_text)
    md_text = re.sub(
        r"^\*\*(.*?)\*\*$", r"### \1", md_text, flags=re.MULTILINE
    )

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "Major_Entry"),
            ("###", "Minor_Entry"),
        ],
        strip_headers=False,
    )
    semantic_docs = markdown_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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

    print(f"Created {len(chunked_docs)} chunks.")
    return chunked_docs

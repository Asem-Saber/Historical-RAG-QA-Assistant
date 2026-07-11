from pathlib import Path

from llama_cloud import LlamaCloud

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

"""Tests for app.ingestion.chunker — markdown chunking logic."""

from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def sample_md(tmp_path):
    """Create a temporary markdown file for chunking tests."""
    content = dedent("""\
        ## Pyramids

        ### Great Pyramid

        The Great Pyramid of Giza was built during the reign of Pharaoh Khufu
        (also known as Cheops) around 2560 BCE. It is the oldest and largest of
        the three pyramids in the Giza pyramid complex. The pyramid originally
        stood at 146.5 metres (481 feet) tall, making it the tallest man-made
        structure for over 3,800 years.

        The pyramid is estimated to contain 2.3 million stone blocks, each
        weighing an average of 2.5 tonnes. The precision of the construction
        is remarkable — the base is level to within just 2.1 centimetres.

        ### Step Pyramid

        The Step Pyramid at Saqqara was built for Pharaoh Djoser by his architect
        Imhotep around 2630 BCE. It is considered the earliest large-scale cut
        stone construction and the first pyramid built in ancient Egypt.

        ## Pharaohs

        ### Khufu

        Khufu was the second pharaoh of the Fourth Dynasty during the Old Kingdom.
        He is generally accepted as having commissioned the Great Pyramid of Giza.

        ### Ramesses II

        Ramesses II, also known as Ramesses the Great, was the third pharaoh of the
        Nineteenth Dynasty. He is often regarded as the most powerful and celebrated
        pharaoh of the Egyptian Empire. He ruled for approximately 66 years.
    """)
    md_file = tmp_path / "test_egypt.md"
    md_file.write_text(content, encoding="utf-8")
    return md_file


class TestCreateChunks:
    def test_returns_non_empty_chunks(self, sample_md):
        from app.ingestion.chunker import create_chunks

        chunks = create_chunks(sample_md)
        assert len(chunks) > 0

    def test_chunks_have_metadata(self, sample_md):
        from app.ingestion.chunker import create_chunks

        chunks = create_chunks(sample_md)
        for chunk in chunks:
            assert "Major_Entry" in chunk.metadata or "Minor_Entry" in chunk.metadata

    def test_chunk_content_has_context_prefix(self, sample_md):
        from app.ingestion.chunker import create_chunks

        chunks = create_chunks(sample_md)
        has_prefix = any("[Subject:" in c.page_content or "[Term:" in c.page_content for c in chunks)
        assert has_prefix

    def test_respects_chunk_size(self, sample_md):
        from app.ingestion.chunker import create_chunks

        chunk_size = 500
        chunks = create_chunks(sample_md, chunk_size=chunk_size, chunk_overlap=50)
        for chunk in chunks:
            # Allow some tolerance for the context prefix added after splitting
            assert len(chunk.page_content) <= chunk_size + 200

    def test_smaller_chunks_produces_more_documents(self, sample_md):
        from app.ingestion.chunker import create_chunks

        large = create_chunks(sample_md, chunk_size=2000, chunk_overlap=100)
        small = create_chunks(sample_md, chunk_size=200, chunk_overlap=50)
        assert len(small) > len(large)

    def test_handles_arabic_tatweel_cleanup(self, tmp_path):
        from app.ingestion.chunker import create_chunks

        content = "## Test\n\n### Entry\n\nSome text with tatweel ـ character in it.\n"
        md_file = tmp_path / "tatweel.md"
        md_file.write_text(content, encoding="utf-8")

        chunks = create_chunks(md_file)
        for chunk in chunks:
            assert "ـ" not in chunk.page_content

    def test_bold_to_header_conversion(self, tmp_path):
        from app.ingestion.chunker import create_chunks

        content = "## Main\n\n**Bold Entry**\n\nSome content under the bold entry.\n"
        md_file = tmp_path / "bold.md"
        md_file.write_text(content, encoding="utf-8")

        chunks = create_chunks(md_file)
        assert len(chunks) >= 1

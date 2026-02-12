"""
Tests for knowledgebase PDF extraction with OCR fallback.

The extraction pipeline in _extract_title_and_text() for PDFs:
1. Try pypdf (text extraction)
2. If pypdf yields < 100 chars, fall back to pymupdf with per-page OCR
3. If pypdf throws, fall back to pymupdf entirely
4. pymupdf OCR triggers per-page when get_text("text") yields < 20 chars

Related: _extract_pdf_with_pymupdf() and _PYMUPDF_MAX_OCR_PAGES
"""

import pytest
from unittest.mock import patch, MagicMock, call
from typing import List, Optional


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _make_pypdf_reader(pages_text: List[str], metadata_title: Optional[str] = None):
    """Mock pypdf.PdfReader."""
    reader = MagicMock()
    pages = []
    for t in pages_text:
        p = MagicMock()
        p.extract_text.return_value = t
        pages.append(p)
    reader.pages = pages

    meta = MagicMock()
    meta.title = metadata_title
    reader.metadata = meta
    return reader


def _make_fitz_doc(pages_text: List[str], ocr_text: Optional[List[str]] = None):
    """Mock fitz (pymupdf) document.
    pages_text: what get_text("text") returns per page
    ocr_text: what get_text("ocr") returns per page (defaults to empty)
    """
    doc = MagicMock()
    doc.__len__ = lambda self: len(pages_text)

    mock_pages = []
    for i, t in enumerate(pages_text):
        page = MagicMock()
        ocr_val = (ocr_text[i] if ocr_text and i < len(ocr_text) else "")

        def make_get_text(text_val, ocr_val):
            def get_text(mode="text"):
                if mode == "ocr":
                    return ocr_val
                return text_val
            return get_text

        page.get_text = make_get_text(t, ocr_val)
        mock_pages.append(page)

    doc.__getitem__ = lambda self, idx: mock_pages[idx]
    doc.close = MagicMock()
    return doc


def _patch_pypdf(reader):
    """Context manager to patch pypdf inline import."""
    mock_mod = MagicMock()
    mock_mod.PdfReader.return_value = reader
    return patch.dict("sys.modules", {"pypdf": mock_mod})


def _patch_fitz(doc):
    """Context manager to patch fitz (pymupdf) inline import."""
    mock_mod = MagicMock()
    mock_mod.open.return_value = doc
    return patch.dict("sys.modules", {"fitz": mock_mod})


def _patch_tesseract_available(available=True):
    """Context manager to patch _TESSERACT_AVAILABLE flag."""
    return patch("app.services.knowledgebase_service._TESSERACT_AVAILABLE", available)


# =============================================================================
# TEXT PDF — pypdf works, no OCR fallback
# =============================================================================

class TestTextPdfExtraction:

    @pytest.mark.asyncio
    async def test_basic_text_extraction(self):
        """pypdf extracts enough text → no pymupdf fallback."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader(
            ["Page one has plenty of content here for testing." * 3,
             "Page two also has content."],
            metadata_title="My Document",
        )
        with _patch_pypdf(reader):
            title, text = await _extract_title_and_text("/fake.pdf", "application/pdf")

        assert title == "My Document"
        assert "Page one" in text
        assert "Page two" in text

    @pytest.mark.asyncio
    async def test_title_from_metadata(self):
        reader = _make_pypdf_reader(
            ["Enough content to pass the 100 char threshold. " * 3],
            metadata_title="  Metadata Title  ",
        )
        from app.services.knowledgebase_service import _extract_title_and_text

        with _patch_pypdf(reader):
            title, _ = await _extract_title_and_text("/f.pdf", "application/pdf")

        assert title == "Metadata Title"

    @pytest.mark.asyncio
    async def test_title_fallback_to_first_line(self):
        reader = _make_pypdf_reader(
            ["  \nActual Title Line\n" + "Body text. " * 20],
            metadata_title="",
        )
        from app.services.knowledgebase_service import _extract_title_and_text

        with _patch_pypdf(reader):
            title, _ = await _extract_title_and_text("/f.pdf", "application/pdf")

        assert title == "Actual Title Line"


# =============================================================================
# SCANNED PDF — pypdf empty, pymupdf OCR fallback
# =============================================================================

class TestOcrFallback:

    @pytest.mark.asyncio
    async def test_ocr_fallback_when_pypdf_empty(self):
        """pypdf returns < 100 chars → pymupdf fallback is invoked."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader(["", "", ""])
        fitz_doc = _make_fitz_doc(
            pages_text=["", "", ""],
            ocr_text=["OCR Chapter 1 content here", "OCR page 2", "OCR page 3"],
        )

        with _patch_pypdf(reader), _patch_fitz(fitz_doc), _patch_tesseract_available():
            title, text = await _extract_title_and_text("/scanned.pdf", "application/pdf")

        assert "OCR Chapter 1" in text
        assert "OCR page 2" in text

    @pytest.mark.asyncio
    async def test_ocr_fallback_when_pypdf_throws(self):
        """pypdf raises → pymupdf fallback is invoked."""
        from app.services.knowledgebase_service import _extract_title_and_text

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.side_effect = Exception("bad pypdf")

        fitz_doc = _make_fitz_doc(
            pages_text=["Full text from pymupdf. " * 10],
        )

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}), _patch_fitz(fitz_doc):
            title, text = await _extract_title_and_text("/bad.pdf", "application/pdf")

        assert "Full text from pymupdf" in text

    @pytest.mark.asyncio
    async def test_ocr_per_page_threshold(self):
        """Pages with < 20 chars of text trigger per-page OCR in pymupdf."""
        from app.services.knowledgebase_service import _extract_pdf_with_pymupdf

        fitz_doc = _make_fitz_doc(
            pages_text=["Short", "This page has plenty of normal text content here."],
            ocr_text=["OCR recovered full content for page 0", ""],
        )

        with _patch_fitz(fitz_doc), _patch_tesseract_available():
            text, title = _extract_pdf_with_pymupdf("/test.pdf", None)

        # Page 0: "Short" is < 20 chars, so OCR should be used
        assert "OCR recovered" in text
        # Page 1: >= 20 chars, normal text kept
        assert "plenty of normal text" in text

    @pytest.mark.asyncio
    async def test_title_from_ocr_content(self):
        """Title derived from OCR'd first page text."""
        from app.services.knowledgebase_service import _extract_pdf_with_pymupdf

        fitz_doc = _make_fitz_doc(
            pages_text=[""],
            ocr_text=["1984 by George Orwell\nChapter 1\nIt was a bright cold day..."],
        )

        with _patch_fitz(fitz_doc), _patch_tesseract_available():
            text, title = _extract_pdf_with_pymupdf("/1984.pdf", None)

        assert title == "1984 by George Orwell"


# =============================================================================
# MIXED PDF
# =============================================================================

class TestMixedPdf:

    @pytest.mark.asyncio
    async def test_mixed_text_and_scanned_pages(self):
        """pymupdf handles mix: text pages kept, empty pages get OCR."""
        from app.services.knowledgebase_service import _extract_pdf_with_pymupdf

        fitz_doc = _make_fitz_doc(
            pages_text=[
                "Chapter 1: This is normal extracted text with enough content.",
                "",  # scanned page
                "Chapter 3: Also normal text.",
                "tiny",  # < 20 chars → OCR
            ],
            ocr_text=[
                "",
                "Chapter 2 recovered via OCR scanning",
                "",
                "Chapter 4 full OCR text recovered here",
            ],
        )

        with _patch_fitz(fitz_doc), _patch_tesseract_available():
            text, _ = _extract_pdf_with_pymupdf("/mixed.pdf", None)

        assert "Chapter 1" in text
        assert "Chapter 2 recovered" in text
        assert "Chapter 3" in text
        assert "Chapter 4 full OCR" in text


# =============================================================================
# CORRUPTED / EMPTY
# =============================================================================

class TestCorruptedPdf:

    @pytest.mark.asyncio
    async def test_both_extractors_fail(self):
        """Both pypdf and pymupdf fail → returns (None, '') via text fallback."""
        from app.services.knowledgebase_service import _extract_title_and_text

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.side_effect = Exception("corrupt")

        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = Exception("also corrupt")

        with patch.dict("sys.modules", {"pypdf": mock_pypdf, "fitz": mock_fitz}):
            title, text = await _extract_title_and_text("/corrupt.pdf", "application/pdf")

        # Falls through to text-file fallback; no crash
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_zero_pages_pdf(self):
        """PDF with zero pages → empty text."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader([])

        with _patch_pypdf(reader):
            title, text = await _extract_title_and_text("/empty.pdf", "application/pdf")

        # 0 pages, < 100 chars → pymupdf fallback triggered but may also return empty
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_page_extract_raises(self):
        """Individual page failure in pypdf is skipped."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader(["Good content. " * 20])
        # Make first page raise, but we need enough pages
        bad_page = MagicMock()
        bad_page.extract_text.side_effect = Exception("bad")
        good_page = MagicMock()
        good_page.extract_text.return_value = "Good content here for testing. " * 5
        reader.pages = [bad_page, good_page]

        with _patch_pypdf(reader):
            title, text = await _extract_title_and_text("/partial.pdf", "application/pdf")

        assert "Good content" in text


# =============================================================================
# PAGE LIMIT
# =============================================================================

class TestPageLimit:

    def test_pymupdf_respects_max_pages(self):
        """_PYMUPDF_MAX_OCR_PAGES limits how many pages are processed."""
        from app.services.knowledgebase_service import _extract_pdf_with_pymupdf, _PYMUPDF_MAX_OCR_PAGES

        num = _PYMUPDF_MAX_OCR_PAGES + 50
        fitz_doc = _make_fitz_doc(
            pages_text=[f"Page {i}" for i in range(num)],
        )
        # Override __len__ to report full count
        fitz_doc.__len__ = lambda self: num

        with _patch_fitz(fitz_doc):
            text, _ = _extract_pdf_with_pymupdf("/huge.pdf", None)

        # Should NOT contain pages beyond the limit
        assert f"Page {_PYMUPDF_MAX_OCR_PAGES + 10}" not in text
        # Should contain pages within limit
        assert "Page 0" in text

    def test_max_pages_constant_is_500(self):
        from app.services.knowledgebase_service import _PYMUPDF_MAX_OCR_PAGES
        assert _PYMUPDF_MAX_OCR_PAGES == 500


# =============================================================================
# TITLE EXTRACTION EDGE CASES
# =============================================================================

class TestTitleEdgeCases:

    @pytest.mark.asyncio
    async def test_short_lines_skipped(self):
        """Lines < 3 chars are not used as title."""
        reader = _make_pypdf_reader(
            ["ab\nReal Title Here\n" + "Body. " * 30],
            metadata_title=None,
        )
        from app.services.knowledgebase_service import _extract_title_and_text

        with _patch_pypdf(reader):
            title, _ = await _extract_title_and_text("/f.pdf", "application/pdf")

        assert title == "Real Title Here"

    @pytest.mark.asyncio
    async def test_long_lines_skipped(self):
        """Lines > 140 chars are skipped as title candidates."""
        long_line = "A" * 200
        reader = _make_pypdf_reader(
            [f"{long_line}\nShort Title\n" + "Body. " * 30],
            metadata_title=None,
        )
        from app.services.knowledgebase_service import _extract_title_and_text

        with _patch_pypdf(reader):
            title, _ = await _extract_title_and_text("/f.pdf", "application/pdf")

        assert title == "Short Title"


# =============================================================================
# NON-PDF SANITY CHECKS
# =============================================================================

class TestNonPdf:

    @pytest.mark.asyncio
    async def test_plain_text_file(self, tmp_path):
        from app.services.knowledgebase_service import _extract_title_and_text

        f = tmp_path / "readme.txt"
        f.write_text("Document Title\n\nBody paragraph here.", encoding="utf-8")

        title, text = await _extract_title_and_text(str(f), "text/plain")
        assert title == "Document Title"
        assert "Body paragraph" in text

    @pytest.mark.asyncio
    async def test_empty_text_file(self, tmp_path):
        from app.services.knowledgebase_service import _extract_title_and_text

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        title, text = await _extract_title_and_text(str(f), "text/plain")
        assert title is None
        assert text == ""

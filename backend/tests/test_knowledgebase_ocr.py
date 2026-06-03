"""
Tests for knowledgebase PDF extraction with OCR fallback.

The extraction pipeline in _extract_title_and_text() for PDFs:
1. Try pypdf (text extraction)
2. If pypdf yields < 100 chars, fall back to pypdfium2 with per-page OCR
3. If pypdf throws, fall back to pypdfium2 entirely
4. OCR triggers per-page when the pypdfium2 text layer yields < 20 chars

OCR uses pypdfium2 (Apache-2.0/BSD-3) to render + pytesseract (Apache-2.0) to
recognise — both permissively licensed (replacing the former AGPL pymupdf path).

Related: _extract_pdf_with_ocr(), _pdfium_page_text(), _ocr_pdfium_page() and
_OCR_MAX_PAGES.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Optional


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


class _FakePdfiumDoc:
    """Lightweight fake of a pypdfium2 PdfDocument.

    Page "handles" are simply their integer index, so the patched
    _pdfium_page_text / _ocr_pdfium_page seams can map index -> content.
    """

    def __init__(self, num_pages: int):
        self._n = num_pages
        self.closed = False

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return idx

    def close(self):
        self.closed = True


def _patch_pypdf(reader):
    """Context manager to patch pypdf inline import."""
    mock_mod = MagicMock()
    mock_mod.PdfReader.return_value = reader
    return patch.dict("sys.modules", {"pypdf": mock_mod})


def _patch_pdfium(doc):
    """Patch the pypdfium2 inline import so PdfDocument(path) -> doc."""
    mock_mod = MagicMock()
    mock_mod.PdfDocument.return_value = doc
    return patch.dict("sys.modules", {"pypdfium2": mock_mod})


def _patch_page_text(text_map: Dict[int, str]):
    """Patch _pdfium_page_text to return per-page text-layer content."""
    return patch(
        "app.services.knowledgebase_service._pdfium_page_text",
        side_effect=lambda page: text_map.get(page, ""),
    )


def _patch_page_ocr(ocr_map: Dict[int, str]):
    """Patch _ocr_pdfium_page to return per-page OCR content."""
    return patch(
        "app.services.knowledgebase_service._ocr_pdfium_page",
        side_effect=lambda page: ocr_map.get(page, ""),
    )


def _patch_tesseract_available(available=True):
    """Context manager to patch _TESSERACT_AVAILABLE flag."""
    return patch("app.services.knowledgebase_service._TESSERACT_AVAILABLE", available)


# =============================================================================
# TEXT PDF — pypdf works, no OCR fallback
# =============================================================================


class TestTextPdfExtraction:

    @pytest.mark.asyncio
    async def test_basic_text_extraction(self):
        """pypdf extracts enough text → no OCR fallback."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader(
            [
                "Page one has plenty of content here for testing." * 3,
                "Page two also has content.",
            ],
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
# SCANNED PDF — pypdf empty, pypdfium2 OCR fallback
# =============================================================================


class TestOcrFallback:

    @pytest.mark.asyncio
    async def test_ocr_fallback_when_pypdf_empty(self):
        """pypdf returns < 100 chars → pypdfium2 OCR fallback is invoked."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader(["", "", ""])
        doc = _FakePdfiumDoc(3)

        with _patch_pypdf(reader), _patch_pdfium(doc), _patch_page_text(
            {0: "", 1: "", 2: ""}
        ), _patch_page_ocr(
            {0: "OCR Chapter 1 content here", 1: "OCR page 2", 2: "OCR page 3"}
        ), _patch_tesseract_available():
            title, text = await _extract_title_and_text(
                "/scanned.pdf", "application/pdf"
            )

        assert "OCR Chapter 1" in text
        assert "OCR page 2" in text

    @pytest.mark.asyncio
    async def test_ocr_fallback_when_pypdf_throws(self):
        """pypdf raises → pypdfium2 fallback is invoked."""
        from app.services.knowledgebase_service import _extract_title_and_text

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.side_effect = Exception("bad pypdf")

        doc = _FakePdfiumDoc(1)

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}), _patch_pdfium(
            doc
        ), _patch_page_text({0: "Full text from the pdf. " * 10}):
            title, text = await _extract_title_and_text("/bad.pdf", "application/pdf")

        assert "Full text from the pdf" in text

    @pytest.mark.asyncio
    async def test_ocr_per_page_threshold(self):
        """Pages with < 20 chars of text-layer trigger per-page OCR."""
        from app.services.knowledgebase_service import _extract_pdf_with_ocr

        doc = _FakePdfiumDoc(2)

        with _patch_pdfium(doc), _patch_page_text(
            {0: "Short", 1: "This page has plenty of normal text content here."}
        ), _patch_page_ocr(
            {0: "OCR recovered full content for page 0", 1: ""}
        ), _patch_tesseract_available():
            text, title = _extract_pdf_with_ocr("/test.pdf", None)

        # Page 0: "Short" is < 20 chars, so OCR should be used
        assert "OCR recovered" in text
        # Page 1: >= 20 chars, normal text kept
        assert "plenty of normal text" in text

    @pytest.mark.asyncio
    async def test_title_from_ocr_content(self):
        """Title derived from OCR'd first page text."""
        from app.services.knowledgebase_service import _extract_pdf_with_ocr

        doc = _FakePdfiumDoc(1)

        with _patch_pdfium(doc), _patch_page_text({0: ""}), _patch_page_ocr(
            {0: "1984 by George Orwell\nChapter 1\nIt was a bright cold day..."}
        ), _patch_tesseract_available():
            text, title = _extract_pdf_with_ocr("/1984.pdf", None)

        assert title == "1984 by George Orwell"


# =============================================================================
# MIXED PDF
# =============================================================================


class TestMixedPdf:

    @pytest.mark.asyncio
    async def test_mixed_text_and_scanned_pages(self):
        """Mix: text pages kept, empty / tiny pages get OCR."""
        from app.services.knowledgebase_service import _extract_pdf_with_ocr

        doc = _FakePdfiumDoc(4)
        text_map = {
            0: "Chapter 1: This is normal extracted text with enough content.",
            1: "",  # scanned page
            2: "Chapter 3: Also normal text.",
            3: "tiny",  # < 20 chars → OCR
        }
        ocr_map = {
            0: "",
            1: "Chapter 2 recovered via OCR scanning",
            2: "",
            3: "Chapter 4 full OCR text recovered here",
        }

        with _patch_pdfium(doc), _patch_page_text(text_map), _patch_page_ocr(
            ocr_map
        ), _patch_tesseract_available():
            text, _ = _extract_pdf_with_ocr("/mixed.pdf", None)

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
        """Both pypdf and pypdfium2 fail → returns (None, '') via text fallback."""
        from app.services.knowledgebase_service import _extract_title_and_text

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.side_effect = Exception("corrupt")

        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.side_effect = Exception("also corrupt")

        with patch.dict("sys.modules", {"pypdf": mock_pypdf, "pypdfium2": mock_pdfium}):
            title, text = await _extract_title_and_text(
                "/corrupt.pdf", "application/pdf"
            )

        # Falls through to text-file fallback; no crash
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_zero_pages_pdf(self):
        """PDF with zero pages → empty text."""
        from app.services.knowledgebase_service import _extract_title_and_text

        reader = _make_pypdf_reader([])
        doc = _FakePdfiumDoc(0)

        with _patch_pypdf(reader), _patch_pdfium(doc):
            title, text = await _extract_title_and_text("/empty.pdf", "application/pdf")

        # 0 pages, < 100 chars → OCR fallback triggered but also returns empty
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
            title, text = await _extract_title_and_text(
                "/partial.pdf", "application/pdf"
            )

        assert "Good content" in text


# =============================================================================
# PAGE LIMIT
# =============================================================================


class TestPageLimit:

    def test_ocr_respects_max_pages(self):
        """_OCR_MAX_PAGES limits how many pages are processed."""
        from app.services.knowledgebase_service import (
            _extract_pdf_with_ocr,
            _OCR_MAX_PAGES,
        )

        num = _OCR_MAX_PAGES + 50
        doc = _FakePdfiumDoc(num)
        # >= 20 chars so OCR is never consulted; we only verify the page cap.
        text_map = {i: f"Page {i} has enough text to avoid ocr." for i in range(num)}

        with _patch_pdfium(doc), _patch_page_text(text_map), _patch_tesseract_available(
            False
        ):
            text, _ = _extract_pdf_with_ocr("/huge.pdf", None)

        # Should NOT contain pages beyond the limit
        assert f"Page {_OCR_MAX_PAGES + 10}" not in text
        # Should contain pages within limit
        assert "Page 0" in text

    def test_max_pages_constant_is_500(self):
        from app.services.knowledgebase_service import _OCR_MAX_PAGES

        assert _OCR_MAX_PAGES == 500


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

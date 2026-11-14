"""Unit tests for document text extraction service (PDF and TXT)."""

import io
import pytest
from pypdf import PdfReader, PdfWriter

from app.services.extraction_service import extraction_service
from app.utils.exceptions import (
    DocumentProcessingException,
    EmptyFileException,
    UnsupportedFileTypeException,
)


def _generate_valid_pdf_bytes(text_lines: list[str]) -> bytes:
    """Generate in-memory valid PDF bytes containing text."""
    # Construct base valid PDF structure with Text stream
    content_lines = " ".join(f"({line}) Tj T*" for line in text_lines)
    stream_content = f"BT /F1 12 Tf 40 100 Td {content_lines} ET".encode("latin-1")
    stream_len = len(stream_content)

    pdf_template = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
        b"4 0 obj <</Length " + str(stream_len).encode("ascii") + b">> stream\n"
        + stream_content + b"\nendstream endobj\n"
        b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000224 00000 n \n"
        b"0000000330 00000 n \n"
        b"trailer <</Size 6 /Root 1 0 R>>\n"
        b"startxref\n405\n%%EOF\n"
    )

    # Use PdfWriter to produce a standardized PDF stream
    reader = PdfReader(io.BytesIO(pdf_template))
    writer = PdfWriter()
    writer.append(reader)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestDocumentExtraction:
    """Test suite for extraction of TXT and PDF documents."""

    def test_extract_txt_utf8(self):
        """Test extraction from standard UTF-8 encoded text."""
        content = "PalmMind AI assignment.\nLine 2 text with regular characters."
        file_bytes = content.encode("utf-8")
        extracted = extraction_service.extract_text(file_bytes=file_bytes, filename="test.txt")
        assert "PalmMind AI assignment." in extracted
        assert "Line 2 text" in extracted

    def test_extract_txt_latin1_encoding(self):
        """Test extraction from Latin-1 / Windows-1252 encoded text."""
        content = "Candidate Résumé with café and naïve spelling."
        file_bytes = content.encode("latin-1")
        extracted = extraction_service.extract_text(file_bytes=file_bytes, filename="resume.txt")
        assert "Résumé" in extracted
        assert "café" in extracted

    def test_extract_txt_utf8_sig_bom(self):
        """Test extraction from UTF-8 text with Byte Order Mark (BOM)."""
        content = "UTF-8 BOM file content."
        file_bytes = content.encode("utf-8-sig")
        extracted = extraction_service.extract_text(file_bytes=file_bytes, filename="bom.txt")
        assert extracted == "UTF-8 BOM file content."

    def test_extract_txt_crlf_normalization(self):
        """Test that Windows CRLF line endings are normalized to LF."""
        content = "Paragraph 1\r\n\r\nParagraph 2\r\nLine 2"
        file_bytes = content.encode("utf-8")
        extracted = extraction_service.extract_text(file_bytes=file_bytes, filename="crlf.txt")
        assert "\r" not in extracted
        assert "Paragraph 1\n\nParagraph 2\nLine 2" == extracted

    def test_extract_pdf_valid(self):
        """Test text extraction from a valid PDF document."""
        pdf_bytes = _generate_valid_pdf_bytes(["Welcome to PalmMind AI", "Interview System"])
        extracted = extraction_service.extract_text(file_bytes=pdf_bytes, filename="document.pdf")
        assert "PalmMind" in extracted
        assert len(extracted) > 0

    def test_reject_unsupported_file_extension(self):
        """Test that unsupported formats (.exe, .docx, .png) raise UnsupportedFileTypeException."""
        with pytest.raises(UnsupportedFileTypeException) as exc_info:
            extraction_service.extract_text(file_bytes=b"dummy binary", filename="malicious.exe")
        assert exc_info.value.status_code == 400
        assert "Unsupported file type 'exe'" in str(exc_info.value.message)

        with pytest.raises(UnsupportedFileTypeException):
            extraction_service.extract_text(file_bytes=b"dummy docx", filename="resume.docx")

    def test_empty_txt_file_zero_bytes(self):
        """Test that a 0-byte file raises EmptyFileException."""
        with pytest.raises(EmptyFileException) as exc_info:
            extraction_service.extract_text(file_bytes=b"", filename="empty.txt")
        assert exc_info.value.status_code == 400
        assert "empty.txt" in exc_info.value.message

    def test_empty_txt_file_whitespace_only(self):
        """Test that a file containing only whitespace raises EmptyFileException."""
        with pytest.raises(EmptyFileException):
            extraction_service.extract_text(file_bytes=b"   \n\n\t  \r\n  ", filename="spaces.txt")

    def test_empty_pdf_without_pages(self):
        """Test that a PDF with zero pages raises EmptyFileException."""
        writer = PdfWriter()
        buf = io.BytesIO()
        writer.write(buf)
        with pytest.raises(EmptyFileException):
            extraction_service.extract_text(file_bytes=buf.getvalue(), filename="blank.pdf")

    def test_corrupt_pdf_file_raises_document_processing_exception(self):
        """Test that corrupted non-PDF bytes with a .pdf extension raise DocumentProcessingException."""
        corrupted_bytes = b"This is definitely not a valid PDF header or format!"
        with pytest.raises(DocumentProcessingException) as exc_info:
            extraction_service.extract_text(file_bytes=corrupted_bytes, filename="corrupt.pdf")
        assert exc_info.value.status_code == 422
        assert "Failed to parse PDF document" in exc_info.value.message

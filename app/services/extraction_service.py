"""Document text extraction service for PDF and TXT files."""

import io
import logging
import os
import re
from typing import List, Optional

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.config import get_settings
from app.utils.exceptions import (
    DocumentProcessingException,
    EmptyFileException,
    UnsupportedFileTypeException,
)

logger = logging.getLogger("palmind.extraction")


class ExtractionService:
    """Service responsible for extracting clean textual content from supported file types."""

    def __init__(self, allowed_extensions: Optional[List[str]] = None) -> None:
        settings = get_settings()
        self.allowed_extensions = [
            ext.lower().lstrip(".")
            for ext in (allowed_extensions or settings.ALLOWED_FILE_TYPES)
        ]

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from uploaded file bytes based on file extension.

        Args:
            file_bytes: Raw binary content of the file.
            filename: Original name of the uploaded file.

        Returns:
            Cleaned and normalized text extracted from the document.

        Raises:
            UnsupportedFileTypeException: If the file extension is not permitted.
            EmptyFileException: If the file is 0 bytes or yields only whitespace.
            DocumentProcessingException: If file parsing fails due to corruption or format errors.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise EmptyFileException(filename=filename)

        _, ext = os.path.splitext(filename)
        clean_ext = ext.lstrip(".").lower()

        if clean_ext not in self.allowed_extensions:
            raise UnsupportedFileTypeException(file_type=clean_ext, allowed=self.allowed_extensions)

        if clean_ext == "pdf":
            raw_text = self._extract_from_pdf(file_bytes, filename)
        elif clean_ext == "txt":
            raw_text = self._extract_from_txt(file_bytes, filename)
        else:
            raise UnsupportedFileTypeException(file_type=clean_ext, allowed=self.allowed_extensions)

        cleaned_text = self._normalize_text(raw_text)

        if not cleaned_text or not cleaned_text.strip():
            logger.warning("File '%s' yielded zero extracted text after cleaning", filename)
            raise EmptyFileException(filename=filename)

        logger.info(
            "Extracted %d characters from '%s' (%s format)",
            len(cleaned_text),
            filename,
            clean_ext,
        )
        return cleaned_text

    def _extract_from_pdf(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from PDF byte stream using PyPDF.

        Args:
            file_bytes: Binary PDF bytes.
            filename: File name for logging and error reporting.

        Returns:
            Concatenated text from all readable pages.
        """
        try:
            stream = io.BytesIO(file_bytes)
            reader = PdfReader(stream)

            if reader.is_encrypted:
                try:
                    # Attempt decryption with empty password for unprompted encrypted PDFs
                    decrypt_result = reader.decrypt("")
                    if decrypt_result == 0:
                        raise DocumentProcessingException(
                            message=f"The PDF '{filename}' is password-protected and cannot be read.",
                            details={"filename": filename, "reason": "encrypted"},
                        )
                except Exception as dec_err:
                    if isinstance(dec_err, DocumentProcessingException):
                        raise
                    raise DocumentProcessingException(
                        message=f"Failed to decrypt PDF '{filename}': {str(dec_err)}",
                        details={"filename": filename, "error": str(dec_err)},
                    )

            if len(reader.pages) == 0:
                raise EmptyFileException(filename=filename)

            pages_text: List[str] = []
            for page_index, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        pages_text.append(extracted.strip())
                except Exception as page_err:
                    logger.warning(
                        "Error extracting text from page %d of '%s': %s",
                        page_index,
                        filename,
                        str(page_err),
                    )

            total_text = "\n\n".join(pages_text)
            return total_text

        except (EmptyFileException, DocumentProcessingException):
            raise
        except PyPdfError as pdf_err:
            logger.error("PyPDF error while parsing '%s': %s", filename, str(pdf_err))
            raise DocumentProcessingException(
                message=f"Failed to parse PDF document '{filename}': {str(pdf_err)}",
                details={"filename": filename, "error": str(pdf_err)},
            )
        except Exception as exc:
            logger.error("Unexpected error reading PDF '%s': %s", filename, str(exc))
            raise DocumentProcessingException(
                message=f"Unexpected error while extracting text from PDF '{filename}': {str(exc)}",
                details={"filename": filename, "error": str(exc)},
            )

    def _extract_from_txt(self, file_bytes: bytes, filename: str) -> str:
        """Safely extract text from TXT bytes by attempting common encodings.

        Handles UTF-8 with BOM, standard UTF-8, CP1252 / Latin-1, and UTF-16 with BOM.
        """
        # Detect UTF-16 by BOM
        if file_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return file_bytes.decode("utf-16")
            except Exception:
                pass

        # Try utf-8-sig first (strips UTF-8 BOM if present), then strict utf-8, then Windows cp1252 / latin-1
        encodings_to_try = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

        for encoding in encodings_to_try:
            try:
                decoded = file_bytes.decode(encoding)
                return decoded
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback with replacement of un-decodable bytes
        try:
            logger.warning("Decoding '%s' with utf-8 fallback (errors='replace')", filename)
            return file_bytes.decode("utf-8", errors="replace")
        except Exception as err:
            raise DocumentProcessingException(
                message=f"Could not decode text file '{filename}': {str(err)}",
                details={"filename": filename, "error": str(err)},
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean and normalize extracted text string.

        - Removes BOM and null bytes
        - Normalizes newlines (\\r\\n and \\r to \\n)
        - Replaces non-breaking spaces with standard spaces
        - Cleans excessive blank lines while preserving paragraph structure
        """
        if not text:
            return ""

        # Remove BOM character and null bytes
        text = text.replace("\ufeff", "").replace("\x00", "")

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace non-breaking spaces with regular space
        text = text.replace("\xa0", " ")

        # Collapse more than two consecutive newlines into two newlines (paragraph boundary)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


# Singleton instance for convenience
extraction_service = ExtractionService()


def get_extraction_service() -> ExtractionService:
    """FastAPI dependency provider for ExtractionService."""
    return extraction_service

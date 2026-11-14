"""Unit and integration tests for document extraction and Document Ingestion API."""

import io
from typing import AsyncGenerator
from unittest.mock import MagicMock
import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfReader, PdfWriter
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes.documents import get_document_service
from app.api.schemas.document import ChunkingStrategyEnum
from app.database.connection import Base, get_db
from app.main import app
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.services.document_service import DocumentService
from app.services.extraction_service import extraction_service
from app.services.vector_service import VectorService
from app.utils.exceptions import (
    DocumentProcessingException,
    EmptyFileException,
    UnsupportedFileTypeException,
    VectorStorageException,
)


def _generate_valid_pdf_bytes(text_lines: list[str]) -> bytes:
    """Generate in-memory valid PDF bytes containing text."""
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


# ==============================================================================
# Ingestion API Integration Test Fixtures
# ==============================================================================

@pytest.fixture
def test_db_session() -> Session:
    """Fixture providing an isolated in-memory SQLite database session with StaticPool."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    testing_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = testing_session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_vector_service() -> VectorService:
    """Fixture providing a VectorService backed by an isolated in-memory Qdrant instance."""
    client = QdrantClient(":memory:", check_compatibility=False)
    service = VectorService(client=client, collection_name="test_api_docs")
    return service


@pytest.fixture
async def api_client(test_db_session: Session, test_vector_service: VectorService) -> AsyncGenerator[AsyncClient, None]:
    """Fixture providing an HTTPX AsyncClient with overridden database and vector dependencies."""
    def override_get_db():
        yield test_db_session

    def override_get_document_service():
        return DocumentService(db=test_db_session, vector_service=test_vector_service)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_document_service] = override_get_document_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


# ==============================================================================
# Ingestion API Test Suite
# ==============================================================================

class TestDocumentIngestionAPI:
    """Integration test suite for the Document Ingestion API endpoints."""

    @pytest.mark.asyncio
    async def test_upload_txt_file_recursive_strategy(self, api_client: AsyncClient, test_db_session: Session):
        """Verify successful upload of a TXT file using recursive chunking strategy."""
        sample_text = (
            "PalmMind AI Interview Assistant Documentation.\n\n"
            "Chapter 1: Architecture\n"
            "The system uses FastAPI for REST endpoints, Qdrant for semantic search, "
            "and SQLite for document and booking persistence.\n\n"
            "Chapter 2: Interview Booking\n"
            "Candidates can schedule technical interviews by providing their name, "
            "email address, preferred date, and available time slot."
        )
        files = {
            "file": ("system_overview.txt", sample_text.encode("utf-8"), "text/plain"),
        }
        data = {
            "chunking_strategy": "recursive",
        }

        response = await api_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 201

        payload = response.json()
        assert payload["success"] is True
        assert isinstance(payload["document_id"], int)
        assert payload["document_id"] > 0
        assert payload["filename"] == "system_overview.txt"
        assert payload["extracted_character_count"] == len(sample_text)
        assert payload["number_of_chunks"] >= 1
        assert payload["selected_chunking_strategy"] == "recursive"
        assert payload["number_of_embeddings_stored"] == payload["number_of_chunks"]

        # Verify SQL persistence
        doc_repo = DocumentRepository(test_db_session)
        persisted_doc = doc_repo.get_by_id(payload["document_id"])
        assert persisted_doc is not None
        assert persisted_doc.filename == "system_overview.txt"
        assert persisted_doc.total_chunks == payload["number_of_chunks"]
        assert len(persisted_doc.chunks) == payload["number_of_chunks"]
        assert persisted_doc.chunks[0].chunk_index == 0

    @pytest.mark.asyncio
    async def test_upload_txt_file_fixed_strategy(self, api_client: AsyncClient):
        """Verify successful upload of a TXT file using fixed-size chunking strategy."""
        sample_text = (
            "Fixed-size chunking divides text into rigid character windows. "
            "It guarantees bounded chunk length regardless of sentence or paragraph structure. "
            "This strategy is suitable for uniform embedding generation."
        )
        files = {
            "file": ("fixed_test.txt", sample_text.encode("utf-8"), "text/plain"),
        }
        data = {
            "chunking_strategy": "fixed",
            "chunk_size": "80",
            "chunk_overlap": "10",
        }

        response = await api_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 201

        payload = response.json()
        assert payload["success"] is True
        assert payload["selected_chunking_strategy"] == "fixed"
        assert payload["number_of_chunks"] >= 2
        assert payload["number_of_embeddings_stored"] == payload["number_of_chunks"]

    @pytest.mark.asyncio
    async def test_upload_pdf_file_success(self, api_client: AsyncClient, test_db_session: Session):
        """Verify successful upload, text extraction, and indexing of a PDF document."""
        pdf_bytes = _generate_valid_pdf_bytes([
            "PalmMind Technical Assignment",
            "Engineering Intern Hiring Guide",
            "Topics: Retrieval Augmented Generation and Machine Learning",
        ])
        files = {
            "file": ("guide.pdf", pdf_bytes, "application/pdf"),
        }
        data = {
            "chunking_strategy": "recursive",
        }

        response = await api_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 201

        payload = response.json()
        assert payload["success"] is True
        assert payload["filename"] == "guide.pdf"
        assert payload["extracted_character_count"] > 0
        assert payload["number_of_chunks"] >= 1
        assert payload["number_of_embeddings_stored"] == payload["number_of_chunks"]

    @pytest.mark.asyncio
    async def test_upload_empty_file_rejected(self, api_client: AsyncClient):
        """Verify that uploading a 0-byte file returns HTTP 400 with EmptyFileException."""
        files = {
            "file": ("empty.txt", b"", "text/plain"),
        }
        data = {"chunking_strategy": "recursive"}

        response = await api_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 400
        payload = response.json()
        assert payload["error"] == "EmptyFileException"
        assert "empty.txt" in payload["message"]

    @pytest.mark.asyncio
    async def test_upload_whitespace_only_file_rejected(self, api_client: AsyncClient):
        """Verify that uploading a file with only spaces/newlines returns HTTP 400."""
        files = {
            "file": ("blank.txt", b"   \n\t  \r\n   ", "text/plain"),
        }
        data = {"chunking_strategy": "recursive"}

        response = await api_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 400
        payload = response.json()
        assert payload["error"] == "EmptyFileException"

    @pytest.mark.asyncio
    async def test_upload_unsupported_format_rejected(self, api_client: AsyncClient):
        """Verify that uploading an unsupported format (.docx, .bin) returns HTTP 400."""
        files = {
            "file": ("report.docx", b"PK\x03\x04 fake docx content", "application/octet-stream"),
        }
        data = {"chunking_strategy": "recursive"}

        response = await api_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 400
        payload = response.json()
        assert payload["error"] == "UnsupportedFileTypeException"
        assert "docx" in payload["message"]

    @pytest.mark.asyncio
    async def test_list_documents_endpoint(self, api_client: AsyncClient):
        """Verify listing documents returns all ingested documents."""
        # Ingest document first
        sample_text = "Sample text for document listing verification."
        files = {"file": ("listing_test.txt", sample_text.encode("utf-8"), "text/plain")}
        upload_resp = await api_client.post("/api/v1/documents/upload", files=files)
        assert upload_resp.status_code == 201

        # Query listing
        list_resp = await api_client.get("/api/v1/documents/")
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert isinstance(docs, list)
        assert len(docs) >= 1
        filenames = [d["filename"] for d in docs]
        assert "listing_test.txt" in filenames

    @pytest.mark.asyncio
    async def test_get_document_detail_endpoint(self, api_client: AsyncClient):
        """Verify getting document details returns metadata and ordered chunks."""
        sample_text = "Section 1: Details test.\n\nSection 2: Verifying chunk hierarchy."
        files = {"file": ("detail_test.txt", sample_text.encode("utf-8"), "text/plain")}
        upload_resp = await api_client.post("/api/v1/documents/upload", files=files)
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["document_id"]

        # Fetch detail
        detail_resp = await api_client.get(f"/api/v1/documents/{doc_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == doc_id
        assert detail["filename"] == "detail_test.txt"
        assert "chunks" in detail
        assert len(detail["chunks"]) >= 1
        assert detail["chunks"][0]["chunk_index"] == 0
        assert "content" in detail["chunks"][0]

    @pytest.mark.asyncio
    async def test_get_nonexistent_document_returns_404(self, api_client: AsyncClient):
        """Verify that fetching an unknown document ID returns HTTP 404."""
        response = await api_client.get("/api/v1/documents/999999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_endpoint(self, api_client: AsyncClient):
        """Verify deleting a document removes it from the database."""
        sample_text = "Content for deletion test."
        files = {"file": ("to_delete.txt", sample_text.encode("utf-8"), "text/plain")}
        upload_resp = await api_client.post("/api/v1/documents/upload", files=files)
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["document_id"]

        # Delete
        delete_resp = await api_client.delete(f"/api/v1/documents/{doc_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["success"] is True

        # Verify not found after deletion
        get_resp = await api_client.get(f"/api/v1/documents/{doc_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_vector_failure(self, test_db_session: Session):
        """Verify that a vector store failure causes an SQLite rollback without orphan records."""
        # Create a mock vector service that fails during upsert
        failing_vector_service = MagicMock(spec=VectorService)
        failing_vector_service.upsert_chunks.side_effect = VectorStorageException(
            "Simulated Qdrant storage failure",
            details={"reason": "connection lost"},
        )

        service = DocumentService(
            db=test_db_session,
            vector_service=failing_vector_service,
        )

        with pytest.raises(VectorStorageException):
            service.ingest_document(
                filename="failing_doc.txt",
                file_bytes=b"This text should not be persisted in SQLite due to vector failure.",
                chunking_strategy=ChunkingStrategyEnum.RECURSIVE,
            )

        # Verify SQLite has no records for failing_doc.txt
        doc_repo = DocumentRepository(test_db_session)
        assert doc_repo.get_by_filename("failing_doc.txt") is None

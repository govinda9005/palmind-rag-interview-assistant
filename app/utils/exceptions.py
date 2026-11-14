"""Custom exception hierarchy for PalmMind AI backend."""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base exception class for all application-specific errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentProcessingException(AppException):
    """Raised when document ingestion or parsing fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=422, details=details)


class UnsupportedFileTypeException(AppException):
    """Raised when an uploaded file type is not supported."""

    def __init__(self, file_type: str, allowed: list) -> None:
        message = f"Unsupported file type '{file_type}'. Allowed types are: {', '.join(allowed)}."
        super().__init__(
            message=message,
            status_code=400,
            details={"file_type": file_type, "allowed": allowed},
        )


class EmptyFileException(AppException):
    """Raised when an uploaded file is empty (0 bytes or no extractable text)."""

    def __init__(self, filename: str) -> None:
        message = f"The uploaded file '{filename}' is empty or contains no extractable text."
        super().__init__(message=message, status_code=400, details={"filename": filename})


class FileSizeExceededException(AppException):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, filename: str, file_size_mb: float, max_size_mb: float) -> None:
        message = (
            f"File '{filename}' ({file_size_mb:.2f} MB) exceeds maximum allowed size "
            f"of {max_size_mb:.2f} MB."
        )
        super().__init__(
            message=message,
            status_code=413,
            details={
                "filename": filename,
                "file_size_mb": file_size_mb,
                "max_size_mb": max_size_mb,
            },
        )


class EmbeddingException(AppException):
    """Raised when embedding model execution fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=500, details=details)


class VectorStorageException(AppException):
    """Raised when Qdrant operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=500, details=details)


class BookingValidationException(AppException):
    """Raised when interview booking payload validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=422, details=details)


class MemoryException(AppException):
    """Raised when Redis conversational memory read/write fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=500, details=details)


class RAGException(AppException):
    """Raised when conversational RAG generation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=500, details=details)


"""API routes package."""

from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router

__all__ = ["documents_router", "chat_router"]

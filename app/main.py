"""FastAPI application entrypoint for PalmMind AI Backend."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.database.init_db import init_db
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router
from app.utils.exceptions import AppException

# Setup structured logger
logger = setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Application starting up: %s (v%s)", settings.PROJECT_NAME, settings.VERSION)
    # Initialize database tables on startup
    init_db()
    yield
    logger.info("Application shutting down: %s", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Document-Based Conversational RAG & Interview Booking System "
        "built for PalmMind AI Internship Assignment."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# Enable CORS for local development and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(chat_router, prefix=settings.API_V1_PREFIX)


# Global Exception Handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions consistently."""
    logger.warning(
        "Application exception on %s %s: status=%d message=%s details=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.message,
        exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler for unexpected server errors."""
    logger.exception("Unhandled error processing request %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the request.",
        },
    )


# Permitted Healthcheck Endpoint
@app.get("/health", tags=["Operational"])
async def health_check() -> Dict[str, Any]:
    """Operational health check endpoint for monitoring application status."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "online",
            "database": "configured",
            "vector_store": "configured",
            "redis_memory": "configured",
            "llm": "configured",
        },
    }


@app.get("/", tags=["Operational"])
async def root() -> Dict[str, str]:
    """Root entrypoint displaying service status and docs URL."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
    }


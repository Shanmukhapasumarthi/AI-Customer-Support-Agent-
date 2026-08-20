from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings, setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks.

    LAZY-LOADING MODE: Everything before `yield` runs once at startup; everything
    after runs at shutdown. We skip embedding and database warm-up to avoid OOM on
    memory-constrained environments (e.g., Render free tier). These will load on
    first use instead, making the first request slower but keeping startup fast.
    """
    setup_logging()
    logger.info("Starting %s (lazy-loading enabled)", settings.app_name)
    logger.info("Embeddings and database will load on first request")

    yield

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "A production-style customer support agent built with LangChain. "
            "Combines RAG over a private knowledge base with database tools, "
            "conversational memory, structured output, and human escalation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS: the browser blocks cross-origin requests by default, and Streamlit
    # runs on a different port to the API. Wide-open origins are fine for local
    # development; in production replace "*" with your actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", tags=["system"])
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
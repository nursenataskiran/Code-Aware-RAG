from __future__ import annotations

import logging
import logging.config

from fastapi import FastAPI

from api.errors import register_error_handlers
from api.routes import chat_router, health_router


def _configure_logging() -> None:
    """Set up a simple, readable log format for the API server."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }
    )


def create_app() -> FastAPI:
    _configure_logging()

    app = FastAPI(
        title="Code-Aware RAG API",
        description=(
            "Ask questions about a set of GitHub repositories. "
            "The system retrieves relevant code chunks and generates grounded answers."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register routes
    app.include_router(health_router)
    app.include_router(chat_router)

    # Register error handlers
    register_error_handlers(app)

    return app


# Module-level app instance — used by `uvicorn api.app:app`
app = create_app()

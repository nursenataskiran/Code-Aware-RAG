"""
ingestion_routes.py
-------------------
FastAPI router for the GitHub repository ingestion endpoint.

POST /api/v1/ingest/github
    Accepts a public GitHub repo URL, downloads supported source files into
    data/raw/<owner__repo>/, and returns a structured result.

Design
~~~~~~
* This route is intentionally thin — it delegates all logic to
  ``src.ingestion.github_ingestor.ingest_github_repo``.
* Domain exceptions raised by the service are mapped here to the
  appropriate HTTP status codes, keeping error-handling explicit and
  consistent with the existing error conventions in this API.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import GitHubIngestRequest, GitHubIngestResponse, SkippedFile
from src.ingestion.github_ingestor import (
    ChunkingError,
    GitHubAPIError,
    GitHubRateLimitError,
    IndexingError,
    InvalidGitHubURLError,
    NoSupportedFilesError,
    RepositoryNotFoundError,
    ingest_github_repo,
)

logger = logging.getLogger(__name__)

ingestion_router = APIRouter(
    prefix="/api/v1/ingest",
    tags=["Ingestion"],
)


@ingestion_router.post(
    "/github",
    response_model=GitHubIngestResponse,
    status_code=200,
    summary="Ingest a public GitHub repository",
    description=(
        "Accepts a public GitHub repository URL, downloads all supported "
        "source files (.py, .md, .ipynb) into data/raw/<owner__repo>/, "
        "chunks them with the existing SmartChunker, and adds the chunks "
        "to the persistent Chroma collection. "
        "If the project is already indexed, returns immediately without "
        "re-downloading or re-indexing."
    ),
)
async def ingest_github(body: GitHubIngestRequest) -> GitHubIngestResponse:
    """Thin route — delegates all work to the ingestor service."""
    logger.info("Ingestion request received for: %s", body.repo_url)

    try:
        result = ingest_github_repo(body.repo_url)
    except InvalidGitHubURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except NoSupportedFilesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GitHubAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Filesystem error during ingestion of %s", body.repo_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ChunkingError as exc:
        logger.exception("Chunking failed for %s", body.repo_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except IndexingError as exc:
        logger.exception("Indexing failed for %s", body.repo_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "Ingestion complete for %s — status: %s, downloaded: %d, skipped: %d",
        body.repo_url,
        result["status"],
        len(result["downloaded_files"]),
        len(result["skipped_files"]),
    )

    return GitHubIngestResponse(
        status=result["status"],
        project_name=result["project_name"],
        repo_url=result["repo_url"],
        downloaded_files=result["downloaded_files"],
        skipped_files=[SkippedFile(**s) for s in result["skipped_files"]],
        indexed_chunks=result["indexed_chunks"],
        message=result["message"],
    )

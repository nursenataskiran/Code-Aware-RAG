"""
github_ingestor.py
------------------
Business logic for ingesting a public GitHub repository into data/raw/.

Responsibilities
~~~~~~~~~~~~~~~~
* Parse and validate the GitHub URL.
* Check idempotency: if the project directory already exists, return early.
* Fetch repository metadata (default branch) via the GitHub REST API.
* List all tree entries recursively via the Git Trees API.
* Filter to supported file extensions, excluding unwanted paths/patterns.
* Download each file while preserving folder structure.
* Return a structured result dict that the route layer can serialise.

Nothing in here triggers chunking, embedding, or vector-store operations.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GITHUB_API_BASE = "https://api.github.com"
RAW_CONTENT_BASE = "https://raw.githubusercontent.com"

#: File extensions the chunking pipeline can handle.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".py", ".md", ".ipynb"})

#: Any path segment matching one of these names is skipped entirely.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        ".env",
    }
)

#: Patterns in file paths that signal credential / secret / key material.
SENSITIVE_PATH_PATTERNS: tuple[str, ...] = (
    "secret",
    "credential",
    "password",
    "private_key",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".p12",
    ".pfx",
    ".key",
)

#: Files larger than this (bytes) are skipped to avoid downloading huge blobs.
MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB

#: Per-request timeout for GitHub API calls and raw file downloads (seconds).
REQUEST_TIMEOUT: int = 20

#: Shared headers sent with every GitHub API call (no auth token needed for
#: public repos; the Accept header requests the v3 API response format).
_GITHUB_HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ── Custom exceptions ─────────────────────────────────────────────────────────


class InvalidGitHubURLError(ValueError):
    """Raised when the supplied URL cannot be parsed as a GitHub repo URL."""


class RepositoryNotFoundError(LookupError):
    """Raised when the GitHub API returns 404 for the repository."""


class GitHubRateLimitError(RuntimeError):
    """Raised when the GitHub API returns 403/429 indicating rate limiting."""


class NoSupportedFilesError(ValueError):
    """Raised when the repo contains no files with supported extensions."""


class GitHubAPIError(RuntimeError):
    """Raised for unexpected upstream failures from the GitHub API."""


class ChunkingError(RuntimeError):
    """Raised when the chunking step produces zero usable chunks."""


class IndexingError(RuntimeError):
    """Raised when the Chroma add step fails."""


# ── Public interface ──────────────────────────────────────────────────────────


def ingest_github_repo(repo_url: str) -> dict[str, Any]:
    """
    Main entry point called by the FastAPI route.

    Parameters
    ----------
    repo_url:
        A public GitHub repository URL such as
        ``https://github.com/owner/repo`` or ``https://github.com/owner/repo/``.

    Returns
    -------
    dict with keys:
        status           – ``"already_indexed"`` | ``"indexed"`` | ``"downloaded_and_indexed"``
        project_name     – normalised name used as the local directory
        repo_url         – the original URL (echoed back)
        downloaded_files – list of relative paths that were written (empty when skipped)
        skipped_files    – list of {path, reason} dicts for skipped entries
        indexed_chunks   – number of chunks added to Chroma (0 when already indexed)
        message          – human-readable summary

    Three-state flow
    ~~~~~~~~~~~~~~~~
    1. Chroma already has chunks for this project → return ``already_indexed``
       immediately (no download, no re-index).
    2. ``data/raw/<project_name>/`` exists but not indexed → skip download,
       chunk + index the local files, return ``indexed``.
    3. Nothing exists → download → chunk → index, return
       ``downloaded_and_indexed``.
    """
    # Import here to avoid a module-level circular import between the two
    # ingestion service modules.
    from src.ingestion.repo_indexer import index_project, is_project_indexed

    owner, repo = _parse_github_url(repo_url)
    project_name = f"{owner}__{repo}"
    project_dir = RAW_DATA_DIR / project_name

    # ── State 1: already indexed in Chroma ──────────────────────────────
    # Chroma check happens first, before touching the filesystem or the network.
    if is_project_indexed(project_name):
        logger.info("Project '%s' is already indexed in Chroma.", project_name)
        return {
            "status": "already_indexed",
            "project_name": project_name,
            "repo_url": repo_url,
            "downloaded_files": [],
            "skipped_files": [],
            "indexed_chunks": 0,
            "message": (
                f"Project '{project_name}' is already indexed. "
                "Use the chat endpoint to query it."
            ),
        }

    # ── State 2: files on disk but not yet indexed ──────────────────────
    # The repo was downloaded previously (or placed manually) but indexing
    # never completed. Skip the download and go straight to indexing.
    if project_dir.exists():
        logger.info(
            "Project dir '%s' exists but is not indexed. Indexing now.",
            project_dir,
        )
        indexed_chunks = index_project(project_dir, project_name)
        return {
            "status": "indexed",
            "project_name": project_name,
            "repo_url": repo_url,
            "downloaded_files": [],
            "skipped_files": [],
            "indexed_chunks": indexed_chunks,
            "message": (
                f"Found existing local files for '{project_name}'. "
                f"Indexed {indexed_chunks} chunk(s) into Chroma — "
                "no download was needed."
            ),
        }

    # ── State 3: nothing exists — full pipeline ─────────────────────────

    # ── Fetch default branch ──────────────────────────────────────────────
    default_branch = _fetch_default_branch(owner, repo)
    logger.info("Default branch for %s/%s: %s", owner, repo, default_branch)

    # ── List all tree entries (recursive) ────────────────────────────────
    all_blobs = _fetch_tree_blobs(owner, repo, default_branch)
    logger.info("Total blobs in tree: %d", len(all_blobs))

    # ── Filter to downloadable files ──────────────────────────────────────
    to_download, skipped_files = _filter_blobs(all_blobs)

    if not to_download:
        raise NoSupportedFilesError(
            f"Repository '{owner}/{repo}' contains no supported files "
            f"({', '.join(sorted(SUPPORTED_EXTENSIONS))}) after filtering."
        )

    logger.info(
        "Files to download: %d | Skipped: %d", len(to_download), len(skipped_files)
    )

    # ── Download and write files ──────────────────────────────────────────
    downloaded_files: list[str] = []
    write_errors: list[tuple[str, str]] = []

    for blob in to_download:
        path: str = blob["path"]
        try:
            content = _download_raw_file(owner, repo, default_branch, path)
            _write_file(project_dir, path, content)
            downloaded_files.append(path)
        except OSError as exc:
            logger.error("Failed to write %s: %s", path, exc)
            write_errors.append((path, f"write error: {exc}"))
        except GitHubRateLimitError:
            # Rate-limit during file download aborts the entire ingestion;
            # clean up and re-raise so the route can return 429.
            _cleanup_directory(project_dir)
            raise
        except GitHubAPIError as exc:
            # Any other per-file download failure is non-fatal: log it,
            # record it as skipped, and continue with the remaining files.
            logger.warning("Skipping %s — download error: %s", path, exc)
            skipped_files.append((path, f"download error: {exc}"))

    if write_errors:
        # Clean up the partially written directory to leave a clean state,
        # then surface the error so the route can return 500.
        _cleanup_directory(project_dir)
        raise OSError(
            f"Failed to write {len(write_errors)} file(s) to {project_dir}. "
            "The partially created directory has been removed."
        )

    # If every candidate file failed to download, this is an upstream failure —
    # not a "no supported files" situation. Raise GitHubAPIError so the route
    # maps it to 502, distinguishing it from repos that genuinely have no
    # supported files (NoSupportedFilesError → 422).
    if not downloaded_files:
        _cleanup_directory(project_dir)
        raise GitHubAPIError(
            f"No files could be downloaded from '{owner}/{repo}'. "
            f"All {len(skipped_files)} candidate file(s) were skipped or failed."
        )

    # ── Chunk and index the newly downloaded project ──────────────────────
    indexed_chunks = index_project(project_dir, project_name)

    message = (
        f"Successfully downloaded {len(downloaded_files)} file(s) from "
        f"'{owner}/{repo}' and indexed {indexed_chunks} chunk(s)."
    )
    if skipped_files:
        message += f" {len(skipped_files)} file(s) were skipped."

    return {
        "status": "downloaded_and_indexed",
        "project_name": project_name,
        "repo_url": repo_url,
        "downloaded_files": downloaded_files,
        "skipped_files": [{"path": p, "reason": r} for p, r in skipped_files],
        "indexed_chunks": indexed_chunks,
        "message": message,
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract ``(owner, repo)`` from a GitHub repository URL.

    Accepts:
        https://github.com/owner/repo
        https://github.com/owner/repo/

    Raises
    ------
    InvalidGitHubURLError
        If the URL is not a valid public GitHub repository URL.
    """
    try:
        parsed = urlparse(url.strip().rstrip("/"))
    except Exception as exc:
        raise InvalidGitHubURLError(f"Cannot parse URL: {url!r}") from exc

    if parsed.scheme not in ("http", "https"):
        raise InvalidGitHubURLError(
            f"URL must start with http:// or https://. Got: {url!r}"
        )
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise InvalidGitHubURLError(
            f"URL must point to github.com. Got: {url!r}"
        )

    # Path should be exactly /owner/repo (after stripping trailing slash above)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 2:
        raise InvalidGitHubURLError(
            f"URL must be of the form https://github.com/owner/repo. Got: {url!r}"
        )

    owner, repo = parts
    # Basic sanity check on owner/repo names (GitHub only allows alphanumeric, - and _)
    _name_re = re.compile(r"^[A-Za-z0-9_.\-]+$")
    if not _name_re.match(owner) or not _name_re.match(repo):
        raise InvalidGitHubURLError(
            f"Owner or repository name contains invalid characters: {url!r}"
        )

    return owner, repo


def _github_get(url: str, *, description: str) -> dict[str, Any]:
    """
    Perform a GET request to the GitHub REST API.

    Maps HTTP status codes to domain exceptions so the route layer can
    translate them cleanly to the appropriate HTTP response codes.

    Parameters
    ----------
    url:         Full GitHub API URL.
    description: Human-readable description used in log messages.

    Raises
    ------
    RepositoryNotFoundError   – 404
    GitHubRateLimitError      – 403 (secondary rate limit) or 429
    GitHubAPIError            – all other non-2xx responses or network errors
    """
    try:
        response = requests.get(url, headers=_GITHUB_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        raise GitHubAPIError(
            f"Request timed out while fetching {description} from GitHub."
        )
    except requests.ConnectionError as exc:
        raise GitHubAPIError(
            f"Network error while fetching {description}: {exc}"
        )

    if response.status_code == 404:
        raise RepositoryNotFoundError(
            f"GitHub returned 404 for {description}. "
            "The repository may not exist or may be private."
        )
    if response.status_code in (403, 429):
        retry_after = response.headers.get("Retry-After", "unknown")
        raise GitHubRateLimitError(
            f"GitHub rate limit exceeded while fetching {description}. "
            f"Retry-After: {retry_after}s."
        )
    if not response.ok:
        raise GitHubAPIError(
            f"GitHub API error {response.status_code} for {description}: "
            f"{response.text[:200]}"
        )

    return response.json()  # type: ignore[return-value]


def _fetch_default_branch(owner: str, repo: str) -> str:
    """Return the default branch name for the given repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    data = _github_get(url, description=f"repository metadata for {owner}/{repo}")
    branch: str = data.get("default_branch", "main")
    return branch


def _fetch_tree_blobs(
    owner: str, repo: str, branch: str
) -> list[dict[str, Any]]:
    """
    Return all blob entries in the repository tree (recursive).

    Uses the Git Trees API:
    GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
    """
    url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}"
        "?recursive=1"
    )
    data = _github_get(url, description=f"file tree for {owner}/{repo}@{branch}")

    if data.get("truncated"):
        logger.warning(
            "GitHub tree response is truncated for %s/%s — "
            "very large repositories may have files missing.",
            owner,
            repo,
        )

    return [entry for entry in data.get("tree", []) if entry.get("type") == "blob"]


def _is_excluded_path(path: str) -> str | None:
    """
    Return a skip reason string if the path should be excluded, else ``None``.

    Checks:
    * Path segments against EXCLUDED_DIRS (e.g. node_modules, venv).
    * Sensitive name patterns (secrets, keys, credentials).
    * File extension: only SUPPORTED_EXTENSIONS are accepted.
    * File size: blobs exceeding MAX_FILE_SIZE_BYTES are skipped.
    """
    parts = Path(path).parts

    for segment in parts:
        segment_lower = segment.lower()
        if segment_lower in EXCLUDED_DIRS:
            return f"excluded directory: {segment}"

    path_lower = path.lower()
    for pattern in SENSITIVE_PATH_PATTERNS:
        if pattern in path_lower:
            return f"sensitive path pattern: {pattern!r}"

    filename = Path(path).name
    if filename == "__init__.py":
        return "package marker file: __init__.py"

    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return f"unsupported extension: {suffix or '(none)'}"

    return None


def _filter_blobs(
    blobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """
    Split *blobs* into (to_download, skipped) lists.

    Each entry in *skipped* is a ``(path, reason)`` tuple.
    Size filtering is applied here using the ``size`` field from the tree API
    (not all entries include it, so absence of the field is treated as OK).
    """
    to_download: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for blob in blobs:
        path: str = blob.get("path", "")

        reason = _is_excluded_path(path)
        if reason:
            skipped.append((path, reason))
            continue

        size: int = blob.get("size", 0)
        if size > MAX_FILE_SIZE_BYTES:
            skipped.append((path, f"file too large: {size} bytes"))
            continue

        to_download.append(blob)

    return to_download, skipped


def _download_raw_file(owner: str, repo: str, branch: str, path: str) -> bytes:
    """
    Download a single raw file from GitHub.

    Uses ``raw.githubusercontent.com`` which does not count against the
    REST API rate limit.

    Branch and path segments are percent-encoded so that names containing
    spaces, unicode characters, or other special characters are handled
    correctly. The forward slashes that separate path segments are preserved
    (``safe="/"`` in the path encode call).

    Raises
    ------
    GitHubRateLimitError     – 429 from raw CDN
    GitHubAPIError           – any other non-2xx, timeout, or network error
    """
    encoded_branch = quote(branch, safe="")
    encoded_path = quote(path, safe="/")
    url = f"{RAW_CONTENT_BASE}/{owner}/{repo}/{encoded_branch}/{encoded_path}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        raise GitHubAPIError(f"Download timed out for: {path}")
    except requests.ConnectionError as exc:
        raise GitHubAPIError(f"Network error downloading {path}: {exc}")

    if response.status_code == 429:
        raise GitHubRateLimitError(f"Rate limited while downloading: {path}")
    if not response.ok:
        raise GitHubAPIError(
            f"Failed to download {path}: HTTP {response.status_code}"
        )

    return response.content


def _write_file(project_dir: Path, relative_path: str, content: bytes) -> None:
    """
    Write *content* to ``project_dir / relative_path``, creating
    intermediate directories as needed.

    Path-traversal guard
    ~~~~~~~~~~~~~~~~~~~~
    Both ``project_dir`` and the computed target path are resolved to their
    absolute, canonical forms before writing.  If the resolved target is not
    inside ``project_dir`` (e.g. a blob whose ``path`` field contains ``..``
    segments) the write is refused with an ``OSError``.

    Raises
    ------
    OSError  – path-traversal attempt detected, or propagated from the
               filesystem layer.
    """
    resolved_root = project_dir.resolve()
    target = (project_dir / relative_path).resolve()

    # Ensure the resolved target is inside the project directory.
    # Path.is_relative_to() was added in Python 3.9; use the str-prefix
    # check as a fallback-safe alternative that works on 3.8+ too.
    try:
        target.relative_to(resolved_root)
    except ValueError:
        raise OSError(
            f"Path traversal detected: '{relative_path}' resolves outside "
            f"the project directory '{resolved_root}'."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _cleanup_directory(directory: Path) -> None:
    """Remove *directory* and all its contents, ignoring errors."""
    import shutil

    try:
        shutil.rmtree(directory)
        logger.info("Cleaned up partial directory: %s", directory)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clean up %s: %s", directory, exc)

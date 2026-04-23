from __future__ import annotations

import time
import threading
from typing import Tuple

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        self._lock = threading.Lock()
        self._count: int = 0
        self._window_start: float = time.monotonic()

    def is_allowed(self) -> Tuple[bool, int]:
        """
        Check whether the current request is within the rate limit.

        Returns (allowed, remaining_requests).
        """
        now = time.monotonic()

        with self._lock:
            # Reset window when it expires
            if now - self._window_start >= self.window_seconds:
                self._count = 0
                self._window_start = now

            if self._count >= self.max_requests:
                return False, 0

            self._count += 1
            remaining = self.max_requests - self._count
            return True, remaining


# Module-level singleton — shared across all requests
_limiter = InMemoryRateLimiter(max_requests=60, window_seconds=60)


async def rate_limit_dependency(request: Request) -> None:
    allowed, remaining = _limiter.is_allowed()
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again in a moment.",
            headers={"Retry-After": str(_limiter.window_seconds)},
        )

"""
A minimal login rate limiter: blocks an identifier (here, the email being
logged in as) after too many failed attempts in a rolling window.

This is intentionally simple and has a real limitation worth knowing: state
lives in a plain in-process dict, so it only works correctly with a single
worker process. Run this behind multiple uvicorn/gunicorn workers (or
multiple machines) and each process has its own independent counter — an
attacker could get several times MAX_ATTEMPTS by getting routed to
different workers. A production version of this would keep the counters in
Redis (or similar shared store) instead, exactly so every worker sees the
same count. Fine for local dev and for understanding the pattern; swap the
storage, not the logic, before deploying multi-process.
"""

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, status

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 300  # 5 minutes

_failed_attempts: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


def _prune(identifier: str, now: float) -> None:
    _failed_attempts[identifier] = [t for t in _failed_attempts[identifier] if now - t < WINDOW_SECONDS]


def check_rate_limit(identifier: str) -> None:
    """Raises 429 if `identifier` has hit MAX_ATTEMPTS failures within WINDOW_SECONDS."""
    now = time.time()
    with _lock:
        _prune(identifier, now)
        if len(_failed_attempts[identifier]) >= MAX_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Try again later.",
            )


def record_failed_attempt(identifier: str) -> None:
    with _lock:
        _failed_attempts[identifier].append(time.time())


def clear_attempts(identifier: str) -> None:
    """Called on successful login — a good login shouldn't count against a future lockout."""
    with _lock:
        _failed_attempts.pop(identifier, None)
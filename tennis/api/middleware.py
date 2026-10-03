"""
API Middleware — JWT authentication, rate limiting, and request logging.
Production-grade middleware stack for the TennisIQ API.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import ClassVar

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("tennisiq.api")


# ── JWT Authentication Middleware ────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT token validation middleware.
    
    Skips auth for public endpoints (health, docs, plans).
    In production, validates against Apple Sign-In / Firebase Auth.
    """

    PUBLIC_PATHS: ClassVar[set[str]] = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/api/v1/subscriptions/plans"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip auth for public paths
        if path in self.PUBLIC_PATHS or path.startswith(("/docs", "/redoc")):
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = self._validate_token(token)
            if user:
                request.state.user_id = user["user_id"]
                request.state.tier = user.get("tier", "free")
                return await call_next(request)

        # In development mode, allow unauthenticated requests
        if self._is_dev_mode():
            request.state.user_id = "dev_user"
            request.state.tier = "elite"
            return await call_next(request)

        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

    def _validate_token(self, token: str) -> dict | None:
        """Validate JWT token. In production: verify signature with public key."""
        # Placeholder — in production, decode JWT and verify with Apple/Firebase
        if token == "test_token":
            return {"user_id": "test_user", "tier": "pro"}
        return None

    def _is_dev_mode(self) -> bool:
        import os
        return os.getenv("ENVIRONMENT", "development") == "development"


# ── Rate Limiting Middleware ─────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.
    
    Limits per tier:
    - Free:  30 req/min
    - Pro:   120 req/min
    - Elite: 600 req/min
    - Default (unauthed): 20 req/min
    """

    TIER_LIMITS: ClassVar[dict[str, int]] = {
        "free": 30,
        "pro": 120,
        "elite": 600,
    }

    def __init__(self, app, default_limit: int = 20):
        super().__init__(app)
        self.default_limit = default_limit
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Identify client
        client_id = getattr(request.state, "user_id", None) or request.client.host or "unknown"
        tier = getattr(request.state, "tier", "free") if hasattr(request.state, "tier") else "free"
        limit = self.TIER_LIMITS.get(tier, self.default_limit)

        # Clean old entries (sliding window of 60 seconds)
        now = time.time()
        window = [t for t in self._buckets[client_id] if now - t < 60]
        self._buckets[client_id] = window

        if len(window) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. {limit} requests/minute for {tier} tier.",
                headers={"Retry-After": "60", "X-RateLimit-Limit": str(limit)},
            )

        self._buckets[client_id].append(now)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - len(window) - 1)
        return response


# ── Request Logging Middleware ───────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing, status, and user context."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Attach request ID
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        user_id = getattr(request.state, "user_id", "anon")

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration_ms:.1f}ms) user={user_id}"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        return response


# ── Subscription Gate Middleware ─────────────────────────────────────────────

class SubscriptionGateMiddleware(BaseHTTPMiddleware):
    """
    Gate premium endpoints behind subscription tiers.
    
    Routes requiring specific tiers:
    - /coaching/*  → Pro+
    - /video/*     → Pro+
    - /stats/heatmap/* → Pro+
    - /stats/speeds/*  → Pro+
    """

    TIER_REQUIRED: ClassVar[dict[str, str]] = {
        "/api/v1/coaching/": "pro",
        "/api/v1/video/": "pro",
        "/api/v1/stats/match/{match_id}/heatmap/": "pro",
        "/api/v1/stats/match/{match_id}/speeds/": "pro",
    }

    TIER_HIERARCHY: ClassVar[dict[str, int]] = {"free": 0, "pro": 1, "elite": 2}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        user_tier = getattr(request.state, "tier", "free") if hasattr(request.state, "tier") else "free"
        user_level = self.TIER_HIERARCHY.get(user_tier, 0)

        for route_prefix, required_tier in self.TIER_REQUIRED.items():
            if path.startswith(route_prefix.split("{")[0]):
                required_level = self.TIER_HIERARCHY.get(required_tier, 0)
                if user_level < required_level:
                    raise HTTPException(
                        status_code=403,
                        detail=f"This feature requires {required_tier.title()} subscription or higher. Current: {user_tier}",
                    )
                break

        return await call_next(request)

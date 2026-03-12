"""
TennisIQ API — FastAPI application factory.
Production-grade REST API for the Tennis Intelligence Platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tennis.config import settings
from tennis.api.routes_sessions import router as sessions_router
from tennis.api.routes_matches import router as matches_router
from tennis.api.routes_events import router as events_router
from tennis.api.routes_stats import router as stats_router
from tennis.api.routes_coaching import router as coaching_router
from tennis.api.routes_video import router as video_router
from tennis.api.routes_subscription import router as subscription_router
from tennis.api.routes_recording import router as recording_router
from tennis.api.routes_linecalls import router as linecalls_router
from tennis.api.routes_upload import router as upload_router
from tennis.api.routes_auth import router as auth_router
from tennis.api.routes_analyze import router as analyze_router
from tennis.api.routes_analytics import router as analytics_router
from tennis.api.routes_replay import router as replay_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TennisIQ API",
        description=(
            "Tennis Match Intelligence Platform. "
            "Single-camera match recording, automated line calling, "
            "player tracking, and post-match analysis."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ──────────────────────────────────────────
    prefix = settings.API_PREFIX
    app.include_router(sessions_router, prefix=f"{prefix}/sessions", tags=["Sessions"])
    app.include_router(matches_router, prefix=f"{prefix}/matches", tags=["Matches"])
    app.include_router(events_router, prefix=f"{prefix}/events", tags=["Events"])
    app.include_router(stats_router, prefix=f"{prefix}/stats", tags=["Stats"])
    app.include_router(coaching_router, prefix=f"{prefix}/coaching", tags=["Coaching"])
    app.include_router(video_router, prefix=f"{prefix}/video", tags=["Video"])
    app.include_router(subscription_router, prefix=f"{prefix}/subscriptions", tags=["Subscriptions"])
    app.include_router(recording_router, prefix=f"{prefix}/recording", tags=["Recording"])
    app.include_router(linecalls_router, prefix=f"{prefix}/linecalls", tags=["Line Calls"])
    app.include_router(upload_router, prefix=f"{prefix}/upload", tags=["Upload"])
    app.include_router(auth_router, prefix=f"{prefix}/auth", tags=["Auth"])
    app.include_router(analyze_router, prefix=f"{prefix}/analyze", tags=["Analyze"])
    app.include_router(analytics_router, prefix=f"{prefix}/analytics", tags=["Analytics"])
    app.include_router(replay_router, prefix=f"{prefix}/replay", tags=["Replay"])

    # ── Health check ────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "app": "TennisIQ",
            "description": "Tennis Match Intelligence",
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    return app


# Module-level app instance for `uvicorn tennis.api.app:app`
app = create_app()

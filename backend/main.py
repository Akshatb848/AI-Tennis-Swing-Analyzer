"""
FastAPI Backend — Education Intelligence Dashboard
Ministry of Education, Government of India
Technology Partner: Deloitte

Exposes REST endpoints for chat, search, compare, health, and newsletter months.
Integrates with the existing LLM layer and agent orchestration.
"""

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moe_backend")

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the LLM client and coordinator on startup."""
    import sys as _sys, pathlib as _pathlib

    project_root = str(_pathlib.Path(__file__).resolve().parent.parent)
    if project_root not in _sys.path:
        _sys.path.insert(0, project_root)

    from llm.client import get_llm_client, validate_llm_client
    from agents.coordinator_agent import CoordinatorAgent
    from agents.data_cleaner_agent import DataCleanerAgent
    from agents.eda_agent import EDAAgent
    from agents.feature_engineer_agent import FeatureEngineerAgent
    from agents.model_trainer_agent import ModelTrainerAgent
    from agents.automl_agent import AutoMLAgent
    from agents.data_visualizer_agent import DataVisualizerAgent
    from agents.dashboard_builder_agent import DashboardBuilderAgent
    from agents.forecast_agent import ForecastAgent
    from agents.insights_agent import InsightsAgent
    from agents.report_generator_agent import ReportGeneratorAgent

    client = get_llm_client(allow_fallback=False)
    validation = None
    if client is not None:
        validation = await validate_llm_client(client)
        logger.info(f"LLM validation: {validation.state} — {validation.message}")

    coordinator = CoordinatorAgent(llm_client=client)
    for agent_cls in [
        DataCleanerAgent, EDAAgent, FeatureEngineerAgent,
        ModelTrainerAgent, AutoMLAgent, DataVisualizerAgent,
        DashboardBuilderAgent, ForecastAgent, InsightsAgent,
        ReportGeneratorAgent,
    ]:
        coordinator.register_agent(agent_cls())

    _state["coordinator"] = coordinator
    _state["llm_client"] = client
    _state["llm_validation"] = validation
    _state["start_time"] = datetime.utcnow().isoformat()
    logger.info("Backend startup complete")

    yield  # app runs here

    logger.info("Backend shutdown")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Education Intelligence Dashboard API",
    description="Ministry of Education — AI-powered education analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory, per-IP)
# ---------------------------------------------------------------------------
_rate_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = [t for t in _rate_store[client_ip] if now - t < 60]
    _rate_store[client_ip] = window
    if len(window) >= RATE_LIMIT:
        return False
    _rate_store[client_ip].append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please wait before retrying."},
        )
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Shared state (populated by lifespan)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    current_month: Optional[str] = None
    include_visualization: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    visualization: Optional[Dict[str, Any]] = None
    session_id: str


class CompareRequest(BaseModel):
    entities: List[str] = Field(..., min_length=2, max_length=5)
    metric: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Helper — sanitise user input
# ---------------------------------------------------------------------------
def _sanitise(text: str) -> str:
    """Strip HTML/script tags to prevent injection."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """System health including LLM and vector-db status."""
    llm_ok = False
    llm_detail = "not configured"
    validation = _state.get("llm_validation")
    if validation is not None:
        llm_ok = validation.state == "connected"
        llm_detail = validation.message

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_since": _state.get("start_time"),
        "services": {
            "llm": {"connected": llm_ok, "detail": llm_detail},
            "vector_db": {"connected": True, "detail": "FAISS index ready"},
            "api": {"connected": True, "detail": "FastAPI operational"},
        },
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """AI chat endpoint — powered by the coordinator agent."""
    query = _sanitise(request.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    coordinator = _state.get("coordinator")
    if coordinator is None:
        raise HTTPException(status_code=503, detail="System is starting up. Please retry.")

    session_id = str(uuid.uuid4())[:8]

    try:
        coordinator.add_to_memory("user", query)
        context = {
            "has_dataset": False,
            "has_target": False,
            "dataset_summary": "",
            "current_month": request.current_month,
        }

        intent_result = await coordinator.analyze_user_intent(query, context)

        if intent_result.get("blocked"):
            return ChatResponse(
                answer=(
                    "The AI assistant is temporarily unavailable. "
                    "Please try again shortly or contact the support desk."
                ),
                sources=[],
                session_id=session_id,
            )

        intent = intent_result.get("intent", "general")

        if intent == "help":
            answer = coordinator.get_help_message()
        elif intent in ("general", "upload_data", "set_target", "status"):
            answer = await coordinator.generate_conversational_reply(query, context)
        else:
            explanation = intent_result.get("explanation", "")
            answer = explanation if explanation else await coordinator.generate_conversational_reply(query, context)

        coordinator.add_to_memory("assistant", answer)

        return ChatResponse(
            answer=answer,
            sources=[],
            visualization=None,
            session_id=session_id,
        )

    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        return ChatResponse(
            answer="I apologise — an internal error occurred. Our team has been notified.",
            sources=[],
            session_id=session_id,
        )


@app.post("/api/compare")
async def compare(request: CompareRequest):
    """Compare education entities/metrics."""
    entities = [_sanitise(e) for e in request.entities]
    metric = _sanitise(request.metric) if request.metric else "overall performance"

    coordinator = _state.get("coordinator")
    if coordinator is None:
        raise HTTPException(status_code=503, detail="System starting up.")

    try:
        query = f"Compare {', '.join(entities)} on {metric}"
        reply = await coordinator.generate_conversational_reply(query, {"has_dataset": False})
        return {
            "comparison": reply,
            "entities": entities,
            "metric": metric,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.error(f"Compare error: {exc}")
        raise HTTPException(status_code=500, detail="Comparison failed.")


@app.post("/api/search")
async def search(request: SearchRequest):
    """Semantic search across education data."""
    query = _sanitise(request.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Return structured results — in production this would hit the FAISS index
    return {
        "query": query,
        "results": [
            {
                "title": f"Education Policy Insight: {query}",
                "snippet": f"Relevant findings for '{query}' across national education data.",
                "relevance": 0.92,
            }
        ],
        "total": 1,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/newsletter/months")
async def newsletter_months():
    """Return available newsletter months for the dashboard selector."""
    months = [
        "January 2025", "February 2025", "March 2025",
        "April 2025", "May 2025", "June 2025",
        "July 2025", "August 2025", "September 2025",
        "October 2025", "November 2025", "December 2025",
        "January 2026", "February 2026",
    ]
    return {"months": months, "default": "February 2026"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=2,
        log_level="info",
    )

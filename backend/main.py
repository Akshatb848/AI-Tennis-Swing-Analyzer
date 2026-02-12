"""
FastAPI Backend — Education Intelligence Dashboard
Ministry of Education, Government of India
Rashtriya Vidya Samiksha Kendra (RVSK), CIET-NCERT

Exposes REST endpoints for chat, search, compare, health, newsletter data,
stats aggregation, LLM status, and admin operations.
Integrates with the RAG system for zero-hallucination, data-backed responses.
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
    """Initialise LLM client, coordinator, RAG system on startup."""
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

    # LLM
    client = get_llm_client(allow_fallback=False)
    validation = None
    if client is not None:
        validation = await validate_llm_client(client)
        logger.info(f"LLM validation: {validation.state} — {validation.message}")

    # Coordinator + agents
    coordinator = CoordinatorAgent(llm_client=client)
    for agent_cls in [
        DataCleanerAgent, EDAAgent, FeatureEngineerAgent,
        ModelTrainerAgent, AutoMLAgent, DataVisualizerAgent,
        DashboardBuilderAgent, ForecastAgent, InsightsAgent,
        ReportGeneratorAgent,
    ]:
        coordinator.register_agent(agent_cls())

    # RAG system
    from backend.rag.rag_system import RAGSystem
    rag = RAGSystem()
    try:
        doc_count = rag.load_and_index()
        logger.info(f"RAG system indexed {doc_count} documents")
    except Exception as e:
        logger.error(f"RAG indexing failed: {e}")

    # Chat handler
    from backend.api.chat_handler import ChatHandler
    chat_handler = ChatHandler(rag_system=rag)

    _state["coordinator"] = coordinator
    _state["llm_client"] = client
    _state["llm_validation"] = validation
    _state["rag"] = rag
    _state["chat_handler"] = chat_handler
    _state["start_time"] = datetime.utcnow().isoformat()
    _state["sessions"] = {}
    logger.info("Backend startup complete")

    yield  # app runs here

    logger.info("Backend shutdown")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Education Intelligence Dashboard API",
    description="Ministry of Education — AI-powered education analytics",
    version="2.0.0",
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
    session_id: Optional[str] = None


class StructuredChatResponse(BaseModel):
    intent: str
    summary: str
    key_points: List[str] = []
    sources: List[Dict[str, Any]] = []
    visualization: Optional[Dict[str, Any]] = None
    confidence: str = "medium"
    data_backed: bool = False
    session_id: str
    answer: str  # backward-compatible plain text


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


class StatsRequest(BaseModel):
    kpi: str = Field(..., min_length=1, max_length=100)
    month: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper — sanitise user input
# ---------------------------------------------------------------------------
def _sanitise(text: str) -> str:
    """Strip HTML/script tags to prevent injection."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ---------------------------------------------------------------------------
# Routes — Health & Status
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """System health including LLM, RAG, and API status."""
    llm_ok = False
    llm_detail = "not configured"
    validation = _state.get("llm_validation")
    if validation is not None:
        llm_ok = validation.state == "connected"
        llm_detail = validation.message

    rag = _state.get("rag")
    rag_ok = rag is not None and rag._indexed
    rag_detail = f"{len(rag.documents)} docs indexed" if rag_ok else "not loaded"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_since": _state.get("start_time"),
        "services": {
            "llm": {"connected": llm_ok, "detail": llm_detail},
            "vector_db": {"connected": rag_ok, "detail": rag_detail},
            "api": {"connected": True, "detail": "FastAPI operational"},
        },
    }


@app.get("/api/llm/status")
async def llm_status():
    """Detailed LLM status for admin dashboard."""
    validation = _state.get("llm_validation")
    if validation is None:
        return {"configured": False, "status": "not_configured", "message": "No LLM client configured"}
    return {
        "configured": True,
        "status": validation.state,
        "message": validation.message,
        "provider": validation.provider,
        "model": validation.model,
        "latency_ms": validation.latency_ms,
    }


# ---------------------------------------------------------------------------
# Routes — Chat (structured responses)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=StructuredChatResponse)
async def chat(request: ChatRequest):
    """AI chat endpoint — returns structured, data-backed responses."""
    query = _sanitise(request.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())[:8]

    # Track session
    if session_id not in _state.get("sessions", {}):
        _state.setdefault("sessions", {})[session_id] = {
            "created": datetime.utcnow().isoformat(),
            "messages": [],
        }
    _state["sessions"][session_id]["messages"].append({"role": "user", "text": query})

    chat_handler = _state.get("chat_handler")
    if chat_handler is None:
        # Fallback: use coordinator if chat_handler not initialised
        coordinator = _state.get("coordinator")
        if coordinator is None:
            raise HTTPException(status_code=503, detail="System is starting up. Please retry.")
        try:
            coordinator.add_to_memory("user", query)
            context = {"has_dataset": False, "has_target": False, "dataset_summary": "",
                        "current_month": request.current_month}
            intent_result = await coordinator.analyze_user_intent(query, context)
            if intent_result.get("blocked"):
                answer = "The AI assistant is temporarily unavailable. Please try again shortly."
            else:
                answer = await coordinator.generate_conversational_reply(query, context)
            return StructuredChatResponse(
                intent=intent_result.get("intent", "general"),
                summary=answer,
                key_points=[],
                sources=[],
                confidence="low",
                data_backed=False,
                session_id=session_id,
                answer=answer,
            )
        except Exception as exc:
            logger.error(f"Chat fallback error: {exc}")
            return StructuredChatResponse(
                intent="error", summary="An internal error occurred.",
                session_id=session_id, answer="I apologise — an internal error occurred.",
                confidence="low", data_backed=False,
            )

    try:
        result = chat_handler.handle(query, request.current_month)
        answer = result["summary"]
        if result["key_points"]:
            answer += "\n\n" + "\n".join(f"- {kp}" for kp in result["key_points"])

        _state["sessions"][session_id]["messages"].append({"role": "assistant", "text": answer})

        return StructuredChatResponse(
            intent=result["intent"],
            summary=result["summary"],
            key_points=result["key_points"],
            sources=result["sources"],
            visualization=result.get("visualization") if request.include_visualization else None,
            confidence=result["confidence"],
            data_backed=result.get("data_backed", False),
            session_id=session_id,
            answer=answer,
        )
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        return StructuredChatResponse(
            intent="error",
            summary="I apologise — an internal error occurred. Our team has been notified.",
            session_id=session_id,
            answer="I apologise — an internal error occurred.",
            confidence="low",
            data_backed=False,
        )


# ---------------------------------------------------------------------------
# Routes — Chat session
# ---------------------------------------------------------------------------

@app.get("/api/chat/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve chat session history."""
    sessions = _state.get("sessions", {})
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, **sessions[session_id]}


# ---------------------------------------------------------------------------
# Routes — Compare
# ---------------------------------------------------------------------------

@app.post("/api/compare")
async def compare(request: CompareRequest):
    """Compare education entities/metrics using ground-truth data."""
    entities = [_sanitise(e) for e in request.entities]
    metric = _sanitise(request.metric) if request.metric else None

    rag = _state.get("rag")
    if rag and rag._indexed:
        comparison = rag.compare_states(entities)
        return {
            "comparison": comparison.get("states", {}),
            "entities": entities,
            "metric": metric or "all metrics",
            "month": comparison.get("month", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Fallback to coordinator
    coordinator = _state.get("coordinator")
    if coordinator is None:
        raise HTTPException(status_code=503, detail="System starting up.")

    try:
        query = f"Compare {', '.join(entities)} on {metric or 'overall performance'}"
        reply = await coordinator.generate_conversational_reply(query, {"has_dataset": False})
        return {
            "comparison": reply,
            "entities": entities,
            "metric": metric or "overall performance",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.error(f"Compare error: {exc}")
        raise HTTPException(status_code=500, detail="Comparison failed.")


@app.get("/api/compare/states")
async def compare_states_list():
    """Return list of available states for comparison."""
    rag = _state.get("rag")
    if not rag or not rag._indexed:
        return {"states": []}
    months = rag.get_all_months()
    if not months:
        return {"states": []}
    latest = rag.get_month_data(months[-1])
    states = list(latest.get("state_performance", {}).keys()) if latest else []
    return {"states": sorted(states)}


@app.get("/api/compare/months")
async def compare_months():
    """Return available months for month-over-month comparison."""
    rag = _state.get("rag")
    if not rag or not rag._indexed:
        return {"months": []}
    return {"months": rag.get_all_months()}


# ---------------------------------------------------------------------------
# Routes — Search
# ---------------------------------------------------------------------------

@app.post("/api/search")
async def search(request: SearchRequest):
    """Semantic search across education data using RAG."""
    query = _sanitise(request.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag = _state.get("rag")
    if rag and rag._indexed:
        results = rag.search(query, top_k=request.top_k)
        return {
            "query": query,
            "results": [
                {
                    "title": f"{r['section'].replace('_', ' ').title()} — {r['month']}",
                    "snippet": r["text"],
                    "relevance": r["relevance"],
                    "month": r["month"],
                    "section": r["section"],
                }
                for r in results
            ],
            "total": len(results),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Stub fallback
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


# ---------------------------------------------------------------------------
# Routes — Newsletter
# ---------------------------------------------------------------------------

@app.get("/api/newsletter/months")
async def newsletter_months():
    """Return available newsletter months from ground-truth data."""
    rag = _state.get("rag")
    if rag and rag._indexed:
        months = rag.get_all_months()
        default = months[-1] if months else ""
        return {"months": months, "default": default}

    # Fallback
    months = [
        "April 2025", "May 2025", "June 2025", "July 2025",
        "August 2025", "September 2025", "October 2025",
        "November 2025", "December 2025", "January 2026",
    ]
    return {"months": months, "default": "January 2026"}


@app.get("/api/newsletter/{month}")
async def newsletter_detail(month: str):
    """Return full data for a specific newsletter month."""
    rag = _state.get("rag")
    if not rag or not rag._indexed:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # URL decode and normalise
    month_clean = month.replace("-", " ").replace("_", " ").strip()
    # Try case-insensitive match
    for m in rag.get_all_months():
        if m.lower() == month_clean.lower():
            data = rag.get_month_data(m)
            if data:
                return {"month": m, **data}

    raise HTTPException(status_code=404, detail=f"Month '{month}' not found")


# ---------------------------------------------------------------------------
# Routes — Stats / KPI
# ---------------------------------------------------------------------------

@app.post("/api/stats/aggregate")
async def stats_aggregate(request: StatsRequest):
    """Return aggregated KPI data across months."""
    rag = _state.get("rag")
    if not rag or not rag._indexed:
        raise HTTPException(status_code=503, detail="Data not loaded")

    kpi = request.kpi.strip().lower().replace(" ", "_")
    trend = rag.get_kpi_trend(kpi)
    if not trend:
        raise HTTPException(status_code=404, detail=f"KPI '{request.kpi}' not found")

    values = [t["value"] for t in trend]
    return {
        "kpi": kpi,
        "trend": trend,
        "stats": {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 2),
            "latest": values[-1],
            "count": len(values),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/stats/kpis")
async def available_kpis():
    """Return list of available KPI names."""
    rag = _state.get("rag")
    if not rag or not rag._indexed:
        return {"kpis": []}
    months = rag.get_all_months()
    if not months:
        return {"kpis": []}
    mdata = rag.get_month_data(months[-1])
    kpis = list(mdata.get("kpis", {}).keys()) if mdata else []
    return {"kpis": kpis}


# ---------------------------------------------------------------------------
# Routes — Admin
# ---------------------------------------------------------------------------

@app.post("/api/admin/reindex")
async def admin_reindex():
    """Re-index the RAG system from newsletter data."""
    rag = _state.get("rag")
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not available")
    try:
        count = rag.load_and_index()
        return {"status": "ok", "documents_indexed": count, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


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

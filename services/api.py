"""FastAPI backend for the multi-agent data science platform."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import Orchestrator
from config.settings import get_settings
from services.metadata import MetadataService

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

orchestrator: Optional[Orchestrator] = None
metadata_svc: Optional[MetadataService] = None


def _serialize(obj: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serialisable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise global services on startup."""
    global orchestrator, metadata_svc
    logger.info("Starting up – initialising Orchestrator and MetadataService")
    orchestrator = Orchestrator()
    try:
        metadata_svc = MetadataService()
        metadata_svc.init_db()
        logger.info("MetadataService initialised")
    except Exception as exc:
        logger.warning("MetadataService unavailable: %s", exc)
        metadata_svc = None
    os.makedirs(get_settings().upload_dir, exist_ok=True)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="DataSciencePlatform API",
    version=get_settings().version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    """Request body for running the pipeline."""
    file_path: str
    target_col: Optional[str] = None


class PipelineResponse(BaseModel):
    """Summarised pipeline result."""
    pipeline_state: Dict[str, str]
    step_results: Dict[str, Any]
    final_insights: Optional[Any] = None


class UploadResponse(BaseModel):
    """Response after uploading a file."""
    file_path: str
    file_name: str
    size_bytes: int


class HealthResponse(BaseModel):
    """Health-check payload."""
    status: str
    version: str


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Return service health and version."""
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.version)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file (CSV, Excel, Parquet, JSON).

    The file is saved to the configured uploads directory and basic
    metadata is returned.
    """
    allowed = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}",
        )

    upload_dir = get_settings().upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, file.filename or "upload")

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    logger.info("Uploaded %s (%d bytes)", dest, len(contents))
    return UploadResponse(
        file_path=dest,
        file_name=file.filename or "upload",
        size_bytes=len(contents),
    )


@app.post("/api/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """Execute the full data-science pipeline on a dataset.

    Stores dataset and experiment metadata in the database when the
    metadata service is available.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    try:
        raw = orchestrator.run_pipeline(req.file_path, req.target_col)
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if metadata_svc is not None:
        try:
            strategy_res = orchestrator.get_step_result("strategy")
            ingest_data = strategy_res.data if strategy_res and strategy_res.success else {}
            profile = ingest_data.get("profile", {})
            stats = profile.get("stats", {})
            ds_id = metadata_svc.save_dataset(
                name=os.path.basename(req.file_path),
                rows=stats.get("row_count", 0),
                cols=stats.get("column_count", 0),
                target=req.target_col,
                problem_type=profile.get("problem_type"),
                profile_json=_serialize(profile),
            )
            model_res = orchestrator.get_step_result("modeling")
            if model_res and model_res.success:
                eval_res = orchestrator.get_step_result("evaluate")
                metrics = _serialize(eval_res.data) if eval_res and eval_res.success else None
                metadata_svc.save_experiment(
                    dataset_id=ds_id,
                    model_type=model_res.data.get("champion_name", "unknown"),
                    params_json=_serialize(model_res.data.get("champion_params")),
                    metrics_json=metrics,
                    status="completed",
                )
        except Exception as exc:
            logger.warning("Failed to store metadata: %s", exc)

    serialized = _serialize(raw)
    return PipelineResponse(**serialized)


@app.get("/api/pipeline/state")
async def pipeline_state():
    """Return the current state of every pipeline step."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    return orchestrator.get_pipeline_state()


@app.get("/api/pipeline/step/{step_name}")
async def pipeline_step(step_name: str):
    """Return the result of a specific pipeline step."""
    valid = {"strategy", "engineering", "exploration", "modeling", "mlops"}
    if step_name not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid step '{step_name}'. Valid: {', '.join(sorted(valid))}",
        )
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")

    result = orchestrator.get_step_result(step_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result for step '{step_name}'")

    return _serialize({
        "step": step_name,
        "success": result.success,
        "errors": result.errors,
        "data": result.data,
        "metadata": result.metadata,
    })


@app.get("/api/datasets")
async def list_datasets():
    """List all datasets stored in the metadata database."""
    if metadata_svc is None:
        raise HTTPException(status_code=503, detail="MetadataService not available")
    try:
        with metadata_svc._connect() as conn:
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM datasets ORDER BY created_at DESC;")
                rows = cur.fetchall()
        return [_serialize(dict(r)) for r in rows]
    except Exception as exc:
        logger.error("Failed to list datasets: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/experiments")
async def list_experiments(dataset_id: int = Query(..., description="Dataset ID")):
    """List experiments for a specific dataset."""
    if metadata_svc is None:
        raise HTTPException(status_code=503, detail="MetadataService not available")
    try:
        rows = metadata_svc.get_experiments(dataset_id)
        return [_serialize(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to list experiments: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("services.api:app", host="0.0.0.0", port=8000)

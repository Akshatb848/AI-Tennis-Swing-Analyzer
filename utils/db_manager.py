import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import datetime


def get_engine():
    db_url = st.secrets["DATABASE_URL"]

    return create_engine(
        db_url,
        poolclass=NullPool,      # 🔥 REQUIRED FOR NEON
        pool_pre_ping=True,
        future=True,
    )


def ensure_project_exists(project_id: str, description: str = "Auto-created project"):
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS project_metadata (
                project_id VARCHAR PRIMARY KEY,
                description TEXT,
                created_at TIMESTAMP
            )
        """))

        conn.execute(text("""
            INSERT INTO project_metadata (project_id, description, created_at)
            VALUES (:pid, :desc, :ts)
            ON CONFLICT (project_id) DO NOTHING
        """), {
            "pid": project_id,
            "desc": description,
            "ts": datetime.datetime.utcnow()
        })


def log_agent_start(project_id: str, agent_name: str):
    engine = get_engine()
    ensure_project_exists(project_id)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO agent_status (project_id, agent_name, status, started_at)
            VALUES (:pid, :agent, 'running', :ts)
        """), {
            "pid": project_id,
            "agent": agent_name,
            "ts": datetime.datetime.utcnow()
        })


def log_agent_success(project_id: str, agent_name: str, output_location: str | None):
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE agent_status
            SET status = 'completed',
                completed_at = :ts,
                output_location = :out
            WHERE project_id = :pid
              AND agent_name = :agent
              AND status = 'running'
        """), {
            "pid": project_id,
            "agent": agent_name,
            "ts": datetime.datetime.utcnow(),
            "out": output_location
        })


def log_agent_failure(project_id: str, agent_name: str, error_message: str):
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE agent_status
            SET status = 'failed',
                completed_at = :ts,
                error_message = :err
            WHERE project_id = :pid
              AND agent_name = :agent
              AND status = 'running'
        """), {
            "pid": project_id,
            "agent": agent_name,
            "ts": datetime.datetime.utcnow(),
            "err": error_message
        })

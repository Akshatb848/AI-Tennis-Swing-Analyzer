import sqlalchemy as sa
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st

from sqlalchemy.pool import NullPool


def get_engine():
    try:
        conn = st.secrets["connections"]["postgresql"]
        url = (
            f"postgresql://{conn['username']}:{conn['password']}"
            f"@{conn['host']}:{conn['port']}/{conn['database']}"
        )
    except Exception:
        from os import environ
        url = environ.get("DATABASE_URL")

    return create_engine(
        url,
        poolclass=NullPool,          # CRITICAL FOR NEON
        pool_pre_ping=True,          # validates connection before use
        connect_args={
            "sslmode": "require"     # Neon requires SSL
        },
    )

    return engine


def init_db():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sa.text("""
        -- Project metadata
        CREATE TABLE IF NOT EXISTS project_metadata (
            project_id VARCHAR(50) PRIMARY KEY,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Agent execution tracking
        CREATE TABLE IF NOT EXISTS agent_status (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(50) REFERENCES project_metadata(project_id),
            agent_name VARCHAR(50),
            status VARCHAR(20),
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            output_location TEXT,
            error_message TEXT
        );

        -- Cleaned data storage
        CREATE TABLE IF NOT EXISTS cleaned_data (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(50) REFERENCES project_metadata(project_id),
            data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Feature store
        CREATE TABLE IF NOT EXISTS feature_store (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(50) REFERENCES project_metadata(project_id),
            features JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Model recommendations (AutoML agent)
        CREATE TABLE IF NOT EXISTS model_recommendations (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(50) REFERENCES project_metadata(project_id),
            problem_type TEXT,
            recommended_models TEXT,
            rationale TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Model training results  ✅ THIS FIXES YOUR ERROR
        CREATE TABLE IF NOT EXISTS model_results (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(50) REFERENCES project_metadata(project_id),
            model_name TEXT,
            metrics JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
                             
        CREATE TABLE IF NOT EXISTS raw_data (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(50) REFERENCES project_metadata(project_id),
        data JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS eda_reports (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(50),
        summary JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS data_warehouse (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(50),
        table_name TEXT,
        row_count INT,
        columns JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
                             
        CREATE TABLE IF NOT EXISTS visualizations (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(50),
        plot_type VARCHAR(50),
        plot_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        UPDATE agent_status
        SET status='pending', error_message=NULL
        WHERE project_id=:pid AND status='failed'
        """)({"pid": project_id}))


from datetime import datetime


def ensure_project_exists(project_id: str, description: str = ""):
    """
    Ensure a project exists in project_metadata.
    Safe to call multiple times.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO project_metadata (project_id, description)
            VALUES (:project_id, :description)
            ON CONFLICT (project_id) DO NOTHING
        """), {
            "project_id": project_id,
            "description": description,
        })

def log_agent_start(project_id: str, agent_name: str):
    """
    Log the start of an agent execution.
    """
    ensure_project_exists(project_id)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO agent_status (
                project_id,
                agent_name,
                status,
                started_at
            )
            VALUES (
                :project_id,
                :agent_name,
                'running',
                :started_at
            )
        """), {
            "project_id": project_id,
            "agent_name": agent_name,
            "started_at": datetime.utcnow(),
        })


def log_agent_success(
    project_id: str,
    agent_name: str,
    output_location: str | None = None,
):
    """
    Mark an agent as completed successfully.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sa.text("""
            UPDATE agent_status
            SET
                status = 'completed',
                completed_at = :completed_at,
                output_location = :output_location,
                error_message = NULL
            WHERE
                project_id = :project_id
                AND agent_name = :agent_name
                AND status = 'running'
        """), {
            "project_id": project_id,
            "agent_name": agent_name,
            "completed_at": datetime.utcnow(),
            "output_location": output_location,
        })


def log_agent_failure(
    project_id: str,
    agent_name: str,
    error_message: str,
):
    """
    Mark an agent as failed.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sa.text("""
            UPDATE agent_status
            SET
                status = 'failed',
                completed_at = :completed_at,
                error_message = :error_message
            WHERE
                project_id = :project_id
                AND agent_name = :agent_name
                AND status = 'running'
        """), {
            "project_id": project_id,
            "agent_name": agent_name,
            "completed_at": datetime.utcnow(),
            "error_message": error_message,
        })



def save_df_to_table(df: pd.DataFrame, table_name: str, project_id: str):
    """
    Save a pandas DataFrame into a Neon table.
    """
    engine = get_engine()
    df = df.copy()
    df["project_id"] = project_id

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )


def query_table(table_name: str, project_id: str) -> pd.DataFrame:
    """
    Query records for a given project_id from Neon PostgreSQL.
    """
    engine = get_engine()

    query = sa.text(f"""
        SELECT *
        FROM {table_name}
        WHERE project_id = :project_id
    """)

    return pd.read_sql(query, engine, params={"project_id": project_id})

def reset_failed_agents(project_id: str):
    """
    Reset failed agents so pipeline can be retried safely.
    """
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            sa.text("""
                UPDATE agent_status
                SET
                    status = 'pending',
                    error_message = NULL,
                    started_at = NULL,
                    completed_at = NULL
                WHERE
                    project_id = :pid
                    AND status = 'failed'
            """),
            {"pid": project_id}
        )
def load_df(query: str, params: tuple | None = None):
    """
    Load SQL query results into a pandas DataFrame.
    """
    engine = get_engine()
    return pd.read_sql(query, engine, params=params)

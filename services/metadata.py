import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

from config.settings import LOG_FORMAT

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class MetadataService:
    """Manages experiment and dataset metadata in PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._database_url = database_url or os.environ.get("DATABASE_URL", "")
        if not self._database_url:
            raise ValueError("DATABASE_URL is not set")

    def _connect(self) -> psycopg2.extensions.connection:
        """Open a new database connection."""
        return psycopg2.connect(self._database_url)

    def init_db(self) -> None:
        """Create the required tables if they do not already exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS datasets (
            id            SERIAL PRIMARY KEY,
            name          TEXT NOT NULL,
            rows          INTEGER NOT NULL,
            cols          INTEGER NOT NULL,
            target        TEXT,
            problem_type  TEXT,
            profile_json  JSONB,
            created_at    TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS experiments (
            id            SERIAL PRIMARY KEY,
            dataset_id    INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            model_type    TEXT NOT NULL,
            params_json   JSONB,
            metrics_json  JSONB,
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS model_artifacts (
            id              SERIAL PRIMARY KEY,
            experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            artifact_path   TEXT NOT NULL,
            created_at      TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        logger.info("Database tables initialised successfully")

    def save_dataset(
        self,
        name: str,
        rows: int,
        cols: int,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        profile_json: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a new dataset record and return its id.

        Args:
            name: Human-readable dataset name.
            rows: Number of rows in the dataset.
            cols: Number of columns in the dataset.
            target: Optional target column name.
            problem_type: Optional classification/regression label.
            profile_json: Optional profiling statistics.

        Returns:
            The auto-generated dataset id.
        """
        sql = """
            INSERT INTO datasets (name, rows, cols, target, problem_type, profile_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        profile = json.dumps(profile_json) if profile_json else None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (name, rows, cols, target, problem_type, profile))
                dataset_id: int = cur.fetchone()[0]
            conn.commit()
        logger.info("Saved dataset '%s' with id %d", name, dataset_id)
        return dataset_id

    def get_dataset(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single dataset by id.

        Args:
            dataset_id: Primary key of the dataset.

        Returns:
            A dict with dataset fields, or None if not found.
        """
        sql = "SELECT * FROM datasets WHERE id = %s;"
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (dataset_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def save_experiment(
        self,
        dataset_id: int,
        model_type: str,
        params_json: Optional[Dict[str, Any]] = None,
        metrics_json: Optional[Dict[str, Any]] = None,
        status: str = "pending",
    ) -> int:
        """Insert a new experiment record and return its id.

        Args:
            dataset_id: Foreign key to the datasets table.
            model_type: Name / type of the model.
            params_json: Optional hyper-parameter dict.
            metrics_json: Optional evaluation metrics dict.
            status: Experiment status string.

        Returns:
            The auto-generated experiment id.
        """
        sql = """
            INSERT INTO experiments (dataset_id, model_type, params_json, metrics_json, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """
        params = json.dumps(params_json) if params_json else None
        metrics = json.dumps(metrics_json) if metrics_json else None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (dataset_id, model_type, params, metrics, status))
                experiment_id: int = cur.fetchone()[0]
            conn.commit()
        logger.info("Saved experiment (model=%s) with id %d", model_type, experiment_id)
        return experiment_id

    def get_experiments(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Return all experiments linked to a dataset.

        Args:
            dataset_id: Foreign key to the datasets table.

        Returns:
            A list of experiment dicts.
        """
        sql = "SELECT * FROM experiments WHERE dataset_id = %s ORDER BY created_at DESC;"
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (dataset_id,))
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def save_artifact(self, experiment_id: int, artifact_path: str) -> int:
        """Store a model artifact reference and return its id.

        Args:
            experiment_id: Foreign key to the experiments table.
            artifact_path: Filesystem path to the serialised model.

        Returns:
            The auto-generated artifact id.
        """
        sql = """
            INSERT INTO model_artifacts (experiment_id, artifact_path)
            VALUES (%s, %s)
            RETURNING id;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (experiment_id, artifact_path))
                artifact_id: int = cur.fetchone()[0]
            conn.commit()
        logger.info("Saved artifact for experiment %d at '%s'", experiment_id, artifact_path)
        return artifact_id

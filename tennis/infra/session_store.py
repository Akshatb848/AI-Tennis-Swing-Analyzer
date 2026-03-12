"""
Session Store — Persistent storage for sessions, matches, events.

Replaces all in-memory dicts with a persistent backend.
Uses SQLite for development, extensible to PostgreSQL for production.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Persistent storage for TennisIQ sessions.

    Stores sessions, match states, events, and video metadata.
    Thread-safe with connection-per-thread pattern.
    """

    def __init__(self, db_path: str = "./data/tennisiq.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                mode TEXT DEFAULT 'match',
                status TEXT DEFAULT 'created',
                player_names TEXT DEFAULT '[]',
                court_surface TEXT DEFAULT 'hard',
                match_config TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                summary TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                player1_name TEXT,
                player2_name TEXT,
                match_format TEXT DEFAULT 'best_of_3',
                status TEXT DEFAULT 'in_progress',
                score_state TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                frame_number INTEGER DEFAULT 0,
                timestamp_ms INTEGER DEFAULT 0,
                data TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS video_jobs (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                source_path TEXT,
                status TEXT DEFAULT 'uploaded',
                progress REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT DEFAULT '',
                resource_id TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_log(user_id);
        """)
        conn.commit()
        logger.info("Session store initialized at %s", self.db_path)

    # ── Session CRUD ─────────────────────────────────────────────────────────

    def save_session(
        self, session_id: str, mode: str = "match", status: str = "created",
        player_names: Optional[list] = None, court_surface: str = "hard",
        match_config: Optional[dict] = None, summary: Optional[dict] = None,
    ):
        """Save or update a session."""
        conn = self._conn
        conn.execute("""
            INSERT OR REPLACE INTO sessions (id, mode, status, player_names, court_surface, match_config, summary, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, mode, status,
            json.dumps(player_names or []),
            court_surface,
            json.dumps(match_config or {}),
            json.dumps(summary or {}),
            datetime.utcnow().isoformat(),
        ))
        conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def list_sessions(
        self, limit: int = 20, offset: int = 0, status: Optional[str] = None,
    ) -> list[dict]:
        """List sessions with optional filtering."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_session_status(self, session_id: str, status: str):
        """Update session status."""
        self._conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.utcnow().isoformat(), session_id),
        )
        self._conn.commit()

    def save_session_summary(self, session_id: str, summary: dict):
        """Save session summary (post-match results)."""
        self._conn.execute(
            "UPDATE sessions SET summary = ?, status = 'completed', updated_at = ? WHERE id = ?",
            (json.dumps(summary), datetime.utcnow().isoformat(), session_id),
        )
        self._conn.commit()

    # ── Events ───────────────────────────────────────────────────────────────

    def save_event(
        self, session_id: str, event_type: str,
        frame_number: int = 0, timestamp_ms: int = 0,
        data: Optional[dict] = None,
    ):
        """Save a processing event."""
        self._conn.execute(
            "INSERT INTO events (session_id, event_type, frame_number, timestamp_ms, data) VALUES (?, ?, ?, ?, ?)",
            (session_id, event_type, frame_number, timestamp_ms, json.dumps(data or {})),
        )
        self._conn.commit()

    def get_events(
        self, session_id: str, event_type: Optional[str] = None, limit: int = 1000,
    ) -> list[dict]:
        """Get events for a session."""
        if event_type:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? AND event_type = ? ORDER BY frame_number LIMIT ?",
                (session_id, event_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY frame_number LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Usage tracking ───────────────────────────────────────────────────────

    def log_usage(self, user_id: str, action: str, resource_type: str = "", resource_id: str = ""):
        """Log a user action for usage tracking."""
        self._conn.execute(
            "INSERT INTO usage_log (user_id, action, resource_type, resource_id) VALUES (?, ?, ?, ?)",
            (user_id, action, resource_type, resource_id),
        )
        self._conn.commit()

    def get_user_usage(self, user_id: str, since: Optional[str] = None) -> dict:
        """Get usage statistics for a user."""
        if since:
            rows = self._conn.execute(
                "SELECT action, COUNT(*) as count FROM usage_log WHERE user_id = ? AND created_at >= ? GROUP BY action",
                (user_id, since),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT action, COUNT(*) as count FROM usage_log WHERE user_id = ? GROUP BY action",
                (user_id,),
            ).fetchall()
        return {row["action"]: row["count"] for row in rows}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row) -> dict:
        """Convert a sqlite3.Row to dict with JSON parsing."""
        d = dict(row)
        for key in ("player_names", "match_config", "summary", "metadata", "data", "score_state"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

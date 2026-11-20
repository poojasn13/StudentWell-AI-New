"""
Phase 9 — Student Evaluator Rating Store
==========================================
Saves and retrieves student ratings of chatbot responses.

Storage: SQLite database at data/evaluations.db

Schema
------
ratings table:
  id               INTEGER PRIMARY KEY
  eval_id          TEXT     — unique UUID for this rating
  session_id       TEXT     — conversation session ID
  system           TEXT     — "normal" | "emotion_aware"
  turn_number      INTEGER
  user_message     TEXT     — the student message being rated
  response_text    TEXT     — the response being rated
  emotion          TEXT     — detected emotion at time of response
  empathy_rating   INTEGER  — 1-5
  relevance_rating INTEGER  — 1-5
  usefulness_rating INTEGER — 1-5
  tone_rating      INTEGER  — 1-5
  feedback_text    TEXT     — optional written comment
  timestamp        TEXT     — ISO datetime
  is_demo          INTEGER  — 1 if this is demo/test data, 0 if real student rating
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = Path("data/evaluations.db")

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the ratings table if it does not exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id           TEXT    NOT NULL UNIQUE,
                session_id        TEXT    NOT NULL,
                system            TEXT    NOT NULL,
                turn_number       INTEGER DEFAULT 0,
                user_message      TEXT    DEFAULT '',
                response_text     TEXT    DEFAULT '',
                emotion           TEXT    DEFAULT 'neutral',
                empathy_rating    INTEGER NOT NULL,
                relevance_rating  INTEGER NOT NULL,
                usefulness_rating INTEGER NOT NULL,
                tone_rating       INTEGER NOT NULL,
                feedback_text     TEXT    DEFAULT '',
                timestamp         TEXT    NOT NULL,
                is_demo           INTEGER DEFAULT 0
            )
        """)
        conn.commit()
    log.debug("Database initialised")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_rating(
    session_id:        str,
    system:            str,
    empathy_rating:    int,
    relevance_rating:  int,
    usefulness_rating: int,
    tone_rating:       int,
    turn_number:       int         = 0,
    user_message:      str         = "",
    response_text:     str         = "",
    emotion:           str         = "neutral",
    feedback_text:     str         = "",
    is_demo:           bool        = False,
) -> str:
    """
    Save a student rating.  Returns the eval_id.

    Parameters
    ----------
    session_id        : conversation session ID
    system            : "normal" or "emotion_aware"
    empathy_rating    : 1–5
    relevance_rating  : 1–5
    usefulness_rating : 1–5
    tone_rating       : 1–5
    turn_number       : which turn in the conversation
    user_message      : the student's question (for context)
    response_text     : the response being rated
    emotion           : detected emotion at time of rating
    feedback_text     : optional written comment
    is_demo           : True for demo/test entries
    """
    # Validate rating values
    for name, val in [
        ("empathy", empathy_rating),
        ("relevance", relevance_rating),
        ("usefulness", usefulness_rating),
        ("tone", tone_rating),
    ]:
        if not (1 <= val <= 5):
            raise ValueError(f"{name}_rating must be 1–5, got {val}")

    eval_id   = "eval_" + uuid.uuid4().hex[:12]
    timestamp = datetime.utcnow().isoformat()

    init_db()
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO ratings (
                eval_id, session_id, system, turn_number,
                user_message, response_text, emotion,
                empathy_rating, relevance_rating, usefulness_rating, tone_rating,
                feedback_text, timestamp, is_demo
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            eval_id, session_id, system, turn_number,
            user_message[:500], response_text[:1000], emotion,
            empathy_rating, relevance_rating, usefulness_rating, tone_rating,
            feedback_text[:500], timestamp, int(is_demo),
        ))
        conn.commit()

    log.info(f"Rating saved: {eval_id} system={system} empathy={empathy_rating} relevance={relevance_rating}")
    return eval_id


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_all_ratings(include_demo: bool = True) -> pd.DataFrame:
    """Return all ratings as a DataFrame."""
    init_db()
    with _get_conn() as conn:
        query = "SELECT * FROM ratings"
        if not include_demo:
            query += " WHERE is_demo = 0"
        query += " ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
    return df


def get_ratings_by_system(system: str, include_demo: bool = True) -> pd.DataFrame:
    """Return ratings for one system ('normal' or 'emotion_aware')."""
    init_db()
    with _get_conn() as conn:
        query = "SELECT * FROM ratings WHERE system = ?"
        params = [system]
        if not include_demo:
            query += " AND is_demo = 0"
        df = pd.read_sql_query(query, conn, params=params)
    return df


def get_summary_stats(include_demo: bool = True) -> dict:
    """
    Return aggregate statistics for the analytics dashboard.
    """
    df = get_all_ratings(include_demo=include_demo)
    if df.empty:
        return {
            "total_ratings": 0,
            "normal":        {"count": 0, "empathy": 0, "relevance": 0, "usefulness": 0, "tone": 0, "avg_overall": 0},
            "emotion_aware": {"count": 0, "empathy": 0, "relevance": 0, "usefulness": 0, "tone": 0, "avg_overall": 0},
        }

    def system_stats(sdf: pd.DataFrame) -> dict:
        if sdf.empty:
            return {"count": 0, "empathy": 0, "relevance": 0, "usefulness": 0, "tone": 0, "avg_overall": 0}
        return {
            "count":       len(sdf),
            "empathy":     round(sdf["empathy_rating"].mean(),    2),
            "relevance":   round(sdf["relevance_rating"].mean(),  2),
            "usefulness":  round(sdf["usefulness_rating"].mean(), 2),
            "tone":        round(sdf["tone_rating"].mean(),       2),
            "avg_overall": round(
                (sdf["empathy_rating"] + sdf["relevance_rating"] +
                 sdf["usefulness_rating"] + sdf["tone_rating"]).mean() / 4, 2
            ),
        }

    return {
        "total_ratings": len(df),
        "normal":        system_stats(df[df["system"] == "normal"]),
        "emotion_aware": system_stats(df[df["system"] == "emotion_aware"]),
    }


def get_feedback_comments(limit: int = 50, include_demo: bool = True) -> list[dict]:
    """Return ratings that have written feedback comments."""
    df = get_all_ratings(include_demo=include_demo)
    if df.empty:
        return []
    comments = df[df["feedback_text"].str.strip() != ""][
        ["eval_id", "system", "emotion", "feedback_text", "timestamp"]
    ].head(limit)
    return comments.to_dict("records")


def get_total_count() -> int:
    init_db()
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM ratings").fetchone()
        return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()

    # Save two demo ratings
    id1 = save_rating(
        session_id="test_session_001",
        system="emotion_aware",
        empathy_rating=5,
        relevance_rating=4,
        usefulness_rating=4,
        tone_rating=5,
        emotion="fear",
        feedback_text="The response felt very understanding.",
        is_demo=True,
    )
    id2 = save_rating(
        session_id="test_session_001",
        system="normal",
        empathy_rating=3,
        relevance_rating=4,
        usefulness_rating=3,
        tone_rating=3,
        emotion="neutral",
        feedback_text="A bit generic, but helpful.",
        is_demo=True,
    )
    print(f"Saved: {id1}, {id2}")

    stats = get_summary_stats()
    print(f"\nSummary stats:")
    print(f"  Total ratings: {stats['total_ratings']}")
    print(f"  Emotion-aware: {stats['emotion_aware']}")
    print(f"  Normal:        {stats['normal']}")

    df = get_all_ratings()
    print(f"\nAll ratings ({len(df)} rows):")
    print(df[["eval_id", "system", "empathy_rating", "relevance_rating", "tone_rating"]].to_string(index=False))

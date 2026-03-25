"""Local SQLite persistence for meal entries and food items."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timedelta
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_DB_NAME = "nutrivoice.db"


def _db_path() -> Path:
    env = os.environ.get("NUTRIVOICE_DB_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / DEFAULT_DB_NAME


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    if own:
        conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                meal_type TEXT,
                source TEXT NOT NULL,
                raw_input TEXT,
                transcript TEXT
            );

            CREATE TABLE IF NOT EXISTS food_items (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
                food_name TEXT NOT NULL,
                quantity REAL,
                unit TEXT,
                calories REAL NOT NULL DEFAULT 0,
                protein REAL NOT NULL DEFAULT 0,
                carbs REAL NOT NULL DEFAULT 0,
                fats REAL NOT NULL DEFAULT 0,
                nutrition_source TEXT NOT NULL,
                confidence_status TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_entries_created_at ON entries(created_at);
            CREATE INDEX IF NOT EXISTS idx_food_items_entry_id ON food_items(entry_id);

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        if own:
            conn.close()


def save_goals(goals: dict[str, float | int]) -> None:
    init_schema()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("goals", json.dumps(goals)),
        )
        conn.commit()
    finally:
        conn.close()


def load_goals(default: dict[str, float | int]) -> dict[str, float | int]:
    init_schema()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", ("goals",)
        ).fetchone()
    if not row:
        return dict(default)
    try:
        data = json.loads(row["value"])
        out = dict(default)
        for k in default:
            if k in data:
                out[k] = float(data[k]) if k != "Calories" else int(float(data[k]))
        return out
    except (json.JSONDecodeError, TypeError):
        return dict(default)


def _new_id() -> str:
    return uuid4().hex


def create_entry(
    *,
    created_at: datetime | None = None,
    meal_type: str | None,
    source: str,
    raw_input: str | None,
    transcript: str | None,
) -> str:
    init_schema()
    entry_id = _new_id()
    ts = (created_at or datetime.now()).replace(microsecond=0).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO entries (id, created_at, meal_type, source, raw_input, transcript)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entry_id, ts, meal_type, source, raw_input, transcript),
        )
        conn.commit()
    return entry_id


def insert_food_item(
    conn: sqlite3.Connection,
    *,
    entry_id: str,
    food_name: str,
    quantity: float | None,
    unit: str | None,
    calories: float,
    protein: float,
    carbs: float,
    fats: float,
    nutrition_source: str,
    confidence_status: str,
) -> str:
    item_id = _new_id()
    conn.execute(
        """
        INSERT INTO food_items (
            id, entry_id, food_name, quantity, unit,
            calories, protein, carbs, fats, nutrition_source, confidence_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            entry_id,
            food_name,
            quantity,
            unit,
            calories,
            protein,
            carbs,
            fats,
            nutrition_source,
            confidence_status,
        ),
    )
    return item_id


def add_meal_with_items(
    *,
    meal_type: str | None,
    source: str,
    raw_input: str | None,
    transcript: str | None,
    items: list[dict[str, Any]],
    created_at: datetime | None = None,
) -> str:
    """Insert one entry and its food rows. Returns entry id."""
    init_schema()
    entry_id = create_entry(
        created_at=created_at,
        meal_type=meal_type,
        source=source,
        raw_input=raw_input,
        transcript=transcript,
    )
    with get_connection() as conn:
        for it in items:
            insert_food_item(
                conn,
                entry_id=entry_id,
                food_name=str(it.get("Name") or it.get("food_name") or "Unknown"),
                quantity=_f(it.get("quantity")),
                unit=(str(it["unit"]) if it.get("unit") is not None else None),
                calories=float(it.get("Calories", 0) or 0),
                protein=float(it.get("Protein", 0) or 0),
                carbs=float(it.get("Carbs", 0) or 0),
                fats=float(it.get("Fats", 0) or 0),
                nutrition_source=str(it.get("nutrition_source", "llm_fallback")),
                confidence_status=str(it.get("confidence_status", "fallback")),
            )
        conn.commit()
    return entry_id


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _day_bounds_local(d: date) -> tuple[str, str]:
    start = datetime.combine(d, time.min)
    end = datetime.combine(d + timedelta(days=1), time.min)
    return start.isoformat(), end.isoformat()


def delete_entries_for_local_date(d: date) -> int:
    """Delete entries whose created_at falls on local calendar date d."""
    init_schema()
    start_s, end_s = _day_bounds_local(d)
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM entries WHERE created_at >= ? AND created_at < ?",
            (start_s, end_s),
        )
        conn.commit()
        return cur.rowcount


def fetch_items_in_range(start_day: date, end_day: date) -> list[dict[str, Any]]:
    """Return flat rows for dashboard; date range is inclusive of both days."""
    init_schema()
    start_s = datetime.combine(start_day, time.min).isoformat()
    end_exclusive = datetime.combine(end_day + timedelta(days=1), time.min).isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                e.id AS entry_id,
                e.created_at,
                e.meal_type,
                e.source,
                e.raw_input,
                e.transcript,
                f.id AS item_id,
                f.food_name,
                f.quantity,
                f.unit,
                f.calories,
                f.protein,
                f.carbs,
                f.fats,
                f.nutrition_source,
                f.confidence_status
            FROM entries e
            JOIN food_items f ON f.entry_id = e.id
            WHERE e.created_at >= ? AND e.created_at < ?
            ORDER BY e.created_at ASC, f.id ASC
            """,
            (start_s, end_exclusive),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "entry_id": r["entry_id"],
                "created_at": r["created_at"],
                "meal_type": r["meal_type"],
                "source": r["source"],
                "raw_input": r["raw_input"],
                "transcript": r["transcript"],
                "item_id": r["item_id"],
                "Name": r["food_name"],
                "quantity": r["quantity"],
                "unit": r["unit"],
                "Calories": r["calories"],
                "Protein": r["protein"],
                "Carbs": r["carbs"],
                "Fats": r["fats"],
                "nutrition_source": r["nutrition_source"],
                "confidence_status": r["confidence_status"],
            }
        )
    return out


def update_food_item_nutrition(
    item_id: str,
    *,
    calories: float,
    protein: float,
    carbs: float,
    fats: float,
    confidence_status: str = "matched",
) -> None:
    init_schema()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE food_items
            SET calories = ?, protein = ?, carbs = ?, fats = ?,
                nutrition_source = 'user_edited', confidence_status = ?
            WHERE id = ?
            """,
            (calories, protein, carbs, fats, confidence_status, item_id),
        )
        conn.commit()

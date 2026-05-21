import hashlib
import json
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

if os.environ.get("VERCEL"):
    DB_PATH = Path("/tmp") / "nrc.db"
else:
    DB_PATH = Path(__file__).parent.parent.parent / "data" / "nrc.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_number  TEXT PRIMARY KEY,
    title          TEXT,
    description    TEXT,
    account        TEXT,
    resources      TEXT,
    status         TEXT,
    created        TEXT,
    total_hours    REAL,
    billed_hours   REAL,
    sub_issue_type TEXT,
    issue_type     TEXT,
    imported_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tickets_account    ON tickets(account);
CREATE INDEX IF NOT EXISTS idx_tickets_issue_type ON tickets(issue_type);
CREATE INDEX IF NOT EXISTS idx_tickets_created    ON tickets(created);
CREATE INDEX IF NOT EXISTS idx_tickets_acct_issue ON tickets(account, issue_type);

CREATE TABLE IF NOT EXISTS recommendation_cache (
    cache_key      TEXT PRIMARY KEY,
    pattern_type   TEXT NOT NULL,
    account        TEXT NOT NULL,
    issue_type     TEXT NOT NULL,
    ticket_numbers TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    model          TEXT NOT NULL,
    created_at     TEXT DEFAULT (datetime('now'))
);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ── ticket store ───────────────────────────────────────────────────────────────

def import_tickets(df: pd.DataFrame, conn: sqlite3.Connection) -> tuple[int, int]:
    """Bulk-insert new tickets, skip duplicates. Returns (new_count, skipped_count)."""
    rows = [
        (
            row["ticket_number"], row["title"], row["description"],
            row["account"], row["resources"], row["status"],
            str(row["created"]), float(row["total_hours"]),
            float(row["billed_hours"]), row["sub_issue_type"], row["issue_type"],
        )
        for _, row in df.iterrows()
    ]

    existing_before = ticket_count(conn)
    conn.executemany(
        """
        INSERT OR IGNORE INTO tickets
            (ticket_number, title, description, account, resources,
             status, created, total_hours, billed_hours,
             sub_issue_type, issue_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    existing_after = ticket_count(conn)
    new_count = existing_after - existing_before
    skipped_count = len(rows) - new_count
    return new_count, skipped_count


def load_tickets(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load tickets from the store, optionally filtered to a date range (inclusive)."""
    query = "SELECT * FROM tickets"
    params = []
    
    if start_date and end_date:
        query += " WHERE created >= ? AND created <= ?"
        params.extend([start_date, end_date])
    elif start_date:
        query += " WHERE created >= ?"
        params.append(start_date)
    elif end_date:
        query += " WHERE created <= ?"
        params.append(end_date)
        
    query += " ORDER BY created"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["total_hours"] = pd.to_numeric(df["total_hours"], errors="coerce").fillna(0.0)
    df["billed_hours"] = pd.to_numeric(df["billed_hours"], errors="coerce").fillna(0.0)
    return df


def ticket_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def _format_date(date_str: Optional[str]) -> str:
    if not date_str:
        return "-"
    try:
        dt = date.fromisoformat(date_str[:10])
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{months[dt.month - 1]} {dt.day}, {dt.year}"
    except Exception:
        return date_str[:10] if date_str else "-"


def ticket_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT COUNT(*) as n, MIN(created) as min_dt, MAX(created) as max_dt FROM tickets").fetchone()
    return {
        "ticket_count": row["n"],
        "min_date": _format_date(row["min_dt"]) if row["n"] > 0 else "-",
        "max_date": _format_date(row["max_dt"]) if row["n"] > 0 else "-",
    }



def since_date_from_window(window_days: int) -> str:
    """Return ISO date string for N days ago."""
    return (date.today() - timedelta(days=window_days)).isoformat()


# ── historical context ─────────────────────────────────────────────────────────

def get_historical_context(
    account: str,
    issue_type: str,
    start_date: str,
    end_date: str,
    conn: sqlite3.Connection,
) -> dict:
    """Query full ticket history to produce trend context for the LLM prompt.

    Compares the detection window (recent period) against an equal-length prior
    period so the LLM knows whether this pattern is new, stable, or escalating.
    """
    # How many days is the detection window?
    d1 = date.fromisoformat(start_date[:10])
    d2 = date.fromisoformat(end_date[:10])
    window_days = max((d2 - d1).days, 1)
    prior_start = (d1 - timedelta(days=window_days)).isoformat()

    # Scope: specific issue_type or whole account if "(multiple)"
    if issue_type and issue_type != "(multiple)":
        scope_clause = "AND issue_type = ?"
        base_params: tuple = (account, issue_type)
    else:
        scope_clause = ""
        base_params = (account,)

    all_time = conn.execute(
        f"SELECT COUNT(*) as n, MIN(created) as first_seen "
        f"FROM tickets WHERE account = ? {scope_clause}",
        base_params,
    ).fetchone()

    recent = conn.execute(
        f"SELECT COUNT(*) FROM tickets "
        f"WHERE account = ? {scope_clause} AND created >= ? AND created <= ?",
        (*base_params, start_date, end_date),
    ).fetchone()[0]

    prior = conn.execute(
        f"SELECT COUNT(*) FROM tickets "
        f"WHERE account = ? {scope_clause} AND created >= ? AND created < ?",
        (*base_params, prior_start, start_date),
    ).fetchone()[0]

    if prior > 0:
        trend_pct = round((recent - prior) / prior * 100, 1)
        trend_label = (
            f"+{trend_pct}% vs prior {window_days}-day period (INCREASING)"
            if trend_pct > 10
            else f"{trend_pct}% vs prior period (DECREASING)"
            if trend_pct < -10
            else f"{trend_pct}% vs prior period (STABLE)"
        )
    else:
        trend_label = "No data in prior period (new or recently emerged pattern)"

    return {
        "all_time_count": all_time["n"],
        "first_seen": (all_time["first_seen"] or "")[:10],
        "recent_count": recent,
        "prior_count": prior,
        "trend_label": trend_label,
    }


# ── recommendation cache ───────────────────────────────────────────────────────

def _cache_key(pattern_type: str, account: str, issue_type: str,
               ticket_numbers: list[str], model: str) -> str:
    canonical = f"{pattern_type}|{account}|{issue_type}|{','.join(sorted(ticket_numbers))}|{model}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def cache_get(pattern_type: str, account: str, issue_type: str,
              ticket_numbers: list[str], model: str,
              conn: sqlite3.Connection) -> Optional[dict]:
    key = _cache_key(pattern_type, account, issue_type, ticket_numbers, model)
    row = conn.execute(
        "SELECT result_json FROM recommendation_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    return json.loads(row["result_json"]) if row else None


def cache_set(pattern_type: str, account: str, issue_type: str,
              ticket_numbers: list[str], model: str,
              result: dict, conn: sqlite3.Connection) -> None:
    key = _cache_key(pattern_type, account, issue_type, ticket_numbers, model)
    conn.execute(
        """
        INSERT OR REPLACE INTO recommendation_cache
            (cache_key, pattern_type, account, issue_type,
             ticket_numbers, result_json, model)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key, pattern_type, account, issue_type,
            json.dumps(sorted(ticket_numbers)),
            json.dumps(result), model,
        ),
    )
    conn.commit()


def cache_clear(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("DELETE FROM recommendation_cache")
    conn.commit()
    return cursor.rowcount


def cache_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM recommendation_cache").fetchone()[0]
    return {"cached_recommendations": total}

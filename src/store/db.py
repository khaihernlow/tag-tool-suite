import hashlib
import json
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor


_SCHEMA = """
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
    imported_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def connect() -> psycopg2.extensions.connection:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
        
    conn = psycopg2.connect(database_url, cursor_factory=DictCursor)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)
    conn.commit()
    return conn


# ── ticket store ───────────────────────────────────────────────────────────────

def import_tickets(df: pd.DataFrame, conn: psycopg2.extensions.connection) -> tuple[int, int]:
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
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tickets
                (ticket_number, title, description, account, resources,
                 status, created, total_hours, billed_hours,
                 sub_issue_type, issue_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticket_number) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    existing_after = ticket_count(conn)
    new_count = existing_after - existing_before
    skipped_count = len(rows) - new_count
    return new_count, skipped_count


def load_tickets(
    conn: psycopg2.extensions.connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load tickets from the store, optionally filtered to a date range (inclusive)."""
    query = "SELECT * FROM tickets"
    params = []
    
    if start_date and end_date:
        query += " WHERE created >= %s AND created <= %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " WHERE created >= %s"
        params.append(start_date)
    elif end_date:
        query += " WHERE created <= %s"
        params.append(end_date)
        
    query += " ORDER BY created"
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["total_hours"] = pd.to_numeric(df["total_hours"], errors="coerce").fillna(0.0) # type: ignore
    df["billed_hours"] = pd.to_numeric(df["billed_hours"], errors="coerce").fillna(0.0) # type: ignore
    return df


def ticket_count(conn: psycopg2.extensions.connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tickets")
        row = cur.fetchone()
        return row[0] if row else 0


def _format_date(date_str: Optional[str]) -> str:
    if not date_str:
        return "-"
    try:
        dt = date.fromisoformat(date_str[:10])
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{months[dt.month - 1]} {dt.day}, {dt.year}"
    except Exception:
        return date_str[:10] if date_str else "-"


def ticket_stats(conn: psycopg2.extensions.connection) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MIN(created), MAX(created) FROM tickets")
        row = cur.fetchone()
    if not row:
        return {"ticket_count": 0, "min_date": "-", "max_date": "-"}
    return {
        "ticket_count": row[0],
        "min_date": _format_date(row[1]) if row[0] > 0 else "-",
        "max_date": _format_date(row[2]) if row[0] > 0 else "-",
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
    conn: psycopg2.extensions.connection,
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
        scope_clause = "AND issue_type = %s"
        base_params: tuple = (account, issue_type)
    else:
        scope_clause = ""
        base_params = (account,)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*), MIN(created) "
            f"FROM tickets WHERE account = %s {scope_clause}",
            base_params,
        )
        all_time = cur.fetchone()
        all_time_count = all_time[0] if all_time else 0
        first_seen = (all_time[1] if all_time and all_time[1] else "")[:10]

        cur.execute(
            f"SELECT COUNT(*) FROM tickets "
            f"WHERE account = %s {scope_clause} AND created >= %s AND created <= %s",
            (*base_params, start_date, end_date),
        )
        recent_row = cur.fetchone()
        recent = recent_row[0] if recent_row else 0

        cur.execute(
            f"SELECT COUNT(*) FROM tickets "
            f"WHERE account = %s {scope_clause} AND created >= %s AND created < %s",
            (*base_params, prior_start, start_date),
        )
        prior_row = cur.fetchone()
        prior = prior_row[0] if prior_row else 0

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
        "all_time_count": all_time_count,
        "first_seen": first_seen,
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
              conn: psycopg2.extensions.connection) -> Optional[dict]:
    key = _cache_key(pattern_type, account, issue_type, ticket_numbers, model)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT result_json FROM recommendation_cache WHERE cache_key = %s", (key,)
        )
        row = cur.fetchone()
    return json.loads(row[0]) if row else None


def cache_set(pattern_type: str, account: str, issue_type: str,
              ticket_numbers: list[str], model: str,
              result: dict, conn: psycopg2.extensions.connection) -> None:
    key = _cache_key(pattern_type, account, issue_type, ticket_numbers, model)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recommendation_cache
                (cache_key, pattern_type, account, issue_type,
                 ticket_numbers, result_json, model)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE SET
                pattern_type = EXCLUDED.pattern_type,
                account = EXCLUDED.account,
                issue_type = EXCLUDED.issue_type,
                ticket_numbers = EXCLUDED.ticket_numbers,
                result_json = EXCLUDED.result_json,
                model = EXCLUDED.model
            """,
            (
                key, pattern_type, account, issue_type,
                json.dumps(sorted(ticket_numbers)),
                json.dumps(result), model,
            ),
        )
    conn.commit()


def cache_clear(conn: psycopg2.extensions.connection) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM recommendation_cache")
        rowcount = cur.rowcount
    conn.commit()
    return rowcount


def cache_stats(conn: psycopg2.extensions.connection) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recommendation_cache")
        row = cur.fetchone()
        total = row[0] if row else 0
    return {"cached_recommendations": total}

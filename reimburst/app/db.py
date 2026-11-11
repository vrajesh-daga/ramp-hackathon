"""SQLite storage. Money is always stored as integer cents to avoid float errors.

To move to Supabase/Postgres later, this file is the only one that talks SQL
(plus the upload folder in config.py).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS clubs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY,
    club_id INTEGER NOT NULL REFERENCES clubs(id),
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('member', 'treasurer'))
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    club_id INTEGER NOT NULL REFERENCES clubs(id),
    name TEXT NOT NULL,
    event_date TEXT NOT NULL,             -- YYYY-MM-DD
    fund TEXT NOT NULL CHECK (fund IN ('GOV', 'MISC')),
    budget_cents INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS policy_rules (
    rule_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    judged_by TEXT NOT NULL CHECK (judged_by IN ('claude', 'code')),
    funds TEXT NOT NULL CHECK (funds IN ('ALL', 'GOV', 'MISC')),
    source TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reuse a number, even after a delete
    member_id INTEGER NOT NULL REFERENCES members(id),
    event_id INTEGER NOT NULL REFERENCES events(id),
    note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    requested_cents INTEGER,
    recommended_cents INTEGER,
    approved_cents INTEGER,
    risk_score INTEGER NOT NULL DEFAULT 0,
    recommendation TEXT NOT NULL DEFAULT '',
    member_message TEXT NOT NULL DEFAULT '',
    treasurer_note TEXT NOT NULL DEFAULT '',
    held_until TEXT,                      -- YYYY-MM-DD, payout waits until the event happened
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    is_current INTEGER NOT NULL DEFAULT 1,
    original_path TEXT NOT NULL,
    processed_path TEXT,
    sha256 TEXT NOT NULL,
    ai_marker TEXT NOT NULL DEFAULT '',
    merchant TEXT NOT NULL DEFAULT '',
    purchase_date TEXT NOT NULL DEFAULT '',
    total_cents INTEGER,
    card_last4 TEXT NOT NULL DEFAULT '',
    extraction_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line_items (
    id INTEGER PRIMARY KEY,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    idx INTEGER NOT NULL,
    description TEXT NOT NULL,
    quantity REAL NOT NULL,
    amount_cents INTEGER NOT NULL,
    taxable INTEGER NOT NULL,
    category TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('allowed', 'disallowed', 'unclear')),
    rule_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    decided_by TEXT NOT NULL              -- 'claude', 'rule' (code backstop) or 'offline'
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    signal TEXT NOT NULL,
    points INTEGER NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    submission_id INTEGER REFERENCES submissions(id),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_calls (
    id INTEGER PRIMARY KEY,
    submission_id INTEGER,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,                 -- 'api', 'cache' or 'offline'
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_notes (       -- the auditor agent's investigations
    id INTEGER PRIMARY KEY,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    status TEXT NOT NULL CHECK (status IN ('running', 'done', 'error')),
    source TEXT NOT NULL,                 -- 'claude' (agent with tools) or 'rules' (no API key)
    summary TEXT NOT NULL DEFAULT '',
    explanation TEXT NOT NULL DEFAULT '',
    suggested_action TEXT NOT NULL DEFAULT '',
    suggested_cents INTEGER,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    steps_json TEXT NOT NULL DEFAULT '[]',  -- every tool call, for the trace on the page
    cost_usd REAL NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_case_notes_sub ON case_notes(submission_id);
CREATE INDEX IF NOT EXISTS idx_receipts_sha ON receipts(sha256);
CREATE INDEX IF NOT EXISTS idx_receipts_sub ON receipts(submission_id);
CREATE INDEX IF NOT EXISTS idx_lines_sub ON line_items(submission_id);
CREATE INDEX IF NOT EXISTS idx_findings_sub ON findings(submission_id);
CREATE INDEX IF NOT EXISTS idx_audit_sub ON audit_log(submission_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """One connection per unit of work; commits on success, rolls back on error."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with tx() as conn:
        conn.executescript(SCHEMA)


def one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    with tx() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def all_rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with tx() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def audit(conn: sqlite3.Connection, submission_id: int | None, actor: str, action: str,
          from_status: str | None = None, to_status: str | None = None, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log (submission_id, actor, action, from_status, to_status, detail, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (submission_id, actor, action, from_status, to_status, detail, now_iso()),
    )


def set_status(conn: sqlite3.Connection, submission_id: int, new_status: str, actor: str,
               action: str, detail: str = "", **fields: Any) -> None:
    """Change a submission's status and write the audit row in the same transaction."""
    row = conn.execute("SELECT status FROM submissions WHERE id = ?", (submission_id,)).fetchone()
    old = row["status"] if row else None
    cols = {"status": new_status, "updated_at": now_iso(), **fields}
    assignments = ", ".join(f"{k} = ?" for k in cols)
    conn.execute(f"UPDATE submissions SET {assignments} WHERE id = ?", (*cols.values(), submission_id))
    audit(conn, submission_id, actor, action, old, new_status, detail)

"""Create (or reset) the demo database: one club, three people, two events, the rules.

Usage:
    python scripts/seed.py            # create tables and seed if empty
    python scripts/seed.py --reset    # wipe submissions, uploads and packets, then seed
    python scripts/seed.py --reset --clear-cache   # also forget cached Claude answers
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import CACHE_DIR, DB_PATH, PACKET_DIR, UPLOAD_DIR  # noqa: E402

RULES = [
    # rule_id, text, judged_by, funds, source
    ("ALC-01", "No alcohol (beer, wine, spirits, hard seltzer) for undergraduate organizations, "
               "from ASUC or miscellaneous funds.", "claude", "ALL",
     "https://data.dailycal.org/2026-01-06-asucspending"),
    ("ABSA-FOOD", "ABSA funds can't pay for food or drinks.", "claude", "ABSA",
     "https://data.dailycal.org/2026-01-06-asucspending"),
    ("ABSA-TRAVEL", "ABSA funds can't pay for travel outside the Bay Area.", "claude", "ABSA",
     "https://data.dailycal.org/2026-01-06-asucspending"),
    ("GIFT-01", "Gift cards, prepaid cards and gifts aren't covered.", "claude", "ALL", "Club policy"),
    ("PERSONAL-01", "Items for one person's own use (clothing, toiletries, personal electronics) "
                    "aren't covered.", "claude", "ALL", "Club policy"),
    ("POST-EVENT", "No refund until the event has happened.", "code", "ALL",
     "https://data.dailycal.org/2026-01-06-asucspending"),
    ("ADVISOR-2500", "Requests over $2,500 need advisor approval.", "code", "ALL",
     "https://studentunion.berkeley.edu/student-org-finance/"),
    ("PR-SIX", "At most six items per CalLink purchase request.", "code", "ALL",
     "https://rocketry.gitbook.io/public/tutorials/submitting-reimbursements"),
]


def wipe(clear_cache: bool) -> None:
    for p in (DB_PATH, DB_PATH.with_suffix(".db-wal"), DB_PATH.with_suffix(".db-shm")):
        p.unlink(missing_ok=True)
    folders = [UPLOAD_DIR, PACKET_DIR] + ([CACHE_DIR] if clear_cache else [])
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)


def seed() -> None:
    db.init_db()
    if db.one("SELECT id FROM clubs LIMIT 1"):
        print("Already seeded. Use --reset to start over.")
        return
    today = date.today()
    with db.tx() as conn:
        conn.execute("INSERT INTO clubs (id, name) VALUES (1, 'Berkeley Builders Club (demo)')")
        conn.executemany(
            "INSERT INTO members (id, club_id, name, email, role) VALUES (?, 1, ?, ?, ?)",
            [(1, "Alex Kim", "alex@berkeley.edu", "member"),
             (2, "Jordan Lee", "jordan@berkeley.edu", "member"),
             (3, "Sam Patel", "sam@berkeley.edu", "treasurer")],
        )
        conn.executemany(
            "INSERT INTO events (id, club_id, name, event_date, fund, budget_cents) VALUES (?, 1, ?, ?, ?, ?)",
            [(1, "Fall Mixer", (today - timedelta(days=2)).isoformat(), "MISC", 30000),
             (2, "Build Night Workshop", (today - timedelta(days=5)).isoformat(), "ABSA", 50000),
             (3, "Spring Showcase", (today + timedelta(days=14)).isoformat(), "MISC", 80000)],
        )
        conn.executemany(
            "INSERT INTO policy_rules (rule_id, text, judged_by, funds, source) VALUES (?, ?, ?, ?, ?)", RULES,
        )
    print(f"Seeded {DB_PATH}")
    print("  Members: Alex Kim, Jordan Lee (members), Sam Patel (treasurer)")
    print("  Events: Fall Mixer (MISC, past), Build Night Workshop (ABSA, past), Spring Showcase (MISC, upcoming)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete the database, uploads and packets first")
    ap.add_argument("--clear-cache", action="store_true", help="with --reset, also delete cached Claude answers")
    args = ap.parse_args()
    if args.reset:
        wipe(args.clear_cache)
    seed()

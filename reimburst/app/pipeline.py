"""The receipt pipeline: prep -> extract -> reconcile -> judge -> fraud signals -> route.

analyze() is pure (no database), so the eval script can run it on labeled
receipts. process_submission() wraps it with the duplicate checks and writes
everything to the database.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

from . import ai, db
from .config import UPLOAD_DIR
from .imaging import ImageError, find_ai_marker, prepare, sha256_bytes
from .policy import (POINTS, Decision, Finding, Line, Reconcile, combine, decide, fmt,
                     reconcile, rules_for_fund, rules_version, to_cents)
from .schemas import ReceiptExtraction

log = logging.getLogger("reimburst.pipeline")

ADVISOR_LIMIT_CENTS = 250_000


@dataclass
class Analysis:
    original_sha: str
    processed_path: Path
    ai_marker: str
    extraction: ReceiptExtraction
    recon: Reconcile
    lines: list[Line]
    extract_source: str
    judge_source: str
    rules_version: str


def analyze(original: bytes, processed_path: Path, event: dict, all_rules: list[dict],
            submission_id: int | None = None) -> Analysis:
    sha = sha256_bytes(original)
    prepare(original, processed_path)
    marker = find_ai_marker(original)

    extraction, extract_source = ai.extract(processed_path.read_bytes(), sha, submission_id)
    recon = reconcile(extraction)

    rules = rules_for_fund(all_rules, event["fund"])
    version = rules_version(rules)
    active_ids = {r["rule_id"] for r in all_rules if r["active"]}
    judgment, judge_source = (None, "skipped")
    if extraction.line_items:
        judgment, judge_source = ai.judge(extraction, event, rules, version, sha, submission_id)
    lines = combine(extraction, judgment, event["fund"], active_ids)
    return Analysis(sha, processed_path, marker, extraction, recon, lines,
                    extract_source, judge_source, version)


def base_findings(a: Analysis) -> list[Finding]:
    """Signals that only need this one receipt."""
    out = []
    if a.ai_marker:
        out.append(Finding("ai_marker", POINTS["ai_marker"],
                           f"The file's metadata contains a generative-AI marker (\"{a.ai_marker}\")."))
    if not a.recon.ok and a.extraction.legible and a.extraction.line_items:
        out.append(Finding("doesnt_add_up", POINTS["doesnt_add_up"], " ".join(a.recon.problems)))
    total = to_cents(a.extraction.total)
    if total > ADVISOR_LIMIT_CENTS:
        out.append(Finding("over_advisor_limit", POINTS["over_advisor_limit"],
                           f"{fmt(total)} is over $2,500, so the club advisor must approve it."))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def duplicate_findings(conn, submission_id: int, a: Analysis) -> list[Finding]:
    """Compare against every other submission's current receipt, from any club.

    - exact_duplicate: the identical file was uploaded before.
    - same_purchase: same merchant, date and total (catches a re-photographed receipt).
    """
    ex = a.extraction
    merchant, total = _norm(ex.merchant_name), to_cents(ex.total)
    others = conn.execute(
        "SELECT submission_id, sha256, merchant, purchase_date, total_cents"
        " FROM receipts WHERE is_current = 1 AND submission_id != ?",
        (submission_id,),
    ).fetchall()
    exact, same = [], []
    for o in others:
        if o["sha256"] == a.original_sha:
            exact.append(o["submission_id"])
        elif (merchant and _norm(o["merchant"]) == merchant and ex.purchase_date
              and o["purchase_date"] == ex.purchase_date and o["total_cents"] == total):
            same.append(o["submission_id"])

    refs = lambda ids: ", ".join(f"#{i}" for i in sorted(set(ids)))  # noqa: E731
    out = []
    if exact:
        out.append(Finding("exact_duplicate", POINTS["exact_duplicate"],
                           f"This exact file was already submitted as {refs(exact)}."))
    if same:
        out.append(Finding("same_purchase", POINTS["same_purchase"],
                           f"Same merchant, date and total as {refs(same)}."))
    return out


def process_submission(submission_id: int) -> None:
    """Run the whole pipeline for one submission. Safe to call from a background task."""
    try:
        _process(submission_id)
    except Exception as exc:  # never leave a submission stuck in "processing"
        if isinstance(exc, (ai.AIError, ImageError)):
            log.warning("Submission #%s: %s", submission_id, exc)
        else:
            log.error("Pipeline failed for #%s\n%s", submission_id, traceback.format_exc())
        message = str(exc) if isinstance(exc, (ai.AIError, ImageError)) else f"Unexpected error: {exc}"
        with db.tx() as conn:
            if isinstance(exc, ImageError):
                db.set_status(conn, submission_id, "needs_info", "system", "unreadable file", message,
                              member_message=f"{message} Please upload a photo (JPEG, PNG or HEIC).",
                              recommendation="Ask the member for a readable photo.", error="")
            else:
                db.set_status(conn, submission_id, "error", "system", "pipeline error", message,
                              error=message,
                              member_message="Something went wrong reading this receipt. Your treasurer has been notified.")


def _process(submission_id: int) -> None:
    sub = db.one("SELECT * FROM submissions WHERE id = ?", (submission_id,))
    if sub is None:
        return
    receipt = db.one("SELECT * FROM receipts WHERE submission_id = ? AND is_current = 1", (submission_id,))
    event = db.one("SELECT * FROM events WHERE id = ?", (sub["event_id"],))
    all_rules = db.all_rows("SELECT * FROM policy_rules")

    with db.tx() as conn:
        db.set_status(conn, submission_id, "processing", "system", "pipeline started", error="")

    original = Path(receipt["original_path"]).read_bytes()
    processed_path = UPLOAD_DIR / f"processed_{receipt['id']}.jpg"
    a = analyze(original, processed_path, event, all_rules, submission_id)
    ex = a.extraction

    with db.tx() as conn:
        findings = base_findings(a) + duplicate_findings(conn, submission_id, a)
        decision: Decision = decide(ex, a.recon, a.lines, findings, event)

        conn.execute(
            "UPDATE receipts SET processed_path = ?, ai_marker = ?, merchant = ?, purchase_date = ?,"
            " total_cents = ?, card_last4 = ?, extraction_json = ? WHERE id = ?",
            (str(processed_path), a.ai_marker, ex.merchant_name, ex.purchase_date,
             to_cents(ex.total), ex.card_last4, ex.model_dump_json(), receipt["id"]),
        )
        conn.execute("DELETE FROM line_items WHERE submission_id = ?", (submission_id,))
        conn.executemany(
            "INSERT INTO line_items (submission_id, idx, description, quantity, amount_cents, taxable,"
            " category, verdict, rule_id, reason, decided_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(submission_id, l.idx, l.description, l.quantity, l.amount_cents, int(l.taxable),
              l.category, l.verdict, l.rule_id, l.reason, l.decided_by) for l in a.lines],
        )
        conn.execute("DELETE FROM findings WHERE submission_id = ?", (submission_id,))
        conn.executemany(
            "INSERT INTO findings (submission_id, signal, points, detail) VALUES (?, ?, ?, ?)",
            [(submission_id, f.signal, f.points, f.detail) for f in findings],
        )
        detail = json.dumps({
            "recommendation": decision.recommendation,
            "risk_score": decision.risk_score,
            "model": ai.MODEL,
            "extract_source": a.extract_source,
            "judge_source": a.judge_source,
            "rules_version": a.rules_version,
        })
        approved = decision.recommended_cents if decision.status == "auto_approved" else None
        db.set_status(conn, submission_id, decision.status, "system", "pipeline decision", detail,
                      requested_cents=decision.requested_cents,
                      recommended_cents=decision.recommended_cents,
                      approved_cents=approved,
                      risk_score=decision.risk_score,
                      recommendation=decision.recommendation,
                      member_message=decision.member_message,
                      held_until=decision.held_until)

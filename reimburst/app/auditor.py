"""The auditor agent: investigates a flagged request and writes a case note for the treasurer.

How it works (with an API key): Claude gets a set of read-only tools (look up the
request, find similar receipts, check the member's history, check the event
budget, read a rule, look at a receipt photo) and decides for itself which to
call and in what order. The Anthropic SDK's Tool Runner runs the loop: Claude
asks for a tool, our Python function runs it, the result goes back, repeat.
Claude finishes by calling write_case_note.

Without an API key (or in offline mode), a rule-based investigator calls the
same tools in a fixed order and writes the note from templates, so the feature
works in every demo.

The agent can only read. It never approves, rejects or moves money: the
treasurer still decides.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from . import ai, db
from .config import ANTHROPIC_API_KEY, EFFORT, MODEL, OFFLINE
from .policy import fmt

log = logging.getLogger("reimburst.auditor")

MAX_TURNS = 10  # hard cap on Claude's tool-calling turns per investigation
ACTIONS = {"approve", "approve_part", "ask_member", "reject"}
ACTION_LABELS = {
    "approve": "Approve in full",
    "approve_part": "Approve part",
    "ask_member": "Ask the member",
    "reject": "Reject",
}

SYSTEM = """You are an auditor helping a student club treasurer review a reimbursement request
that the system flagged. Investigate with the tools, then call write_case_note exactly once.

How to investigate:
- Start with get_submission to see the request, its line items and its flags.
- Follow the flags: for possible duplicates, use find_similar_receipts and, if a match
  turns up, view_receipt on both requests to compare the photos. Use get_member_history
  to see whether this member has a pattern, and get_event_budget if money is tight.
- Use get_rule when a line was struck and you want the rule's exact wording.
- Stop once you can explain the flag. Most cases need 3 to 6 tool calls.

How to write the note:
- Be fair and specific. Most flags are honest mistakes (a teammate claiming a shared
  purchase twice, a blurry photo). Say what the evidence shows; never call anyone a fraud.
- Cite request numbers and dollar amounts.
- Suggest one action: approve, approve_part, ask_member or reject. The treasurer decides."""


# ---------- read-only tools ---------------------------------------------------------
# Each returns a short JSON string for Claude plus a one-line summary for the trace.

def _submission(sid: int) -> dict | None:
    return db.one(
        """SELECT s.*, m.name AS member_name, m.club_id, e.name AS event_name, e.event_date, e.fund,
                  e.budget_cents, r.merchant, r.purchase_date, r.card_last4, r.total_cents, r.ai_marker,
                  r.sha256, r.processed_path, r.original_path
           FROM submissions s JOIN members m ON m.id = s.member_id JOIN events e ON e.id = s.event_id
           LEFT JOIN receipts r ON r.submission_id = s.id AND r.is_current = 1
           WHERE s.id = ?""", (sid,))


def tool_get_submission(submission_id: int) -> tuple[str, str]:
    s = _submission(submission_id)
    if s is None:
        return json.dumps({"error": f"No request {submission_id}"}), f"Request {submission_id} not found"
    lines = db.all_rows("SELECT description, amount_cents, taxable, verdict, rule_id, reason FROM line_items"
                        " WHERE submission_id = ? ORDER BY idx", (submission_id,))
    findings = db.all_rows("SELECT signal, points, detail FROM findings WHERE submission_id = ?", (submission_id,))
    data = {
        "request": submission_id, "status": s["status"], "member": {"id": s["member_id"], "name": s["member_name"]},
        "event": {"id": s["event_id"], "name": s["event_name"], "date": s["event_date"], "fund": s["fund"]},
        "member_note": s["note"], "submitted_at": s["created_at"],
        "receipt": {"merchant": s["merchant"], "purchase_date": s["purchase_date"],
                    "card_last4": s["card_last4"], "total": fmt(s["total_cents"]),
                    "ai_metadata_marker": s["ai_marker"] or None},
        "requested": fmt(s["requested_cents"]), "system_recommends": fmt(s["recommended_cents"]),
        "system_recommendation": s["recommendation"], "risk_score": s["risk_score"],
        "line_items": [{"item": l["description"], "amount": fmt(l["amount_cents"]), "taxable": bool(l["taxable"]),
                        "verdict": l["verdict"], "rule": l["rule_id"], "reason": l["reason"]} for l in lines],
        "flags": findings,
    }
    flags = ", ".join(f["signal"].replace("_", " ") for f in findings) or "no flags"
    return json.dumps(data), f"Read request {submission_id}: {s['merchant'] or 'unknown merchant'}, {fmt(s['requested_cents'])}, {flags}"


def tool_find_similar_receipts(submission_id: int) -> tuple[str, str]:
    s = _submission(submission_id)
    if s is None:
        return json.dumps({"error": f"No request {submission_id}"}), f"Request {submission_id} not found"
    others = db.all_rows(
        """SELECT r.submission_id, r.sha256, r.merchant, r.purchase_date, r.card_last4, r.total_cents,
                  s.status, s.created_at, m.name AS member_name, m.id AS member_id, e.name AS event_name, c.name AS club
           FROM receipts r JOIN submissions s ON s.id = r.submission_id JOIN members m ON m.id = s.member_id
           JOIN events e ON e.id = s.event_id JOIN clubs c ON c.id = m.club_id
           WHERE r.is_current = 1 AND r.submission_id != ?""", (submission_id,))
    norm = lambda x: "".join(ch for ch in (x or "").lower() if ch.isalnum())  # noqa: E731
    matches = []
    for o in others:
        why = []
        if s["sha256"] and o["sha256"] == s["sha256"]:
            why.append("identical file")
        same_merchant = bool(s["merchant"]) and norm(o["merchant"]) == norm(s["merchant"])
        same_date = bool(s["purchase_date"]) and o["purchase_date"] == s["purchase_date"]
        same_total = s["total_cents"] is not None and o["total_cents"] == s["total_cents"]
        same_card = bool(s["card_last4"]) and o["card_last4"] == s["card_last4"]
        if same_merchant and same_date and same_total:
            why.append("same merchant, date and total")
        elif same_merchant and same_total:
            why.append("same merchant and total, different date")
        elif same_card and same_date:
            why.append("same card on the same day")
        if why:
            matches.append({"request": o["submission_id"], "why": why, "member": o["member_name"],
                            "same_member": o["member_id"] == s["member_id"], "club": o["club"],
                            "event": o["event_name"], "merchant": o["merchant"], "purchase_date": o["purchase_date"],
                            "card_last4": o["card_last4"], "total": fmt(o["total_cents"]),
                            "status": o["status"], "submitted_at": o["created_at"]})
    matches = matches[:10]
    if matches:
        summary = "Found " + ", ".join(f"request {m['request']} ({m['why'][0]})" for m in matches)
    else:
        summary = "No similar receipts found"
    return json.dumps({"matches": matches}), summary


def tool_get_member_history(member_id: int) -> tuple[str, str]:
    m = db.one("SELECT name FROM members WHERE id = ?", (member_id,))
    if m is None:
        return json.dumps({"error": f"No member {member_id}"}), f"Member {member_id} not found"
    rows = db.all_rows(
        """SELECT s.id, s.status, s.requested_cents, s.approved_cents, s.created_at, e.name AS event_name,
                  (SELECT GROUP_CONCAT(f.signal, ', ') FROM findings f WHERE f.submission_id = s.id AND f.points > 0) AS flags
           FROM submissions s JOIN events e ON e.id = s.event_id WHERE s.member_id = ? ORDER BY s.id""", (member_id,))
    history = [{"request": r["id"], "event": r["event_name"], "status": r["status"],
                "requested": fmt(r["requested_cents"]), "approved": fmt(r["approved_cents"]),
                "flags": r["flags"] or None, "submitted_at": r["created_at"]} for r in rows]
    flagged = sum(1 for h in history if h["flags"])
    rejected = sum(1 for h in history if h["status"] == "rejected")
    return (json.dumps({"member": m["name"], "requests": history}),
            f"Checked {m['name']}'s history: {len(history)} requests, {flagged} flagged, {rejected} rejected")


def tool_get_event_budget(event_id: int) -> tuple[str, str]:
    e = db.one("SELECT * FROM events WHERE id = ?", (event_id,))
    if e is None:
        return json.dumps({"error": f"No event {event_id}"}), f"Event {event_id} not found"
    t = db.one(
        """SELECT COALESCE(SUM(CASE WHEN status IN ('auto_approved','approved','partially_approved','packeted')
                                    THEN approved_cents ELSE 0 END), 0) AS approved,
                  COALESCE(SUM(CASE WHEN status = 'in_review' THEN recommended_cents ELSE 0 END), 0) AS pending
           FROM submissions WHERE event_id = ?""", (event_id,))
    remaining = e["budget_cents"] - t["approved"]
    data = {"event": e["name"], "date": e["event_date"], "fund": e["fund"], "budget": fmt(e["budget_cents"]),
            "approved_so_far": fmt(t["approved"]), "pending_review": fmt(t["pending"]),
            "remaining_before_pending": fmt(remaining)}
    return json.dumps(data), f"Checked the {e['name']} budget: {fmt(remaining)} of {fmt(e['budget_cents'])} left"


def tool_get_rule(rule_id: str) -> tuple[str, str]:
    r = db.one("SELECT * FROM policy_rules WHERE rule_id = ?", ((rule_id or "").strip().upper(),))
    if r is None:
        return json.dumps({"error": f"No rule {rule_id}"}), f"Rule {rule_id} not found"
    return (json.dumps({"rule": r["rule_id"], "text": r["text"], "applies_to": r["funds"], "source": r["source"]}),
            f"Read rule {r['rule_id']}")


def view_receipt_content(submission_id: int) -> tuple[list[dict] | str, str]:
    s = _submission(submission_id)
    path = (s or {}).get("processed_path") or (s or {}).get("original_path")
    if not path or not Path(path).exists():
        return f"No readable photo for request {submission_id}.", f"No photo for request {submission_id}"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    media = "image/jpeg" if path.endswith((".jpg", ".jpeg")) else "image/png"
    return ([{"type": "text", "text": f"Receipt photo for request {submission_id}:"},
             {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}],
            f"Looked at the receipt photo for request {submission_id}")


# ---------- storage -------------------------------------------------------------------

def _touch(sid: int) -> None:
    """Bump the request's updated_at so open pages notice the change."""
    with db.tx() as conn:
        conn.execute("UPDATE submissions SET updated_at = ? WHERE id = ?", (db.now_iso(), sid))


def _save_steps(case_id: int, steps: list[dict]) -> None:
    with db.tx() as conn:
        conn.execute("UPDATE case_notes SET steps_json = ? WHERE id = ?", (json.dumps(steps), case_id))


def _finish(case_id: int, sid: int, note: dict, steps: list[dict], cost: float, source: str) -> None:
    action = note.get("suggested_action", "ask_member")
    with db.tx() as conn:
        conn.execute(
            """UPDATE case_notes SET status = 'done', summary = ?, explanation = ?, suggested_action = ?,
                      suggested_cents = ?, evidence_json = ?, steps_json = ?, cost_usd = ?, source = ?, finished_at = ?
               WHERE id = ?""",
            (note.get("summary", ""), note.get("likely_explanation", ""), action, note.get("suggested_cents"),
             json.dumps(note.get("evidence", [])), json.dumps(steps), cost, source, db.now_iso(), case_id))
        db.audit(conn, sid, "Auditor agent", "wrote a case note", None, None,
                 f"Suggests: {ACTION_LABELS.get(action, action)}. {len(steps)} checks, ${cost:.4f}.")
        conn.execute("UPDATE submissions SET updated_at = ? WHERE id = ?", (db.now_iso(), sid))


def _fail(case_id: int, sid: int, message: str, steps: list[dict]) -> None:
    with db.tx() as conn:
        conn.execute("UPDATE case_notes SET status = 'error', error = ?, steps_json = ?, finished_at = ? WHERE id = ?",
                     (message, json.dumps(steps), db.now_iso(), case_id))
        conn.execute("UPDATE submissions SET updated_at = ? WHERE id = ?", (db.now_iso(), sid))


def start_case(sid: int) -> int:
    with db.tx() as conn:
        cur = conn.execute("INSERT INTO case_notes (submission_id, status, source, created_at) VALUES (?, 'running', ?, ?)",
                           (sid, "claude" if use_claude() else "rules", db.now_iso()))
        conn.execute("UPDATE submissions SET updated_at = ? WHERE id = ?", (db.now_iso(), sid))
        return cur.lastrowid


def use_claude() -> bool:
    return bool(ANTHROPIC_API_KEY) and not OFFLINE


# ---------- the Claude agent (Tool Runner) ------------------------------------------------

def _normalize_note(raw: dict) -> dict:
    action = str(raw.get("suggested_action", "")).strip().lower().replace(" ", "_")
    if action not in ACTIONS:
        action = "ask_member"
    cents = None
    try:
        dollars = float(raw.get("suggested_amount_dollars") or 0)
        cents = int(round(dollars * 100)) if dollars > 0 else None
    except (TypeError, ValueError):
        pass
    evidence = raw.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return {"summary": str(raw.get("summary", "")).strip(),
            "likely_explanation": str(raw.get("likely_explanation", "")).strip(),
            "suggested_action": action, "suggested_cents": cents,
            "evidence": [str(e) for e in evidence][:8]}


def _run_claude(case_id: int, sid: int) -> None:
    from anthropic import beta_tool

    steps: list[dict] = []
    result: dict = {}

    def record(name: str, args: dict, summary: str) -> None:
        steps.append({"tool": name, "input": args, "summary": summary})
        _save_steps(case_id, steps)

    # Each @beta_tool function's docstring is what Claude reads to decide when to use it.
    @beta_tool
    def get_submission(submission_id: int) -> str:
        """Get a reimbursement request: member, event, receipt facts, line items with verdicts, and fraud flags.

        Args:
            submission_id: The request number.
        """
        out, summary = tool_get_submission(submission_id)
        record("get_submission", {"submission_id": submission_id}, summary)
        return out

    @beta_tool
    def find_similar_receipts(submission_id: int) -> str:
        """Find other requests, from any club, whose receipt matches this one: identical file, same merchant/date/total, or same card on the same day.

        Args:
            submission_id: The request to compare against everything else.
        """
        out, summary = tool_find_similar_receipts(submission_id)
        record("find_similar_receipts", {"submission_id": submission_id}, summary)
        return out

    @beta_tool
    def get_member_history(member_id: int) -> str:
        """List every request a member has made, with status, amounts and any flags.

        Args:
            member_id: The member's id (from get_submission).
        """
        out, summary = tool_get_member_history(member_id)
        record("get_member_history", {"member_id": member_id}, summary)
        return out

    @beta_tool
    def get_event_budget(event_id: int) -> str:
        """Get an event's budget, how much is already approved, and how much is waiting for review.

        Args:
            event_id: The event's id (from get_submission).
        """
        out, summary = tool_get_event_budget(event_id)
        record("get_event_budget", {"event_id": event_id}, summary)
        return out

    @beta_tool
    def get_rule(rule_id: str) -> str:
        """Get the full text and source of a funding rule, such as ALC-01 or ABSA-FOOD.

        Args:
            rule_id: The rule's id.
        """
        out, summary = tool_get_rule(rule_id)
        record("get_rule", {"rule_id": rule_id}, summary)
        return out

    @beta_tool
    def view_receipt(submission_id: int) -> list:
        """Look at the receipt photo for a request. Use it to compare two receipts that might be the same purchase.

        Args:
            submission_id: The request whose photo to view.
        """
        content, summary = view_receipt_content(submission_id)
        record("view_receipt", {"submission_id": submission_id}, summary)
        return [{"type": "text", "text": content}] if isinstance(content, str) else content

    @beta_tool
    def write_case_note(summary: str, likely_explanation: str, suggested_action: str,
                        suggested_amount_dollars: float, evidence: list[str]) -> str:
        """Finish the investigation by writing the case note the treasurer will read. Call this exactly once.

        Args:
            summary: Two or three plain sentences on what you found, citing request numbers and amounts.
            likely_explanation: The most likely innocent or non-innocent explanation, stated fairly.
            suggested_action: One of approve, approve_part, ask_member, reject.
            suggested_amount_dollars: The amount to pay if approving, otherwise 0.
            evidence: Short bullet points of the facts that support the note.
        """
        result.update(_normalize_note({"summary": summary, "likely_explanation": likely_explanation,
                                       "suggested_action": suggested_action,
                                       "suggested_amount_dollars": suggested_amount_dollars,
                                       "evidence": evidence}))
        record("write_case_note", {"suggested_action": result["suggested_action"]}, "Wrote the case note")
        return "Saved. The investigation is complete; reply with one short sentence."

    client = ai._get_client()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        tools=[get_submission, find_similar_receipts, get_member_history, get_event_budget,
               get_rule, view_receipt, write_case_note],
        messages=[{"role": "user", "content": f"Investigate request {sid} and write a case note."}],
        max_iterations=MAX_TURNS,
        output_config={"effort": EFFORT},
    )
    cost = 0.0
    final_text = ""
    started = time.monotonic()
    for message in runner:  # one item per Claude turn; tools run between turns
        latency = int((time.monotonic() - started) * 1000)
        cost += ai._log_call(sid, "audit", "api", message.usage, latency)
        started = time.monotonic()
        text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text").strip()
        if text:
            final_text = text
        if result:  # the note is written; no need for more turns
            break

    if not result:
        if not final_text:
            raise ai.AIError("The auditor stopped without writing a case note.")
        result.update(_normalize_note({"summary": final_text, "suggested_action": "ask_member"}))
    _finish(case_id, sid, result, steps, cost, "claude")


# ---------- the no-API-key investigator ------------------------------------------------------

def _run_rules(case_id: int, sid: int) -> None:
    """Same tools, fixed order, notes written from templates. No model involved."""
    steps: list[dict] = []

    def call(name: str, fn, *args) -> Any:
        out, summary = fn(*args)
        steps.append({"tool": name, "input": list(args), "summary": summary})
        _save_steps(case_id, steps)
        time.sleep(0.4)  # pace the trace so the treasurer can watch it fill in
        return json.loads(out) if isinstance(out, str) and out.startswith("{") else out

    sub = call("get_submission", tool_get_submission, sid)
    if "error" in sub:
        raise ai.AIError(sub["error"])
    similar = call("find_similar_receipts", tool_find_similar_receipts, sid)["matches"]
    history = call("get_member_history", tool_get_member_history, sub["member"]["id"])["requests"]
    budget = call("get_event_budget", tool_get_event_budget, sub["event"]["id"])
    struck_rules = sorted({l["rule"] for l in sub["line_items"] if l["verdict"] == "disallowed"})
    for rule_id in struck_rules:
        call("get_rule", tool_get_rule, rule_id)
    if similar:
        call("view_receipt", view_receipt_content, sid)
        call("view_receipt", view_receipt_content, similar[0]["request"])

    signals = {f["signal"] for f in sub["flags"]}
    name = sub["member"]["name"].split(" ")[0]
    others_flagged = [h for h in history if h["flags"] and h["request"] != sid]
    evidence: list[str] = []
    parts: list[str] = []
    action, amount = "approve", sub["system_recommends"]

    if similar:
        m = similar[0]
        who = "the same member" if m["same_member"] else m["member"]
        evidence.append(f"Request {m['request']}: {m['merchant']}, {m['purchase_date']}, {m['total']}, card ending {m['card_last4'] or 'unknown'}")
        shared = ("Most likely an accidental double submission of the same purchase."
                  if m["same_member"] else
                  f"Most likely a shared purchase claimed by both {name} and {m['member'].split(' ')[0]}; "
                  "only one of them should be paid.")
        if m["request"] < sid:  # the match came first, so this request is the copy
            parts.append(f"Request {sid} matches the earlier request {m['request']} ({m['why'][0]}), submitted by {who} "
                         f"for {m['event']} and currently {m['status'].replace('_', ' ')}.")
            explanation = shared
            action = "reject" if m["status"] not in ("rejected", "needs_info") else "ask_member"
        else:  # this request came first; the later one is the copy
            parts.append(f"Request {sid} came first. Request {m['request']}, submitted later by {who}, "
                         f"is the same purchase ({m['why'][0]}).")
            explanation = shared + f" Pay this one and reject request {m['request']}."
            action = "approve_part" if any(l["verdict"] == "disallowed" for l in sub["line_items"]) else "approve"
    elif "ai_marker" in signals:
        parts.append(f"The photo's file metadata has a generative-AI tag (\"{sub['receipt']['ai_metadata_marker']}\"), "
                     "which image generators and AI editors write into files they create.")
        explanation = (f"It may be a real purchase with an edited image, so ask {name} for the paper receipt "
                       "or a card statement showing the charge before paying.")
        action = "ask_member"
    elif "doesnt_add_up" in signals:
        parts.append(f"The items and tax support {sub['system_recommends']}, but the printed total is {sub['receipt']['total']}.")
        explanation = "Either the photo was misread or the total was changed; the items are what the club can verify."
        action, amount = "approve_part", sub["system_recommends"]
    else:
        parts.append(f"No duplicates or signs of tampering for request {sid}.")
        explanation = "Nothing beyond the struck lines looks unusual."
        action = "approve_part" if any(l["verdict"] == "disallowed" for l in sub["line_items"]) else "approve"

    struck = [l for l in sub["line_items"] if l["verdict"] == "disallowed"]
    if struck:
        parts.append("Not covered: " + ", ".join(f"{l['item']} ({l['amount']}, {l['rule']})" for l in struck) + ".")
    for f in sub["flags"]:
        evidence.append(f"Flag: {f['detail']}")
    evidence.append(f"{name} has {len(history)} request(s), {len(others_flagged)} other flagged.")
    evidence.append(f"{budget['event']} budget: {budget['remaining_before_pending']} of {budget['budget']} left.")
    if others_flagged and action != "reject":
        parts.append(f"{name} has {len(others_flagged)} other flagged request(s), so a quick check-in is worth it.")

    cents = None
    if action in ("approve", "approve_part") and amount not in (None, "—"):
        cents = int(round(float(amount.replace("$", "").replace(",", "")) * 100))
    note = {"summary": " ".join(parts) or f"Request {sid} has no unusual signals.", "likely_explanation": explanation,
            "suggested_action": action, "suggested_cents": cents, "evidence": evidence}
    steps.append({"tool": "write_case_note", "input": {"suggested_action": action}, "summary": "Wrote the case note"})
    _finish(case_id, sid, note, steps, 0.0, "rules")


def investigate(case_id: int, sid: int) -> None:
    """Background task entry point. Never leaves a case stuck in 'running'."""
    try:
        if use_claude():
            _run_claude(case_id, sid)
        else:
            _run_rules(case_id, sid)
    except Exception as exc:
        import anthropic

        if isinstance(exc, anthropic.APIStatusError):
            body = exc.body if isinstance(exc.body, dict) else {}
            detail = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else None
            msg = f"Claude API error {exc.status_code}: {detail or exc.message}"
        elif isinstance(exc, anthropic.APIConnectionError):
            msg = "Couldn't reach the Claude API. Check the internet connection."
        elif isinstance(exc, ai.AIError):
            msg = str(exc)
        else:
            msg = f"Unexpected error: {exc}"
            log.error("Auditor failed for request %s\n%s", sid, traceback.format_exc())
        log.warning("Auditor stopped on request %s: %s", sid, msg)
        steps = (db.one("SELECT steps_json FROM case_notes WHERE id = ?", (case_id,)) or {}).get("steps_json")
        _fail(case_id, sid, msg, json.loads(steps) if steps else [])

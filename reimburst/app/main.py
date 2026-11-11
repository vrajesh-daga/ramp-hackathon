"""Reimburst web app: one FastAPI server for the member pages, the treasurer
queue and the background pipeline.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from . import auditor, db
from .config import MAX_UPLOAD_BYTES, MODEL, OFFLINE, UPLOAD_DIR
from .imaging import sha256_bytes
from .pipeline import process_submission
from .policy import fmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Reimburst", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

STATUS_LABELS = {
    "received": ("Received", "gray"),
    "processing": ("Reading receipt…", "gray"),
    "needs_info": ("Needs info", "amber"),
    "in_review": ("In review", "blue"),
    "auto_approved": ("Auto-approved", "green"),
    "approved": ("Approved", "green"),
    "partially_approved": ("Partly approved", "green"),
    "rejected": ("Rejected", "red"),
    "error": ("Error", "red"),
}
SIGNAL_LABELS = {
    "exact_duplicate": "Exact duplicate",
    "same_purchase": "Same purchase twice",
    "ai_marker": "AI-generation marker",
    "doesnt_add_up": "Doesn't add up",
    "over_advisor_limit": "Over $2,500",
}
WORKING = ("received", "processing")
APPROVED = ("auto_approved", "approved", "partially_approved")


def _when(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%b %d, %I:%M %p").replace(" 0", " ")
    except ValueError:
        return iso


@app.exception_handler(StarletteHTTPException)
async def friendly_errors(request: Request, exc: StarletteHTTPException):
    """Show errors on the page, not as raw JSON.

    A rejected form (a question with no note, an amount that's too high) sends
    you back to the page you came from with the message in a red banner. A bad
    page address shows a simple error page. The /api endpoints still answer JSON.
    """
    message = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
    wants_html = "text/html" in request.headers.get("accept", "")
    if request.url.path.startswith("/api") or not wants_html:
        return JSONResponse({"detail": message}, status_code=exc.status_code)
    back = request.headers.get("referer", "")
    if request.method == "POST" and back:
        parts = urlsplit(back)
        query = [(k, v) for k, v in parse_qsl(parts.query) if k != "error"] + [("error", message)]
        return RedirectResponse(urlunsplit(parts._replace(query=urlencode(query), fragment="")), status_code=303)
    return templates.TemplateResponse(request, "error.html", {"message": message, "code": exc.status_code},
                                      status_code=exc.status_code)


templates.env.filters["money"] = fmt
templates.env.filters["when"] = _when
# Stylesheet version: changes whenever style.css does, so browsers never show a stale copy.
ASSET_V = str(int((APP_DIR / "static" / "style.css").stat().st_mtime))
templates.env.globals.update(STATUS_LABELS=STATUS_LABELS, SIGNAL_LABELS=SIGNAL_LABELS,
                             OFFLINE=OFFLINE, MODEL=MODEL, ASSET_V=ASSET_V)


# ---------- helpers --------------------------------------------------------------------

def current_user(request: Request) -> dict:
    uid = request.cookies.get("uid", "")
    user = db.one("SELECT * FROM members WHERE id = ?", (int(uid),)) if uid.isdigit() else None
    if user is None:
        user = db.one("SELECT * FROM members ORDER BY role = 'treasurer', id LIMIT 1")
    if user is None:
        raise HTTPException(500, "No members found. Run: python scripts/seed.py")
    return user


def render(request: Request, name: str, **ctx) -> object:
    user = current_user(request)
    members = db.all_rows("SELECT * FROM members WHERE club_id = ? ORDER BY role, name", (user["club_id"],))
    return templates.TemplateResponse(request, name, {"user": user, "members": members, **ctx})


def load_submission(sid: int) -> dict:
    sub = db.one(
        """SELECT s.*, m.name AS member_name, m.club_id, e.name AS event_name, e.event_date, e.fund
           FROM submissions s JOIN members m ON m.id = s.member_id JOIN events e ON e.id = s.event_id
           WHERE s.id = ?""",
        (sid,),
    )
    if sub is None:
        raise HTTPException(404, "Submission not found")
    return sub


def parse_dollars(text: str) -> int:
    cleaned = re.sub(r"[^0-9.]", "", text or "")
    if not re.fullmatch(r"\d+(\.\d{1,2})?", cleaned):
        raise HTTPException(400, "Enter the amount as a number, like 18.60.")
    whole, _, frac = cleaned.partition(".")
    return int(whole) * 100 + int((frac + "00")[:2])


async def save_upload(file: UploadFile) -> tuple[Path, bytes]:
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 20 MB.")
    ext = Path(file.filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext or ""):
        ext = ".bin"
    path = UPLOAD_DIR / f"orig_{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    return path, data


def add_receipt(conn, sid: int, path: Path, data: bytes) -> None:
    conn.execute("UPDATE receipts SET is_current = 0 WHERE submission_id = ?", (sid,))
    conn.execute(
        "INSERT INTO receipts (submission_id, original_path, sha256, created_at) VALUES (?, ?, ?, ?)",
        (sid, str(path), sha256_bytes(data), db.now_iso()),
    )


# ---------- pages -----------------------------------------------------------------------

@app.get("/")
def home(request: Request):
    user = current_user(request)
    return RedirectResponse("/treasurer" if user["role"] == "treasurer" else "/member", status_code=303)


@app.get("/switch")
def switch(uid: int):
    if db.one("SELECT id FROM members WHERE id = ?", (uid,)) is None:
        raise HTTPException(404, "No such member")
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("uid", str(uid), max_age=60 * 60 * 24 * 30, samesite="lax")
    return resp


@app.get("/member")
def member_home(request: Request):
    user = current_user(request)
    events = db.all_rows(  # past events first (most recent on top), upcoming after
        "SELECT * FROM events WHERE club_id = ? AND name <> 'Other' AND name NOT LIKE 'Other: %'"
        " ORDER BY event_date > date('now', 'localtime'), event_date DESC", (user["club_id"],))
    subs = db.all_rows(
        """SELECT s.*, e.name AS event_name FROM submissions s JOIN events e ON e.id = s.event_id
           WHERE s.member_id = ? ORDER BY s.id DESC""",
        (user["id"],),
    )
    return render(request, "member.html", events=events, subs=subs)


OTHER_EVENT = "Other"


def other_event(club_id: int, what: str):
    """An 'Other: <what it was for>' event for spending that isn't tied to a listed event (created on first use)."""
    name = f"{OTHER_EVENT}: {what}"
    row = db.one("SELECT * FROM events WHERE club_id = ? AND name = ?", (club_id, name))
    if row is None:
        with db.tx() as conn:
            conn.execute("INSERT INTO events (club_id, name, event_date, fund) VALUES (?, ?, date('now', 'localtime'), 'MISC')",
                         (club_id, name))
        row = db.one("SELECT * FROM events WHERE club_id = ? AND name = ?", (club_id, name))
    return row


@app.post("/submit")
async def submit(request: Request, background: BackgroundTasks, event_id: str = Form(...),
                 note: str = Form(""), other_text: str = Form(""), file: UploadFile = File(...)):
    user = current_user(request)
    if event_id == "other":
        what = " ".join(other_text.split())[:60]
        if not what:
            raise HTTPException(400, "Tell us what it was for.")
        event = other_event(user["club_id"], what)
    else:
        event = db.one("SELECT * FROM events WHERE id = ? AND club_id = ?",
                       (event_id if event_id.isdigit() else -1, user["club_id"]))
    if event is None:
        raise HTTPException(400, "Pick one of your club's events, or choose Other.")
    event_id = event["id"]
    path, data = await save_upload(file)
    with db.tx() as conn:
        now = db.now_iso()
        cur = conn.execute(
            "INSERT INTO submissions (member_id, event_id, note, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 'received', ?, ?)",
            (user["id"], event_id, note.strip()[:500], now, now),
        )
        sid = cur.lastrowid
        add_receipt(conn, sid, path, data)
        db.audit(conn, sid, user["name"], "submitted", None, "received", file.filename or "")
    background.add_task(process_submission, sid)
    return RedirectResponse(f"/s/{sid}", status_code=303)


@app.get("/s/{sid}")
def submission_page(request: Request, sid: int):
    user = current_user(request)
    sub = load_submission(sid)
    if user["role"] != "treasurer" and sub["member_id"] != user["id"]:
        raise HTTPException(403, "This isn't your submission.")
    receipt = db.one("SELECT * FROM receipts WHERE submission_id = ? AND is_current = 1", (sid,))
    lines = db.all_rows("SELECT * FROM line_items WHERE submission_id = ? ORDER BY idx", (sid,))
    findings = db.all_rows("SELECT * FROM findings WHERE submission_id = ? ORDER BY points DESC", (sid,))
    audit = db.all_rows("SELECT * FROM audit_log WHERE submission_id = ? ORDER BY id", (sid,))
    extraction = None
    if receipt and receipt["extraction_json"]:
        extraction = json.loads(receipt["extraction_json"])
    rules = {r["rule_id"]: r["text"] for r in db.all_rows("SELECT rule_id, text FROM policy_rules")}
    case = None
    if user["role"] == "treasurer":
        case = db.one("SELECT * FROM case_notes WHERE submission_id = ? ORDER BY id DESC LIMIT 1", (sid,))
        if case:
            case["steps"] = json.loads(case["steps_json"])
            case["evidence"] = json.loads(case["evidence_json"])
    return render(request, "submission.html", sub=sub, receipt=receipt, lines=lines, findings=findings,
                  audit=audit, extraction=extraction, rules=rules, case=case,
                  working=sub["status"] in WORKING, auditor_uses_claude=auditor.use_claude(),
                  ACTION_LABELS=auditor.ACTION_LABELS)


@app.post("/s/{sid}/investigate")
def investigate(request: Request, sid: int, background: BackgroundTasks):
    """Start the auditor agent on a request. It runs in the background; the page shows its steps live."""
    user = current_user(request)
    if user["role"] != "treasurer":
        raise HTTPException(403, "Only the treasurer can run the auditor.")
    sub = load_submission(sid)
    if sub["status"] in WORKING:
        raise HTTPException(400, "Wait until the receipt has been read.")
    running = db.one("SELECT id FROM case_notes WHERE submission_id = ? AND status = 'running'", (sid,))
    if running is None:
        case_id = auditor.start_case(sid)
        background.add_task(auditor.investigate, case_id, sid)
    return RedirectResponse(f"/s/{sid}#auditor", status_code=303)


@app.get("/api/s/{sid}/case")
def api_case(sid: int):
    c = db.one("SELECT id, status, steps_json FROM case_notes WHERE submission_id = ? ORDER BY id DESC LIMIT 1", (sid,))
    if c is None:
        return JSONResponse({"status": "none", "steps": []})
    return JSONResponse({"id": c["id"], "status": c["status"],
                         "steps": [st["summary"] for st in json.loads(c["steps_json"])]})


@app.get("/receipt/{rid}/image")
def receipt_image(rid: int):
    r = db.one("SELECT processed_path, original_path FROM receipts WHERE id = ?", (rid,))
    if r is None:
        raise HTTPException(404)
    path = r["processed_path"] or r["original_path"]
    if not Path(path).exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/s/{sid}/reply")
async def member_reply(request: Request, sid: int, background: BackgroundTasks,
                       note: str = Form(""), file: UploadFile | None = File(None)):
    user = current_user(request)
    sub = load_submission(sid)
    if sub["member_id"] != user["id"]:
        raise HTTPException(403, "This isn't your submission.")
    if sub["status"] not in ("needs_info", "error"):
        raise HTTPException(400, "This request isn't waiting on you.")
    note = note.strip()[:500]
    has_file = file is not None and (file.filename or "") != ""
    if not has_file and not note:
        raise HTTPException(400, "Add a note or a new photo.")
    path = data = None
    if has_file:
        path, data = await save_upload(file)
    with db.tx() as conn:
        if note:
            conn.execute("UPDATE submissions SET note = TRIM(note || ' ' || ?) WHERE id = ?",
                         (f"[reply] {note}", sid))
        if has_file:
            add_receipt(conn, sid, path, data)
            db.set_status(conn, sid, "received", user["name"], "uploaded a new photo", note)
        else:
            db.set_status(conn, sid, "in_review", user["name"], "replied", note,
                          member_message="Thanks! Your treasurer will take another look.")
    if has_file:
        background.add_task(process_submission, sid)
    return RedirectResponse(f"/s/{sid}", status_code=303)


@app.post("/s/{sid}/decide")
def treasurer_decide(request: Request, sid: int, background: BackgroundTasks, action: str = Form(...),
                     amount: str = Form(""), note: str = Form("")):
    user = current_user(request)
    if user["role"] != "treasurer":
        raise HTTPException(403, "Only the treasurer can decide.")
    sub = load_submission(sid)
    if sub["status"] in WORKING:
        raise HTTPException(400, "This request can't be changed right now.")
    note = note.strip()[:500]
    requested = sub["requested_cents"] or 0

    with db.tx() as conn:
        if action in ("approve", "override"):
            cents = sub["recommended_cents"] if action == "approve" else parse_dollars(amount)
            if cents is None or cents <= 0:
                raise HTTPException(400, "There's no amount to approve. Enter one, or reject.")
            if requested and cents > requested:
                raise HTTPException(400, f"Can't approve more than the receipt total ({fmt(requested)}).")
            status = "approved" if cents == requested else "partially_approved"
            hold = (f" Payout waits until after the event on {sub['held_until']}." if sub["held_until"] else "")
            message = f"Approved by your treasurer: {fmt(cents)}.{hold}" + (f" Note: {note}" if note else "")
            db.set_status(conn, sid, status, user["name"], f"{action}d {fmt(cents)}", note,
                          approved_cents=cents, member_message=message, treasurer_note=note)
        elif action == "ask":
            if not note:
                raise HTTPException(400, "Type your question in the note box first, then click Ask a question.")
            db.set_status(conn, sid, "needs_info", user["name"], "asked for info", note,
                          member_message=f"Your treasurer asked: {note}", treasurer_note=note)
        elif action == "reject":
            message = "Not approved." + (f" {note}" if note else "")
            db.set_status(conn, sid, "rejected", user["name"], "rejected", note,
                          approved_cents=0, member_message=message, treasurer_note=note)
        elif action == "retry":
            db.set_status(conn, sid, "received", user["name"], "retried the pipeline", note)
        else:
            raise HTTPException(400, "Unknown action")
    if action == "retry":
        background.add_task(process_submission, sid)
    return RedirectResponse(f"/s/{sid}", status_code=303)


@app.post("/s/{sid}/delete")
def delete_submission(request: Request, sid: int):
    """Treasurer only: remove a request, its rows, and its photos for good.

    One audit_log row (with no submission attached) records who deleted what,
    so a deletion still leaves a trace. Claude usage stays in ai_calls, unlinked,
    so the cost total still counts money that was really spent.
    """
    user = current_user(request)
    if user["role"] != "treasurer":
        raise HTTPException(403, "Only the treasurer can delete requests.")
    sub = load_submission(sid)
    if sub["club_id"] != user["club_id"]:
        raise HTTPException(403, "This request belongs to another club.")
    if sub["status"] in WORKING:
        raise HTTPException(400, "Wait until the receipt has been read, then delete it.")
    if db.one("SELECT id FROM case_notes WHERE submission_id = ? AND status = 'running'", (sid,)):
        raise HTTPException(400, "A check is still running on this request. Delete it when it's done.")

    receipts = db.all_rows("SELECT original_path, processed_path FROM receipts WHERE submission_id = ?", (sid,))
    with db.tx() as conn:
        for table in ("line_items", "findings", "case_notes", "receipts", "audit_log"):
            conn.execute(f"DELETE FROM {table} WHERE submission_id = ?", (sid,))
        conn.execute("UPDATE ai_calls SET submission_id = NULL WHERE submission_id = ?", (sid,))
        conn.execute("DELETE FROM submissions WHERE id = ?", (sid,))
        amount = fmt(sub["requested_cents"]) if sub["requested_cents"] is not None else "no amount read"
        db.audit(conn, None, user["name"], f"deleted request {sid}", sub["status"], None,
                 f"{sub['member_name']}, {sub['event_name']}, {amount}")

    uploads = UPLOAD_DIR.resolve()
    for r in receipts:  # only ever remove files inside the upload folder
        for p in (r["original_path"], r["processed_path"]):
            if p and Path(p).resolve().parent == uploads:
                Path(p).unlink(missing_ok=True)
    return RedirectResponse(f"/treasurer?notice=Request+{sid}+deleted.", status_code=303)


@app.get("/treasurer")
def treasurer(request: Request):
    user = current_user(request)
    if user["role"] != "treasurer":  # members never see the queue, flags, or other people's amounts
        return RedirectResponse("/member", status_code=303)
    base = """SELECT s.*, m.name AS member_name, e.name AS event_name, r.merchant,
                     (SELECT COUNT(*) FROM findings f WHERE f.submission_id = s.id AND f.points > 0) AS flags
              FROM submissions s JOIN members m ON m.id = s.member_id JOIN events e ON e.id = s.event_id
              LEFT JOIN receipts r ON r.submission_id = s.id AND r.is_current = 1
              WHERE m.club_id = ?"""
    cid = (user["club_id"],)
    sections = [
        ("Needs your review", db.all_rows(base + " AND s.status IN ('in_review', 'error')"
                                          " ORDER BY s.risk_score DESC, s.id", cid)),
        ("Reading now", db.all_rows(base + " AND s.status IN ('received', 'processing') ORDER BY s.id", cid)),
        ("Waiting on the member", db.all_rows(base + " AND s.status = 'needs_info' ORDER BY s.id", cid)),
        ("Approved", db.all_rows(base + " AND s.status IN ('auto_approved', 'approved',"
                                 " 'partially_approved') ORDER BY s.id DESC", cid)),
        ("Rejected", db.all_rows(base + " AND s.status = 'rejected' ORDER BY s.updated_at DESC LIMIT 25", cid)),
    ]
    stats = db.one(
        """SELECT COUNT(*) AS total,
                  COALESCE(SUM(CASE WHEN s.status IN ('auto_approved','approved','partially_approved')
                       THEN s.approved_cents ELSE 0 END), 0) AS approved_cents
           FROM submissions s JOIN members m ON m.id = s.member_id WHERE m.club_id = ?""", cid)
    stats["auto"] = db.one(
        """SELECT COUNT(DISTINCT a.submission_id) AS n FROM audit_log a
           JOIN submissions s ON s.id = a.submission_id JOIN members m ON m.id = s.member_id
           WHERE m.club_id = ? AND a.action = 'pipeline decision' AND a.to_status = 'auto_approved'""", cid)["n"]
    return render(request, "treasurer.html", sections=sections, stats=stats)


# ---------- tiny JSON endpoints for polling ------------------------------------------------

@app.get("/api/s/{sid}")
def api_status(sid: int):
    s = db.one("SELECT status, updated_at FROM submissions WHERE id = ?", (sid,))
    if s is None:
        raise HTTPException(404)
    return JSONResponse(s)


@app.get("/api/queue-version")
def api_queue_version():
    v = db.one("SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), '') AS t FROM submissions")
    return JSONResponse({"v": f"{v['n']}|{v['t']}"})

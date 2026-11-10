"""Reimburst web app: one FastAPI server for the member pages, the treasurer
queue, the packet builder and the background pipeline.

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

from . import db
from .config import MAX_UPLOAD_BYTES, MODEL, OFFLINE, UPLOAD_DIR
from .imaging import sha256_bytes
from .packets import build_packet, ready_for_packet
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
    "packeted": ("In CalLink packet", "purple"),
    "error": ("Error", "red"),
}
SIGNAL_LABELS = {
    "exact_duplicate": "Exact duplicate",
    "same_purchase": "Same purchase twice",
    "near_duplicate": "Edited copy of an earlier receipt",
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


templates.env.filters["money"] = fmt
templates.env.filters["when"] = _when
templates.env.globals.update(STATUS_LABELS=STATUS_LABELS, SIGNAL_LABELS=SIGNAL_LABELS,
                             OFFLINE=OFFLINE, MODEL=MODEL)


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
        raise HTTPException(400, "Enter an amount like 18.60")
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
        "SELECT * FROM events WHERE club_id = ?"
        " ORDER BY event_date > date('now', 'localtime'), event_date DESC", (user["club_id"],))
    subs = db.all_rows(
        """SELECT s.*, e.name AS event_name FROM submissions s JOIN events e ON e.id = s.event_id
           WHERE s.member_id = ? ORDER BY s.id DESC""",
        (user["id"],),
    )
    return render(request, "member.html", events=events, subs=subs)


@app.post("/submit")
async def submit(request: Request, background: BackgroundTasks, event_id: int = Form(...),
                 note: str = Form(""), file: UploadFile = File(...)):
    user = current_user(request)
    event = db.one("SELECT * FROM events WHERE id = ? AND club_id = ?", (event_id, user["club_id"]))
    if event is None:
        raise HTTPException(400, "Pick one of your club's events.")
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
    cost = db.one("SELECT COALESCE(SUM(cost_usd), 0) AS usd FROM ai_calls WHERE submission_id = ?", (sid,))["usd"]
    extraction = None
    if receipt and receipt["extraction_json"]:
        extraction = json.loads(receipt["extraction_json"])
    rules = {r["rule_id"]: r["text"] for r in db.all_rows("SELECT rule_id, text FROM policy_rules")}
    return render(request, "submission.html", sub=sub, receipt=receipt, lines=lines, findings=findings,
                  audit=audit, cost=cost, extraction=extraction, rules=rules,
                  working=sub["status"] in WORKING)


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
    if sub["status"] in WORKING or sub["status"] == "packeted":
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
                raise HTTPException(400, "Write the question for the member.")
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


@app.get("/treasurer")
def treasurer(request: Request):
    user = current_user(request)
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
        ("Approved, not yet filed", db.all_rows(base + " AND s.status IN ('auto_approved', 'approved',"
                                                " 'partially_approved') ORDER BY s.id DESC", cid)),
        ("Done", db.all_rows(base + " AND s.status IN ('packeted', 'rejected') ORDER BY s.updated_at DESC LIMIT 25", cid)),
    ]
    stats = db.one(
        """SELECT COUNT(*) AS total,
                  COALESCE(SUM(CASE WHEN s.status IN ('auto_approved','approved','partially_approved','packeted')
                       THEN s.approved_cents ELSE 0 END), 0) AS approved_cents
           FROM submissions s JOIN members m ON m.id = s.member_id WHERE m.club_id = ?""", cid)
    stats["auto"] = db.one(
        """SELECT COUNT(DISTINCT a.submission_id) AS n FROM audit_log a
           JOIN submissions s ON s.id = a.submission_id JOIN members m ON m.id = s.member_id
           WHERE m.club_id = ? AND a.action = 'pipeline decision' AND a.to_status = 'auto_approved'""", cid)["n"]
    cost = db.one("SELECT COALESCE(SUM(cost_usd), 0) AS usd, COUNT(*) AS calls FROM ai_calls WHERE source = 'api'")
    return render(request, "treasurer.html", sections=sections, stats=stats, cost=cost)


@app.get("/packets")
def packets_page(request: Request):
    user = current_user(request)
    ready = ready_for_packet(user["club_id"])
    held = db.all_rows(
        """SELECT s.id, s.held_until, s.approved_cents, m.name AS member_name FROM submissions s
           JOIN members m ON m.id = s.member_id
           WHERE m.club_id = ? AND s.status IN ('auto_approved','approved','partially_approved')
             AND s.packet_id IS NULL AND s.held_until > date('now', 'localtime') ORDER BY s.held_until""",
        (user["club_id"],),
    )
    packets = db.all_rows("SELECT * FROM packets WHERE club_id = ? ORDER BY id DESC", (user["club_id"],))
    return render(request, "packets.html", ready=ready, held=held, packets=packets)


@app.post("/packets/build")
def packets_build(request: Request):
    user = current_user(request)
    if user["role"] != "treasurer":
        raise HTTPException(403, "Only the treasurer can build packets.")
    build_packet(user["club_id"], user["name"])
    return RedirectResponse("/packets", status_code=303)


@app.get("/packets/{pid}/download")
def packet_download(pid: int):
    p = db.one("SELECT * FROM packets WHERE id = ?", (pid,))
    if p is None or not Path(p["pdf_path"]).exists():
        raise HTTPException(404)
    return FileResponse(p["pdf_path"], media_type="application/pdf", filename=f"callink_packet_{pid}.pdf")


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

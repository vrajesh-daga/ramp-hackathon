"""Build one CalLink-ready PDF from approved requests.

CalLink purchase requests hold at most six items, so approved receipts are
grouped six at a time. Each group gets a cover sheet, then one page per receipt.
"""
from __future__ import annotations

from datetime import date, datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from . import db
from .config import PACKET_DIR
from .policy import fmt

ITEMS_PER_REQUEST = 6
READY_STATUSES = ("auto_approved", "approved", "partially_approved")


def ready_for_packet(club_id: int, today: date | None = None) -> list[dict]:
    today_s = (today or date.today()).isoformat()
    marks = ",".join("?" for _ in READY_STATUSES)
    return db.all_rows(
        f"""SELECT s.*, m.name AS member_name, e.name AS event_name, e.event_date, e.fund,
                   r.merchant, r.purchase_date, r.processed_path, r.original_path
            FROM submissions s
            JOIN members m ON m.id = s.member_id
            JOIN events e ON e.id = s.event_id
            JOIN receipts r ON r.submission_id = s.id AND r.is_current = 1
            WHERE m.club_id = ? AND s.status IN ({marks}) AND s.packet_id IS NULL
              AND (s.held_until IS NULL OR s.held_until <= ?)
            ORDER BY s.id""",
        (club_id, *READY_STATUSES, today_s),
    )


def _cover(c: canvas.Canvas, club: str, group: list[dict], n: int, of: int) -> None:
    w, h = letter
    y = h - 72
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, y, f"Purchase request {n} of {of}")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(72, y, f"{club}  ·  prepared by Reimburst on {datetime.now():%b %d, %Y}")
    y -= 36

    cols = [72, 100, 210, 330, 440, 510]
    c.setFont("Helvetica-Bold", 10)
    for x, label in zip(cols[:-1], ["#", "Member", "Event", "Merchant", "Date"]):
        c.drawString(x, y, label)
    c.drawRightString(w - 72, y, "Approved")
    y -= 6
    c.line(72, y, w - 72, y)
    y -= 16
    c.setFont("Helvetica", 10)
    total = 0
    for i, s in enumerate(group, start=1):
        amount = s["approved_cents"] or 0
        total += amount
        values = [str(i), s["member_name"][:19], s["event_name"][:19], (s["merchant"] or "")[:19],
                  s["purchase_date"] or "", fmt(amount)]
        for x, v in zip(cols[:-1], values[:-1]):
            c.drawString(x, y, v)
        c.drawRightString(w - 72, y, values[-1])  # amounts line up on the right edge
        y -= 18
        if amount != (s["requested_cents"] or 0):
            c.setFont("Helvetica-Oblique", 8.5)
            c.drawString(100, y + 6, f"Receipt total {fmt(s['requested_cents'])}; only covered items are claimed.")
            c.setFont("Helvetica", 10)
            y -= 10
    c.line(72, y + 8, w - 72, y + 8)
    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(cols[4], y, "Total")
    c.drawRightString(w - 72, y, fmt(total))
    y -= 40
    c.setFont("Helvetica", 9)
    c.drawString(72, y, "Each receipt follows on its own page. Request numbers refer to Reimburst submission IDs:")
    y -= 14
    c.drawString(72, y, ", ".join(f"#{s['id']}" for s in group))
    c.showPage()


def _receipt_page(c: canvas.Canvas, s: dict, i: int) -> None:
    w, h = letter
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, h - 60, f"Item {i}: submission #{s['id']}, {s['member_name']}, {s['merchant'] or 'receipt'}")
    c.setFont("Helvetica", 10)
    c.drawString(72, h - 76, f"Approved {fmt(s['approved_cents'])} of {fmt(s['requested_cents'])}"
                             f"  ·  {s['event_name']} ({s['event_date']}, {s['fund']} funds)")
    path = s["processed_path"] or s["original_path"]
    try:
        img = ImageReader(path)
        iw, ih = img.getSize()
        box_w, box_h = w - 144, h - 140
        scale = min(box_w / iw, box_h / ih)
        dw, dh = iw * scale, ih * scale
        c.drawImage(img, (w - dw) / 2, h - 96 - dh, dw, dh)
    except Exception:
        c.drawString(72, h - 110, "(Receipt image could not be loaded.)")
    c.showPage()


def build_packet(club_id: int, actor: str) -> int | None:
    """Create the PDF, mark the requests as packeted, and return the packet id."""
    items = ready_for_packet(club_id)
    if not items:
        return None
    club = db.one("SELECT name FROM clubs WHERE id = ?", (club_id,))["name"]
    groups = [items[i:i + ITEMS_PER_REQUEST] for i in range(0, len(items), ITEMS_PER_REQUEST)]

    with db.tx() as conn:
        cur = conn.execute(
            "INSERT INTO packets (club_id, pdf_path, item_count, request_count, total_cents, created_at)"
            " VALUES (?, '', ?, ?, ?, ?)",
            (club_id, len(items), len(groups), sum(s["approved_cents"] or 0 for s in items), db.now_iso()),
        )
        packet_id = cur.lastrowid
        pdf_path = PACKET_DIR / f"packet_{packet_id}.pdf"

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.setTitle(f"Reimburst packet {packet_id}")
        for n, group in enumerate(groups, start=1):
            _cover(c, club, group, n, len(groups))
            for i, s in enumerate(group, start=1):
                _receipt_page(c, s, i)
        c.save()

        conn.execute("UPDATE packets SET pdf_path = ? WHERE id = ?", (str(pdf_path), packet_id))
        for s in items:
            db.set_status(conn, s["id"], "packeted", actor, "added to CalLink packet",
                          f"packet #{packet_id}", packet_id=packet_id)
    return packet_id

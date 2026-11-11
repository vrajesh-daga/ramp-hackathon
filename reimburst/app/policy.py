"""Deterministic money logic: reconcile the receipt's math, combine Claude's line
verdicts with fixed rules, compute the approvable amount, and route the request.

Nothing in this file calls a model. If Claude misreads a number, reconcile()
catches it; if Claude allows a bottle of wine, the fixed rule backstop strikes it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from .schemas import Judgment, ReceiptExtraction

TOLERANCE_CENTS = 2  # printed amounts can be off by a cent of rounding

# Points per fraud/risk signal. These are starting guesses to tune on labeled
# receipts; what matters is that every flag carries a reason a treasurer can read.
POINTS = {
    "exact_duplicate": 100,   # the very same file was already submitted
    "same_purchase": 60,      # same merchant, date and total as an earlier receipt
    "ai_marker": 60,          # file metadata names a generative AI tool
    "doesnt_add_up": 30,      # line items, tax and total don't reconcile
    "over_advisor_limit": 0,  # not risky, but needs advisor sign-off
}
HOLD_SCORE = 50    # at or above: held with a warning banner
REVIEW_SCORE = 20  # at or above: a treasurer must look


def to_cents(x: float) -> int:
    return int((Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100))


def fmt(cents: int | None) -> str:
    if cents is None:
        return "—"
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


# ---------- rules -------------------------------------------------------------------

def rules_for_fund(all_rules: list[dict], fund: str) -> list[dict]:
    """Active rules Claude judges, limited to those that apply to this fund."""
    return [r for r in all_rules
            if r["active"] and r["judged_by"] == "claude" and r["funds"] in ("ALL", fund)]


def rules_version(rules: list[dict]) -> str:
    """A short fingerprint of the rule text, so a rule edit invalidates cached judgments."""
    blob = "\n".join(f"{r['rule_id']}|{r['funds']}|{r['text']}" for r in sorted(rules, key=lambda r: r["rule_id"]))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def backstop(category: str, fund: str, active_ids: set[str]) -> tuple[str, str, str] | None:
    """Fixed rules that apply no matter what the model says.

    Returns (verdict, rule_id, reason) or None. Categories come from Claude's
    reading, so this is defense in depth, not a replacement for review.
    """
    if category == "alcohol" and "ALC-01" in active_ids:
        return "disallowed", "ALC-01", "Club funds can't pay for alcohol."
    if category == "gift_card" and "GIFT-01" in active_ids:
        return "disallowed", "GIFT-01", "Gift cards aren't covered."
    if category == "personal" and "PERSONAL-01" in active_ids:
        return "disallowed", "PERSONAL-01", "Personal items aren't covered."
    if fund == "ABSA" and category in ("food", "nonalcoholic_drink") and "ABSA-FOOD" in active_ids:
        return "disallowed", "ABSA-FOOD", "ABSA funds can't pay for food or drinks."
    if fund == "ABSA" and category == "travel" and "ABSA-TRAVEL" in active_ids:
        return "unclear", "ABSA-TRAVEL", "ABSA funds can't pay for travel outside the Bay Area; check where this trip went."
    return None


@dataclass
class Line:
    idx: int
    description: str
    quantity: float
    amount_cents: int
    taxable: bool
    category: str
    verdict: str
    rule_id: str
    reason: str
    decided_by: str


def combine(extraction: ReceiptExtraction, judgment: Judgment | None, fund: str,
            active_ids: set[str]) -> list[Line]:
    """Merge Claude's verdicts with the fixed-rule backstop. The stricter answer wins."""
    by_index = {v.index: v for v in judgment.lines} if judgment else {}
    severity = {"allowed": 0, "unclear": 1, "disallowed": 2}
    out: list[Line] = []
    for i, li in enumerate(extraction.line_items):
        if judgment is None:
            verdict, rule_id, reason, by = "allowed", "NONE", "Covered.", "rule"
        elif i in by_index:
            v = by_index[i]
            verdict, rule_id, reason, by = v.verdict, v.rule_id, v.reason, "claude"
            if verdict != "allowed" and rule_id not in active_ids | {"GAP"}:
                rule_id = "GAP"  # never show a rule that doesn't exist
            if verdict == "allowed":
                rule_id = "NONE"
        else:
            verdict, rule_id, reason, by = "unclear", "GAP", "No verdict was returned for this line.", "rule"

        forced = backstop(li.category, fund, active_ids)
        if forced and severity[forced[0]] > severity[verdict]:
            verdict, rule_id, reason = forced
            by = "rule"
        out.append(Line(i, li.description, li.quantity, to_cents(li.amount), li.taxable,
                        li.category, verdict, rule_id, reason, by))
    return out


# ---------- math ------------------------------------------------------------------

@dataclass
class Reconcile:
    ok: bool
    lines_cents: int
    subtotal_cents: int
    computed_total_cents: int
    problems: list[str] = field(default_factory=list)


def reconcile(ex: ReceiptExtraction) -> Reconcile:
    """Check that the receipt's own numbers agree with each other.

    lines - discounts should equal the subtotal, and subtotal + tax + tip should
    equal the total. A misread digit or an edited total breaks one of these.
    """
    lines = sum(to_cents(li.amount) for li in ex.line_items)
    discounts = to_cents(ex.discounts)
    subtotal = to_cents(ex.subtotal)
    tax, tip, total = to_cents(ex.tax), to_cents(ex.tip), to_cents(ex.total)
    problems = []

    if subtotal == 0:  # subtotal not printed; use the items
        subtotal = lines - discounts
    elif abs((lines - discounts) - subtotal) > TOLERANCE_CENTS:
        problems.append(f"Items add up to {fmt(lines - discounts)} but the subtotal says {fmt(subtotal)}.")

    computed_total = subtotal + tax + tip
    if abs(computed_total - total) > TOLERANCE_CENTS:
        problems.append(f"Subtotal + tax + tip is {fmt(computed_total)} but the total says {fmt(total)}.")
    if total <= 0:
        problems.append("No total could be read.")
    return Reconcile(not problems, lines, subtotal, computed_total, problems)


def approvable_cents(ex: ReceiptExtraction, lines: list[Line]) -> int:
    """How much of the receipt the club can pay for.

    Allowed lines are paid in full. Tax is split in proportion to the taxable
    lines that are allowed (striking a taxable bottle of wine removes its share
    of tax, not a non-taxable item's). Receipt-level discounts and tip are split
    in proportion to the allowed share of all items.
    """
    all_items = Decimal(sum(l.amount_cents for l in lines))
    if all_items <= 0:
        return 0
    allowed = Decimal(sum(l.amount_cents for l in lines if l.verdict == "allowed"))
    taxable = Decimal(sum(l.amount_cents for l in lines if l.taxable))
    allowed_taxable = Decimal(sum(l.amount_cents for l in lines if l.taxable and l.verdict == "allowed"))

    share = allowed / all_items
    tax_share = (allowed_taxable / taxable) if taxable > 0 else share

    amount = (allowed
              - Decimal(to_cents(ex.discounts)) * share
              + Decimal(to_cents(ex.tax)) * tax_share
              + Decimal(to_cents(ex.tip)) * share)
    cents = int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(0, min(cents, to_cents(ex.total)))


# ---------- routing ------------------------------------------------------------------

@dataclass
class Finding:
    signal: str
    points: int
    detail: str


@dataclass
class Decision:
    status: str              # auto_approved | in_review | needs_info
    requested_cents: int
    recommended_cents: int
    risk_score: int
    recommendation: str      # one line for the treasurer
    member_message: str      # what the member sees
    held_until: str | None


def decide(ex: ReceiptExtraction, recon: Reconcile, lines: list[Line], findings: list[Finding],
           event: dict, today: date | None = None) -> Decision:
    today = today or date.today()
    requested = to_cents(ex.total)
    recommended = approvable_cents(ex, lines)
    score = sum(f.points for f in findings)
    held_until = event["event_date"] if event["event_date"] > today.isoformat() else None
    hold_note = (f" Payout waits until after the event on {event['event_date']}." if held_until else "")

    # Can't read it: ask the member, not the treasurer.
    if ex.document_type == "not_a_receipt" or not ex.legible or not ex.line_items:
        reason = ("This doesn't look like a receipt." if ex.document_type == "not_a_receipt"
                  else "We couldn't read every line and the total on this photo.")
        return Decision("needs_info", requested, 0, score,
                        "Ask the member for a clearer photo.",
                        f"{reason} Please upload a clear, flat photo of the whole receipt.", held_until)

    disallowed = [l for l in lines if l.verdict == "disallowed"]
    unclear = [l for l in lines if l.verdict == "unclear"]
    signals = {f.signal for f in findings}

    if score >= HOLD_SCORE:
        if "exact_duplicate" in signals or "same_purchase" in signals:
            rec = "Hold: this looks like a receipt that was already submitted. Reject unless explained."
        else:
            rec = f"Hold: high risk ({score} points). Check the flags before paying."
        return Decision("in_review", requested, recommended, score, rec,
                        "Thanks! This one needs a closer look from your treasurer.", held_until)

    struck = ", ".join(f"{l.description} ({l.reason.rstrip('.')})" for l in disallowed)
    if disallowed and recommended == 0:
        rec = "Reject: nothing on this receipt is covered."
        msg = f"Nothing on this receipt is covered: {struck}."
    elif disallowed:
        rec = f"Approve part: {fmt(recommended)} of {fmt(requested)}."
        msg = f"Most of this is covered ({fmt(recommended)}). Not covered: {struck}.{hold_note}"
    else:
        rec = f"Approve {fmt(recommended)}."
        msg = f"Approved: {fmt(recommended)}.{hold_note}"

    if unclear:
        rec += f" {len(unclear)} line(s) need your call."
    if not recon.ok:
        rec += " The receipt's math doesn't add up."
    if "over_advisor_limit" in signals:
        rec += " Needs advisor sign-off (over $2,500)."

    if score >= REVIEW_SCORE or disallowed or unclear or "over_advisor_limit" in signals:
        if disallowed and recommended > 0:
            msg = (f"Your treasurer will confirm, but {fmt(recommended)} looks covered. "
                   f"Not covered: {struck}.{hold_note}")
        elif not disallowed:
            msg = "Thanks! Your treasurer will take a quick look."
        return Decision("in_review", requested, recommended, score, rec, msg, held_until)
    return Decision("auto_approved", requested, recommended, score, rec, msg, held_until)

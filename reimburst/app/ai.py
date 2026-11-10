"""The two Claude calls: read the receipt, then judge each line against the rules.

Both use structured outputs (output_config.format), so the reply is JSON in
exactly our schema. We still re-validate it with Pydantic, and code re-checks
every number afterwards (see policy.py), so the model never gets the last word
on money.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any

from pydantic import ValidationError

from . import db
from .config import (ANTHROPIC_API_KEY, CACHE_DIR, EFFORT, MODEL, OFFLINE, PRICES,
                     SAMPLES_DIR, USE_CACHE)
from .schemas import EXTRACTION_SCHEMA, Judgment, ReceiptExtraction, judgment_schema


class AIError(RuntimeError):
    """Something went wrong talking to Claude; the message is safe to show a treasurer."""


EXTRACT_SYSTEM = """You read photos of purchase receipts for a student club's reimbursement system.
Transcribe exactly what is printed. Never guess a number you can't read; if a key
number (a line amount, the tax or the total) is hidden by blur, folds or cropping,
set legible to false.

Rules:
- One line_items entry per purchased item, in printed order. Do not include
  subtotal, tax, tip, total, payment, change or discount lines as items.
- amount is the line's total in dollars (after any line-level discount), as a
  positive number.
- taxable is true when the receipt marks the line as taxable (often a "T", "TX"
  or "F"/"N" code column); false when it's marked non-taxable or unmarked.
- category is your best guess at what the item is. Use "alcohol" for beer, wine,
  spirits and hard seltzer; "gift_card" for gift or prepaid cards; "personal" for
  items for one person's own use (clothing, toiletries, a personal phone case).
- discounts: receipt-level discounts as one positive number, 0 if none.
- subtotal, tax, tip, total: as printed; 0 if a field isn't printed (total is
  always printed on a real receipt).
- purchase_date as YYYY-MM-DD. If the year is two digits, assume 20xx.
- document_type "not_a_receipt" if the image isn't a receipt, invoice or order
  confirmation at all."""

JUDGE_SYSTEM = """You review line items from a student club's reimbursement request against
the club's funding rules. For each line, decide:
- "allowed" with rule_id "NONE" when no rule forbids it;
- "disallowed" with the rule_id of the rule it breaks;
- "unclear" with rule_id "GAP" when you can't tell what the item is, or no rule
  clearly covers a doubtful item. A treasurer will decide those.

Only cite rules from the list. Judge the item, not the person: reasons should
be one short, neutral sentence a student would find fair (say what the rule
is, never accuse). member_summary: one or two friendly sentences telling the
member what will be covered."""


# ---------- cache and offline answers -------------------------------------------

def _cache_path(kind: str, key: str):
    return CACHE_DIR / f"{kind}_{key}.json"


def _cache_get(kind: str, key: str) -> dict | None:
    if not USE_CACHE:
        return None
    p = _cache_path(kind, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _cache_put(kind: str, key: str, value: dict) -> None:
    if USE_CACHE:
        _cache_path(kind, key).write_text(json.dumps(value, indent=2))


def _offline_truth(sha: str) -> dict | None:
    truth_file = SAMPLES_DIR / "truth.json"
    if not truth_file.exists():
        return None
    return json.loads(truth_file.read_text()).get(sha)


# ---------- logging --------------------------------------------------------------

def _log_call(submission_id: int | None, purpose: str, source: str, usage: Any = None,
              latency_ms: int = 0) -> float:
    inp = out = cache_read = 0
    if usage is not None:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    p_in, p_out, p_cache = PRICES.get(MODEL, (0.0, 0.0, 0.0))
    cost = (inp * p_in + out * p_out + cache_read * p_cache) / 1_000_000
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO ai_calls (submission_id, purpose, model, input_tokens, output_tokens,"
            " cache_read_tokens, cost_usd, latency_ms, source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (submission_id, purpose, MODEL, inp, out, cache_read, cost, latency_ms, source, db.now_iso()),
        )
    return cost


# ---------- the API call ---------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise AIError("ANTHROPIC_API_KEY isn't set. Add it to .env, or run with "
                          "REIMBURST_OFFLINE=1 to use the bundled sample receipts.")
        import anthropic

        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=180.0, max_retries=2)
    return _client


def _call(purpose: str, system: str, content: list[dict], schema: dict,
          submission_id: int | None, max_tokens: int) -> dict:
    import anthropic

    client = _get_client()
    started = time.monotonic()
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,  # thinking tokens count toward this too
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.APIConnectionError as exc:
        raise AIError("Couldn't reach the Claude API. Check the internet connection.") from exc
    except anthropic.APIStatusError as exc:
        raise AIError(f"Claude API error {exc.status_code}: {exc.message}") from exc

    latency_ms = int((time.monotonic() - started) * 1000)
    _log_call(submission_id, purpose, "api", resp.usage, latency_ms)

    if resp.stop_reason == "refusal":
        raise AIError("Claude declined to process this file.")
    if resp.stop_reason == "max_tokens":
        raise AIError("Claude's answer was cut off (max_tokens). Try again or raise the limit.")

    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIError("Claude returned output that wasn't valid JSON.") from exc


# ---------- public functions -----------------------------------------------------

def extract(jpeg_bytes: bytes, original_sha: str, submission_id: int | None = None) -> tuple[ReceiptExtraction, str]:
    """Read the receipt. Returns (extraction, source) where source is api/cache/offline."""
    key = f"{MODEL}_{original_sha}"
    raw = _cache_get("extract", key)
    source = "cache"
    if raw is None and OFFLINE:
        raw = _offline_truth(original_sha)
        source = "offline"
        if raw is None:
            raise AIError("Offline mode only knows the bundled sample receipts. "
                          "Set ANTHROPIC_API_KEY and REIMBURST_OFFLINE=0 to read new receipts.")
    if raw is None:
        content = [
            # Image first, then the instruction: Claude reads images best this way.
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpeg_bytes).decode()}},
            {"type": "text", "text": "Extract this receipt."},
        ]
        raw = _call("extract", EXTRACT_SYSTEM, content, EXTRACTION_SCHEMA, submission_id, max_tokens=16000)
        source = "api"
        _cache_put("extract", key, raw)
    else:
        _log_call(submission_id, "extract", source)

    try:
        return ReceiptExtraction.model_validate(raw), source
    except ValidationError as exc:
        raise AIError(f"Claude's receipt reading didn't match the schema: {exc.errors()[0]['msg']}") from exc


def judge(extraction: ReceiptExtraction, event: dict, rules: list[dict], rules_version: str,
          original_sha: str, submission_id: int | None = None) -> tuple[Judgment, str] | tuple[None, str]:
    """Judge each line against the active rules for this event's fund.

    Returns (None, 'offline') in offline mode; the caller then uses the
    deterministic rule backstop alone.
    """
    key = f"{MODEL}_{original_sha}_{event['fund']}_{rules_version}"
    raw = _cache_get("judge", key)
    if raw is not None:
        _log_call(submission_id, "judge", "cache")
        return Judgment.model_validate(raw), "cache"
    if OFFLINE:
        _log_call(submission_id, "judge", "offline")
        return None, "offline"

    rule_lines = "\n".join(f"- {r['rule_id']}: {r['text']}" for r in rules)
    lines = [
        {"index": i, "description": li.description, "quantity": li.quantity,
         "amount": li.amount, "category_guess": li.category}
        for i, li in enumerate(extraction.line_items)
    ]
    prompt = (
        f"Rules that apply to this event:\n{rule_lines}\n\n"
        f"Event: {event['name']} on {event['event_date']}, paid from {event['fund']} funds.\n"
        f"Merchant: {extraction.merchant_name or 'unknown'}\n\n"
        f"Line items:\n{json.dumps(lines, indent=2)}\n\n"
        "Give a verdict for every line index."
    )
    raw = _call("judge", JUDGE_SYSTEM, [{"type": "text", "text": prompt}],
                judgment_schema([r["rule_id"] for r in rules]), submission_id, max_tokens=8000)
    try:
        judgment = Judgment.model_validate(raw)
    except ValidationError as exc:
        raise AIError(f"Claude's line judgment didn't match the schema: {exc.errors()[0]['msg']}") from exc
    _cache_put("judge", key, raw)
    return judgment, "api"

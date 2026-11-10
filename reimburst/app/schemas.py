"""The data contracts between Claude and the rest of the pipeline.

Each contract exists twice: a JSON Schema sent to Claude (structured outputs
force the reply into this exact shape) and a Pydantic model that re-checks the
reply in Python. Structured outputs reject numeric limits like `minimum` and
limit optional fields, so every field is required and "not printed" is "" or 0.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

CATEGORIES = [
    "food", "nonalcoholic_drink", "alcohol", "gift_card", "supplies",
    "decorations", "travel", "fees", "personal", "other",
]
DOCUMENT_TYPES = [
    "itemized_receipt", "summary_receipt", "invoice",
    "order_confirmation", "card_statement", "not_a_receipt",
]
VERDICTS = ["allowed", "disallowed", "unclear"]


def _lower(v: object) -> object:
    # Structured outputs don't guarantee the capitalization of enum values,
    # so compare and store them lowercased.
    return v.strip().lower() if isinstance(v, str) else v


class LineItem(BaseModel):
    description: str
    quantity: float
    amount: float
    taxable: bool
    category: Literal[
        "food", "nonalcoholic_drink", "alcohol", "gift_card", "supplies",
        "decorations", "travel", "fees", "personal", "other",
    ]

    @field_validator("category", mode="before")
    @classmethod
    def _norm_category(cls, v: object) -> object:
        return _lower(v)


class ReceiptExtraction(BaseModel):
    document_type: Literal[
        "itemized_receipt", "summary_receipt", "invoice",
        "order_confirmation", "card_statement", "not_a_receipt",
    ]
    merchant_name: str
    merchant_city: str
    purchase_date: str
    payer_name: str
    card_last4: str
    line_items: list[LineItem]
    discounts: float
    subtotal: float
    tax: float
    tip: float
    total: float
    legible: bool

    @field_validator("document_type", mode="before")
    @classmethod
    def _norm_doc(cls, v: object) -> object:
        return _lower(v)


class LineVerdict(BaseModel):
    index: int
    verdict: Literal["allowed", "disallowed", "unclear"]
    rule_id: str
    reason: str

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v: object) -> object:
        return _lower(v)

    @field_validator("rule_id", mode="before")
    @classmethod
    def _upper_rule(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v


class Judgment(BaseModel):
    lines: list[LineVerdict]
    member_summary: str


def _obj(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


EXTRACTION_SCHEMA = _obj({
    "document_type": {"type": "string", "enum": DOCUMENT_TYPES},
    "merchant_name": {"type": "string", "description": "Store name as printed"},
    "merchant_city": {"type": "string", "description": "City as printed, or empty string"},
    "purchase_date": {"type": "string", "description": "YYYY-MM-DD, or empty string if unreadable"},
    "payer_name": {"type": "string", "description": "Cardholder or customer name if printed, else empty string"},
    "card_last4": {"type": "string", "description": "Last four card digits if printed, else empty string"},
    "line_items": {
        "type": "array",
        "items": _obj({
            "description": {"type": "string", "description": "Item text exactly as printed"},
            "quantity": {"type": "number", "description": "1 if no quantity is printed"},
            "amount": {"type": "number", "description": "Line total in dollars after any line discount"},
            "taxable": {"type": "boolean", "description": "True if the receipt marks this line taxable"},
            "category": {"type": "string", "enum": CATEGORIES},
        }),
    },
    "discounts": {"type": "number", "description": "Receipt-level discounts as a positive number, 0 if none"},
    "subtotal": {"type": "number", "description": "Printed subtotal, 0 if not printed"},
    "tax": {"type": "number", "description": "Total tax, 0 if none"},
    "tip": {"type": "number", "description": "Tip or gratuity, 0 if none"},
    "total": {"type": "number", "description": "Printed grand total"},
    "legible": {"type": "boolean", "description": "False if blur, folds or cropping hide a key number"},
})


def judgment_schema(rule_ids: list[str]) -> dict:
    """Built per request so rule_id can only be a rule that is actually active."""
    return _obj({
        "lines": {
            "type": "array",
            "items": _obj({
                "index": {"type": "integer", "description": "The line's index as given"},
                "verdict": {"type": "string", "enum": VERDICTS},
                "rule_id": {"type": "string", "enum": sorted(set(rule_ids) | {"NONE", "GAP"})},
                "reason": {"type": "string", "description": "One plain-English sentence"},
            }),
        },
        "member_summary": {"type": "string", "description": "One or two friendly sentences for the member"},
    })

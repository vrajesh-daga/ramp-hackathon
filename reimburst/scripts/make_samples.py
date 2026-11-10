"""Generate the demo receipts (clearly marked as test data) plus their ground truth.

Writes:
  samples/*.jpg|png        the receipt images to upload in the demo
  samples/truth.json       what each file says, keyed by SHA-256 (used by offline mode)
  evals/labels.json        expected outcomes, used by evals/run_evals.py

All merchants are made up. Every receipt is labeled SAMPLE / TEST DATA.

Usage: python scripts/make_samples.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
EVALS = ROOT / "evals"


def font(size: int, bold: bool = False):
    names = (["DejaVuSansMono-Bold.ttf", "Menlo.ttc", "Courier New Bold.ttf"] if bold
             else ["DejaVuSansMono.ttf", "Menlo.ttc", "Courier New.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def money(x: float) -> str:
    return f"{x:,.2f}"


def draw_receipt(r: dict) -> Image.Image:
    """Draw a receipt on paper, then place it on a table-colored background."""
    W, pad = 560, 32
    f, fb, fs = font(22), font(26, bold=True), font(18)
    rows: list[tuple[str, str, object]] = [("SAMPLE RECEIPT - TEST DATA", "", fs), ("", "", f),
                                           (r["merchant_name"].upper(), "", fb),
                                           (r["address"], "", fs), (r["merchant_city"].upper() + ", CA", "", fs),
                                           ("", "", f), (f"DATE {r['printed_date']}   {r['time']}", "", f),
                                           ("-" * 34, "", f)]
    for li in r["line_items"]:
        flag = " T" if li["taxable"] else " N"
        rows.append((li["printed"], money(li["amount"]) + flag, f))
    rows += [("-" * 34, "", f), ("SUBTOTAL", money(r["subtotal"]), f), ("TAX", money(r["tax"]), f)]
    if r["tip"]:
        rows.append(("TIP", money(r["tip"]), f))
    rows += [("TOTAL", money(r["printed_total"]), fb), ("", "", f)]
    if r["card_last4"]:
        rows.append((f"VISA ************{r['card_last4']}", "", f))
    if r["payer_name"]:
        rows.append((f"CARDHOLDER {r['payer_name'].upper()}", "", f))
    rows += [("", "", f), ("THANK YOU!", "", f)]

    line_h = 34
    H = pad * 2 + line_h * len(rows)
    paper = Image.new("RGB", (W, H), (252, 251, 246))
    d = ImageDraw.Draw(paper)
    y = pad
    for left, right, fnt in rows:
        header_lines = ("SAMPLE RECEIPT - TEST DATA", "THANK YOU!", r["address"], r["merchant_city"].upper() + ", CA")
        centered = left in header_lines or (fnt is fb and not right)
        if centered and not right:
            tw = d.textlength(left, font=fnt)
            d.text(((W - tw) / 2, y), left, fill=(25, 25, 25), font=fnt)
        else:
            d.text((pad, y), left, fill=(25, 25, 25), font=fnt)
        if right:
            tw = d.textlength(right, font=fnt)
            d.text((W - pad - tw, y), right, fill=(25, 25, 25), font=fnt)
        y += line_h

    bg = Image.new("RGB", (W + 240, H + 200), (122, 118, 110))
    shadow = Image.new("RGB", (W, H), (90, 87, 80))
    bg.paste(shadow, (128, 108))
    bg.paste(paper, (120, 100))
    return bg


def truth_of(r: dict) -> dict:
    """The extraction we expect Claude to return for this receipt."""
    return {
        "document_type": "itemized_receipt",
        "merchant_name": r["merchant_name"],
        "merchant_city": r["merchant_city"],
        "purchase_date": r["purchase_date"],
        "payer_name": r["payer_name"],
        "card_last4": r["card_last4"],
        "line_items": [{"description": li["printed"], "quantity": li["qty"], "amount": li["amount"],
                        "taxable": li["taxable"], "category": li["category"]} for li in r["line_items"]],
        "discounts": 0.0,
        "subtotal": r["subtotal"],
        "tax": r["tax"],
        "tip": r["tip"],
        "total": r["printed_total"],
        "legible": True,
    }


def li(printed, amount, taxable, category, qty=1.0):
    return {"printed": printed, "amount": amount, "taxable": taxable, "category": category, "qty": qty}


RECEIPTS = {
    "1_clean_supplies": {
        "merchant_name": "Cal Craft Supply", "address": "2410 Example Ave", "merchant_city": "Berkeley",
        "printed_date": "10/28/26", "purchase_date": "2026-10-28", "time": "14:12",
        "line_items": [li("POSTER BOARD 22X28 4@2.99", 11.96, True, "supplies", 4.0),
                       li("MARKERS 12PK", 8.49, True, "supplies"),
                       li("PACKING TAPE", 4.99, True, "supplies")],
        "subtotal": 25.44, "tax": 2.61, "tip": 0.0, "printed_total": 28.05,
        "card_last4": "4821", "payer_name": "Alex Kim",
    },
    "2_grocery_with_wine": {
        "merchant_name": "Bear Market Grocery", "address": "1800 Sample St", "merchant_city": "Berkeley",
        "printed_date": "10/30/26", "purchase_date": "2026-10-30", "time": "18:40",
        "line_items": [li("TORTILLA CHIPS 3@4.00", 12.00, False, "food", 3.0),
                       li("PAPER PLATES 50CT", 6.00, True, "supplies"),
                       li("CAB SAUV 750ML", 18.00, True, "alcohol")],
        "subtotal": 36.00, "tax": 2.40, "tip": 0.0, "printed_total": 38.40,
        "card_last4": "7310", "payer_name": "Jordan Lee",
    },
    "4_edited_total_TEST": {
        "merchant_name": "Southside Snacks", "address": "2500 Demo Way", "merchant_city": "Berkeley",
        "printed_date": "10/31/26", "purchase_date": "2026-10-31", "time": "12:03",
        "line_items": [li("WATER 24PK", 5.99, False, "nonalcoholic_drink"),
                       li("PARTY CUPS 50CT", 4.50, True, "supplies"),
                       li("NAPKINS 200CT", 3.25, True, "supplies")],
        # The items, tax and total don't agree: someone "edited" the total.
        "subtotal": 13.74, "tax": 0.80, "tip": 0.0, "printed_total": 83.74,
        "card_last4": "7310", "payer_name": "Jordan Lee",
    },
    "5_ai_marker": {
        "merchant_name": "Telegraph Print Shop", "address": "2600 Mock Blvd", "merchant_city": "Berkeley",
        "printed_date": "11/02/26", "purchase_date": "2026-11-02", "time": "10:15",
        "line_items": [li("FLYERS COLOR 100CT", 40.00, True, "supplies")],
        "subtotal": 40.00, "tax": 4.10, "tip": 0.0, "printed_total": 44.10,
        "card_last4": "4821", "payer_name": "Alex Kim",
    },
}

XMP_AI = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
    'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
    '</rdf:RDF></x:xmpmeta>'
)


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    EVALS.mkdir(exist_ok=True)
    truth: dict[str, dict] = {}
    files: dict[str, Path] = {}

    for name, r in RECEIPTS.items():
        img = draw_receipt(r)
        if name == "5_ai_marker":
            info = PngInfo()
            info.add_itxt("XML:com.adobe.xmp", XMP_AI)  # uncompressed, as editors write it
            path = SAMPLES / f"{name}.png"
            img.save(path, "PNG", pnginfo=info)
        else:
            path = SAMPLES / f"{name}.jpg"
            img.save(path, "JPEG", quality=92)
        files[name] = path
        truth[hashlib.sha256(path.read_bytes()).hexdigest()] = truth_of(r)

    # 3: the clean receipt "photographed again": slightly rotated, cropped, darker
    # and re-compressed, so its bytes differ but the purchase is the same.
    base = Image.open(files["1_clean_supplies"])
    w, h = base.size
    re_photo = base.rotate(1.5, resample=Image.BICUBIC, expand=False, fillcolor=(122, 118, 110))
    re_photo = re_photo.crop((14, 10, w - 6, h - 12))
    re_photo = ImageEnhance.Brightness(re_photo).enhance(0.92)
    buf = io.BytesIO()
    re_photo.save(buf, "JPEG", quality=80)
    path = SAMPLES / "3_duplicate_rephoto.jpg"
    path.write_bytes(buf.getvalue())
    files["3_duplicate_rephoto"] = path
    truth[hashlib.sha256(buf.getvalue()).hexdigest()] = truth_of(RECEIPTS["1_clean_supplies"])

    (SAMPLES / "truth.json").write_text(json.dumps(truth, indent=2))

    labels = [
        {"file": "samples/1_clean_supplies.jpg", "fund": "MISC", "merchant": "Cal Craft Supply",
         "purchase_date": "2026-10-28", "total": 28.05, "expected_status": "auto_approved",
         "expected_approved": 28.05, "disallowed_keywords": []},
        {"file": "samples/2_grocery_with_wine.jpg", "fund": "MISC", "merchant": "Bear Market Grocery",
         "purchase_date": "2026-10-30", "total": 38.40, "expected_status": "in_review",
         "expected_approved": 18.60, "disallowed_keywords": ["CAB SAUV"]},
        {"file": "samples/2_grocery_with_wine.jpg", "fund": "ABSA", "merchant": "Bear Market Grocery",
         "purchase_date": "2026-10-30", "total": 38.40, "expected_status": "in_review",
         "expected_approved": 6.60, "disallowed_keywords": ["CAB SAUV", "TORTILLA"]},
        {"file": "samples/4_edited_total_TEST.jpg", "fund": "MISC", "merchant": "Southside Snacks",
         "purchase_date": "2026-10-31", "total": 83.74, "expected_status": "in_review",
         "expected_approved": 14.54, "disallowed_keywords": []},
        {"file": "samples/5_ai_marker.png", "fund": "MISC", "merchant": "Telegraph Print Shop",
         "purchase_date": "2026-11-02", "total": 44.10, "expected_status": "in_review",
         "expected_approved": 44.10, "disallowed_keywords": []},
    ]
    (EVALS / "labels.json").write_text(json.dumps(labels, indent=2))
    for p in sorted(files.values()):
        print("wrote", p.relative_to(ROOT))
    print("wrote samples/truth.json and evals/labels.json")


if __name__ == "__main__":
    sys.exit(main())

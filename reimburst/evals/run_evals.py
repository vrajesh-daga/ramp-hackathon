"""Measure how well Reimburst reads and judges receipts against hand labels.

Usage (from the project folder):
    python evals/run_evals.py                          # uses ANTHROPIC_API_KEY
    python evals/run_evals.py --offline                # bundled samples only, no API calls
    python evals/run_evals.py --model claude-haiku-5-5 --no-cache   # compare a cheaper model

Labels live in evals/labels.json. Add your own receipts to evals/receipts/ and a
label for each one (copy an existing entry and edit it).

Outputs (all in evals/results/):
    eval_results.csv   one row per receipt
    eval_summary.json  the headline numbers
    eval_metrics.png   bar chart for the slide
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "evals" / "labels.json"))
    ap.add_argument("--offline", action="store_true", help="no API calls; bundled samples only")
    ap.add_argument("--model", help="e.g. claude-haiku-5-5 (default: REIMBURST_MODEL or claude-sonnet-5-5)")
    ap.add_argument("--no-cache", action="store_true", help="always call the API, ignore cached answers")
    args = ap.parse_args()

    # Settings are read when the app package is imported, so set them first.
    RESULTS.mkdir(parents=True, exist_ok=True)
    eval_db = RESULTS / "eval.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(eval_db) + suffix).unlink(missing_ok=True)
    os.environ["REIMBURST_DB"] = str(eval_db)  # keep eval runs out of the demo database
    if args.offline:
        os.environ["REIMBURST_OFFLINE"] = "1"
    if args.model:
        os.environ["REIMBURST_MODEL"] = args.model
    if args.no_cache:
        os.environ["REIMBURST_CACHE"] = "0"
    sys.path.insert(0, str(ROOT))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from app import db
    from app.ai import AIError
    from app.config import MODEL
    from app.pipeline import analyze, base_findings
    from app.policy import decide, to_cents
    from scripts.seed import RULES

    db.init_db()
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO policy_rules (rule_id, text, judged_by, funds, source) VALUES (?, ?, ?, ?, ?)", RULES)
    all_rules = db.all_rows("SELECT * FROM policy_rules")
    labels = json.loads(Path(args.labels).read_text())
    past = (date.today() - timedelta(days=3)).isoformat()

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    rows = []
    for i, lab in enumerate(labels):
        path = ROOT / lab["file"]
        event = {"name": "Eval event", "event_date": past, "fund": lab.get("fund", "MISC")}
        row = {"file": lab["file"], "fund": event["fund"]}
        try:
            a = analyze(path.read_bytes(), RESULTS / "processed" / f"{i}.jpg", event, all_rules)
        except AIError as exc:
            print(f"  skipped {lab['file']}: {exc}")
            row["error"] = str(exc)
            rows.append(row)
            continue
        ex = a.extraction
        d = decide(ex, a.recon, a.lines, base_findings(a), event)

        # Reading accuracy: did we get the three facts a treasurer checks first?
        # A total counts as right if it's within 2 cents of the label.
        row["total_ok"] = abs(to_cents(ex.total) - to_cents(lab["total"])) <= 2
        row["date_ok"] = ex.purchase_date == lab["purchase_date"]
        row["merchant_ok"] = bool(norm(lab["merchant"])) and (
            norm(lab["merchant"]) in norm(ex.merchant_name) or norm(ex.merchant_name) in norm(lab["merchant"]))

        # Catching disallowed lines. Recall = of the lines that SHOULD be struck,
        # what share did we strike? (Missing one means paying for something the
        # club can't.) False strikes = lines we struck that the label says are fine.
        struck = [l for l in a.lines if l.verdict == "disallowed"]
        keywords = [k.lower() for k in lab.get("disallowed_keywords", [])]
        hits = sum(any(k in l.description.lower() for l in struck) for k in keywords)
        row["should_strike"] = len(keywords)
        row["struck_correctly"] = hits
        row["false_strikes"] = sum(not any(k in l.description.lower() for k in keywords) for l in struck)

        # Agreement: does Reimburst route the request and size the payout the
        # way a careful treasurer (the label) would?
        row["status"] = d.status
        row["status_ok"] = d.status == lab["expected_status"]
        row["amount"] = d.recommended_cents / 100
        row["amount_ok"] = abs(d.recommended_cents - to_cents(lab["expected_approved"])) <= 2
        row["sources"] = f"{a.extract_source}/{a.judge_source}"
        rows.append(row)
        print(f"  {lab['file']} ({event['fund']}): {d.status}, ${d.recommended_cents / 100:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "eval_results.csv", index=False)
    scored = df[df.get("error").isna()] if "error" in df else df
    n = len(scored)
    if n == 0:
        print("No receipts were scored.")
        return

    calls = db.all_rows("SELECT source, cost_usd FROM ai_calls")
    api_cost = sum(c["cost_usd"] for c in calls if c["source"] == "api")
    should = int(scored["should_strike"].sum())
    summary = {
        "model": MODEL,
        "receipts_scored": n,
        "total_accuracy": float(scored["total_ok"].mean()),
        "date_accuracy": float(scored["date_ok"].mean()),
        "merchant_accuracy": float(scored["merchant_ok"].mean()),
        # Recall is undefined with nothing to catch, so report None rather than 100%.
        "disallowed_recall": (float(scored["struck_correctly"].sum()) / should) if should else None,
        "false_strikes": int(scored["false_strikes"].sum()),
        "route_agreement": float(scored["status_ok"].mean()),
        "amount_agreement": float(scored["amount_ok"].mean()),
        "api_cost_usd": round(api_cost, 4),
        "api_cost_per_receipt_usd": round(api_cost / n, 4),
        "answers_from": sorted(scored["sources"].unique().tolist()),
    }
    (RESULTS / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    metrics = {
        "Total read right": summary["total_accuracy"],
        "Date read right": summary["date_accuracy"],
        "Merchant read right": summary["merchant_accuracy"],
        "Disallowed lines caught": summary["disallowed_recall"] or 0.0,
        "Route matches label": summary["route_agreement"],
        "Payout matches label": summary["amount_agreement"],
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(list(metrics)[::-1], [v * 100 for v in list(metrics.values())[::-1]], color="#1f5f4a")
    ax.bar_label(bars, labels=[f"{v * 100:.0f}%" for v in list(metrics.values())[::-1]], padding=4)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Percent (of receipts; of should-strike lines for 'caught')")
    offline_only = summary["answers_from"] == ["offline/offline"]
    who = "saved offline answers, not a real model run" if offline_only else (
        f"{MODEL}, ${summary['api_cost_per_receipt_usd']:.3f} per receipt")
    ax.set_title(f"[EVAL] Reimburst on {n} labeled receipts ({who})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "eval_metrics.png", dpi=160)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()

# Reimburst

A club member photographs a receipt. Claude reads every line, code checks the math and the club's rules, and the treasurer only has to decide the few requests that get flagged. Approved requests become a ready-to-file CalLink packet.

This is the hackathon prototype: one FastAPI app running on your laptop, with SQLite for storage. There's no cloud backend and nothing to deploy.

## Quick start (Mac)

You need Python 3.11 or newer. Check with `python3 --version`. If yours is older, run `brew install python@3.12`.

```bash
cd reimburst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then paste your key after ANTHROPIC_API_KEY=
python scripts/seed.py          # creates data/reimburst.db with a demo club
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Use the dropdown in the top-right corner to switch between the two members and the treasurer.

**No API key yet?** Run `REIMBURST_OFFLINE=1 uvicorn app.main:app --host 0.0.0.0 --port 8000`. Offline mode makes no API calls and uses saved answers for the receipts in `samples/`, so the whole app works for testing and rehearsing. Any other image shows an error in offline mode.

**From a phone:** keep the phone on the same Wi-Fi as the laptop, run `ipconfig getifaddr en0` on the laptop, and open `http://THAT-IP:8000` on the phone. Campus Wi-Fi often blocks phone-to-laptop traffic. If it does, connect both devices to a phone hotspot.

**Start over:** `python scripts/seed.py --reset` wipes requests, uploads and packets. Add `--clear-cache` to also forget Claude's saved answers.

## Demo script (about 3 minutes)

Run `python scripts/seed.py --reset` first, then upload these files from `samples/` in this order:

| # | Who | File | What happens |
| --- | --- | --- | --- |
| 1 | Alex Kim | `1_clean_supplies.jpg` | Auto-approved, $28.05. No human needed. |
| 2 | Jordan Lee | `2_grocery_with_wine.jpg` | "Approve part": $18.60 of $38.40. The wine is struck under ALC-01, and only the wine's share of tax is removed. |
| 3 | Jordan Lee | `3_duplicate_rephoto.jpg` | Held: same merchant, date and total as #1, even though it's a different photo. |
| 4 | Jordan Lee | `4_edited_total_TEST.jpg` | "Doesn't add up": the items support $14.54, but the printed total says $83.74. |
| 5 | Alex Kim | `5_ai_marker.png` | Held: the file's metadata says it was made by a generative AI tool. |

Then switch to **Sam Patel (treasurer)**. The queue is sorted by risk, and each flag explains itself. Approve #2, reject #3, then open **CalLink packets** and build the PDF. Every action shows up in the request's audit trail.

Try the **Build Night Workshop** event too: it's paid from ABSA funds, so food gets struck under ABSA-FOOD. **Spring Showcase** hasn't happened yet, so its approvals wait for the event date before they can go into a packet.

To demo with a real phone photo, print a sample receipt (or bring a real one) and run with your API key. Offline mode only knows the exact sample files.

## How it works

```
upload ─► prep ─► Claude reads ─► code checks math ─► Claude + fixed rules judge lines ─► fraud signals ─► route
          HEIC→JPEG,   structured     lines = subtotal,      per-line verdict with           duplicates,        auto-approve /
          rotate,      JSON output    + tax + tip = total    a rule ID; code backstop        AI marker,         ask member /
          resize                                             strikes alcohol, gift cards     doesn't add up     treasurer
```

- **Two Claude calls per receipt** (`app/ai.py`): one reads the image into a strict JSON schema, the other judges each line against the active rules. Both use structured outputs, so the reply always matches the schema. Answers are cached by file hash, so a re-upload is instant and free.
- **Code gets the last word on money** (`app/policy.py`): the receipt's math is reconciled in integer cents, and the approvable amount is computed in code. A fixed rule backstop strikes alcohol, gift cards and personal items (and food on ABSA funds) even if the model allowed them.
- **Fraud signals** (`app/pipeline.py`): exact duplicate (file hash), same purchase (merchant + date + total, which catches re-photos), AI-tool metadata, and math that doesn't add up (which also catches an edited total). Each signal is worth points. 50 or more holds the request; any flag or struck line sends it to the treasurer.
- **Members never see fraud flags.** They see a neutral "your treasurer will take a look" message. Only the treasurer can reject.

## Evals (backend experimentation, not a product feature)

The app never runs or shows evals. This is a separate script for measuring how well the pipeline reads and judges receipts while you tune it.

```bash
python evals/run_evals.py                               # real Claude calls (uses your key)
python evals/run_evals.py --model claude-haiku-5-5 --no-cache   # cheaper-model comparison
python evals/run_evals.py --offline                     # tests the harness only, no API calls
```

Results go to `evals/results/`: a CSV, a JSON summary and a `[EVAL]` chart. The bundled labels only cover the sample receipts. Add about 30 real receipts to `evals/receipts/`, with a label for each one in `evals/labels.json` (copy an existing entry), so the numbers mean something. The offline run always scores 100% because it replays saved answers.

## Project layout

```
app/
  main.py        routes and pages (FastAPI + Jinja2)
  pipeline.py    runs one receipt end to end; duplicate checks
  ai.py          the two Claude calls, cache, offline mode, cost logging
  policy.py      reconcile, rule backstop, partial-approval math, routing
  imaging.py     HEIC/rotation/resize, file hash, AI-marker scan
  packets.py     CalLink packet PDF (ReportLab)
  schemas.py     extraction and judgment schemas
  db.py          SQLite schema and helpers (money stored as integer cents)
  templates/     HTML pages      static/style.css   (no CDN, works offline)
scripts/seed.py          demo club, members, events, rules
scripts/make_samples.py  regenerates samples/ and evals/labels.json
evals/run_evals.py       offline experiments: accuracy, recall, agreement and cost
samples/                 demo receipts (fictional merchants, marked TEST DATA)
```

## Settings (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | (empty) | Your key from platform.claude.com |
| `REIMBURST_MODEL` | `claude-sonnet-5-5` | Model for both calls |
| `REIMBURST_EFFORT` | `medium` | `low` is faster and cheaper; `high` is more careful |
| `REIMBURST_OFFLINE` | `0` | `1` = no API calls, sample receipts only |
| `REIMBURST_CACHE` | `1` | Reuse answers for files seen before |

## Changes from the plan doc

- **SQLite instead of Supabase.** Everything runs on one laptop, so a local file database removes setup and keeps working when the network doesn't. All SQL lives in `app/db.py` if you want to swap it out later.
- **Two duplicate checks only: exact file hash, and same merchant + date + total.** A pixel fingerprint was tried and dropped, because low-resolution fingerprints rated two *different* receipts as more alike than two photos of the *same* receipt. An item-overlap check was dropped too, to keep the demo simple; an edited total is still caught because the math doesn't add up.
- **Plain CSS instead of Tailwind,** so pages look right even without internet.
- **Deferred:** real logins, payer-name matching, tax-rate and split-purchase checks, the auditor agent, and the Ramp sandbox.

## Troubleshooting

- **"ANTHROPIC_API_KEY isn't set":** add it to `.env`, restart uvicorn, then press "Retry" on the request (as the treasurer).
- **"Claude API error 400 …":** usually a typo in the model name. Check `REIMBURST_MODEL`.
- **The phone can't load the page:** see "From a phone" above, and check that macOS's firewall allows incoming connections for Python.
- **HEIC photos fail to open:** run `pip install pillow-heif`.

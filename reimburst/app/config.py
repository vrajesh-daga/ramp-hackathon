"""Settings and paths for Reimburst. Everything lives under the project folder."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that holds app/, scripts/, data/ ...
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# All runtime output goes under data/ so it's easy to wipe and never committed.
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("REIMBURST_DB", str(DATA_DIR / "reimburst.db")))
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
PACKET_DIR = DATA_DIR / "packets"
SAMPLES_DIR = ROOT / "samples"

for _d in (DATA_DIR, UPLOAD_DIR, CACHE_DIR, PACKET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MODEL = os.getenv("REIMBURST_MODEL", "claude-sonnet-5-5").strip()
# "low" | "medium" | "high". Medium is a good balance of care and speed for receipts.
EFFORT = os.getenv("REIMBURST_EFFORT", "medium").strip()

# Offline mode: no API calls. Uses saved answers for the bundled sample receipts
# (samples/truth.json) and the result cache. Good for testing the app with no key.
OFFLINE = os.getenv("REIMBURST_OFFLINE", "0").strip() in ("1", "true", "yes")
# Cache Claude's answers by file hash so a re-upload is instant and free.
USE_CACHE = os.getenv("REIMBURST_CACHE", "1").strip() in ("1", "true", "yes")

# List prices in USD per million tokens (input, output, cache read).
# Source: https://platform.claude.com/docs/en/about-claude/pricing (Nov 2026)
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}

# Image limits for current models: 2,576 px long edge, 4,784 visual tokens,
# one token per 28x28 px patch. We shrink to fit so we never pay for pixels
# the model would throw away.
MAX_LONG_EDGE = 2576
MAX_VISUAL_TOKENS = 4784

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

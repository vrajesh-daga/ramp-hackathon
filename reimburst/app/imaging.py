"""Image prep and the fraud checks that don't need a model.

Everything here is plain code on purpose: hashes and metadata are facts about
the file, so we never ask Claude about them. (Anthropic's vision docs say
Claude can't reliably tell whether an image is AI-generated.)
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

from PIL import Image, ImageOps

from .config import MAX_LONG_EDGE, MAX_VISUAL_TOKENS

try:  # iPhones shoot HEIC; Claude only accepts JPEG, PNG, GIF and WebP.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover
    HEIC_SUPPORTED = False


class ImageError(ValueError):
    """The upload isn't an image we can read."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(original: bytes, out_path: Path) -> Image.Image:
    """Rotate per EXIF, convert to RGB, shrink to the model's limits, save as JPEG."""
    try:
        img = Image.open(io.BytesIO(original))
        img.load()
    except Exception as exc:
        hint = "" if HEIC_SUPPORTED else " (HEIC needs the pillow-heif package)"
        raise ImageError(f"Couldn't open this file as an image{hint}.") from exc

    img = ImageOps.exif_transpose(img)  # phones store rotation as a tag, not pixels
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    # Shrink so (a) the long edge is at most 2,576 px and (b) the image costs at
    # most ~4,784 visual tokens (one token per 28x28 px patch). The 0.97 factor
    # leaves room for the per-patch rounding up.
    scale = min(
        1.0,
        MAX_LONG_EDGE / max(w, h),
        math.sqrt(0.97 * MAX_VISUAL_TOKENS * 28 * 28 / (w * h)),
    )
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return img


def dhash(img: Image.Image, size: int = 8) -> str:
    """Difference hash: a 64-bit fingerprint of the image's brightness pattern.

    Shrink to 9x8 grayscale, then record for each pixel whether it's brighter
    than its right-hand neighbour. Two photos of the same receipt give mostly
    the same bits even after re-compression or small shifts in lighting, so a
    small Hamming distance between fingerprints means "probably the same picture".
    """
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    """Number of differing bits between two hex fingerprints."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# Byte patterns that generative tools write into metadata (EXIF, XMP, PNG text
# chunks, or C2PA Content Credentials). "trainedAlgorithmicMedia" is the IPTC
# standard term for "made by a generative model". A screenshot strips all of
# this, so a missing marker proves nothing; only a present one counts.
AI_MARKERS = [
    b"compositewithtrainedalgorithmicmedia",
    b"trainedalgorithmicmedia",
    b"midjourney",
    b"dall-e",
    b"dall\xc2\xb7e",
    b"stable diffusion",
    b"adobe firefly",
    b"gpt-image",
    b"chatgpt",
    b"imagen by google",
]


def find_ai_marker(original: bytes) -> str:
    """Return the first generative-AI marker found in the file's bytes, or ''."""
    haystack = original.lower()
    for marker in AI_MARKERS:
        if marker in haystack:
            return marker.decode("utf-8", errors="replace")
    return ""

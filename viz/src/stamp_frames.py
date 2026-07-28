#!/usr/bin/env python3
"""Burn a persistent category mark into a rendered frame sequence.

📖 Category definitions are not restated here — they live in exactly one place:
   https://app.notion.com/p/3aa472a5fb6981ebaaa7cf2e996f1e8b
   This file only draws the mark that page assigns to each category.

Non-negotiable for any photoreal outdoor clip. The hardware does not exist —
nothing here has ever been built — and a photoreal clip of a self-balancing
onewheel outdoors will be read by most viewers as video of a real board. The
whole point of the asset is that it gets shared, and sharing strips every
caption the page puts around it. A card caption cannot travel with the file;
pixels can.

Deliberately *not* an ffmpeg filter at encode time: the mark has to live in the
frames, so re-encoding at a different bitrate, trimming, or exporting a still
cannot silently drop it.

    python3 viz/src/stamp_frames.py out/shuttle_outdoor --category concept

VERTICAL DELIVERIES
-------------------
Stamp the vertical frames, never the landscape ones you then crop. A mark
placed for a 16:9 frame sits where a 9:16 cut does not reach, and a `CONCEPT`
clip that arrives unmarked is worse than one that never carried a mark: the
absence reads as a claim. Placement here is driven by the frame's own shape, so
a portrait frame gets a portrait-safe position without anyone remembering to
ask for one — and the mark's box is reported and checked before a single frame
is written, so "the mark survives" is measured rather than assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delivery import CATEGORY_SLUGS, MARKS, safe_insets  # noqa: E402

AMBER = (242, 162, 74)
CLOUD = (244, 248, 247)


def _font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=1)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _layout(size_px: tuple[int, int], text: str, letterspace: bool, corner: str):
    """Geometry of the mark in pixels: (x, y, text width, type size, scrim box).

    One function, used by both the drawing and the checking, because those two
    agreeing is the whole guarantee. Two implementations of the same arithmetic
    is how a mark ends up checked in one position and drawn in another.
    """
    w, h = size_px
    _, bottom = safe_insets(w, h)
    size = max(13, round(h * 0.021))
    pad = round(h * 0.035)
    spaced = " ".join(text) if letterspace else text
    tw = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(spaced, font=_font(size))

    # Lift clear of the platform's caption band. `bottom` is 0 for a landscape
    # frame, so 16:9 assets re-stamp to exactly the pixels they shipped with.
    y = h - pad - size - round(bottom * h)
    x = pad if corner == "bl" else w - pad - tw
    box = (x - size * 0.7, y - size * 0.5, x + tw + size * 0.7, y + size * 1.55)
    return x, y, tw, size, spaced, box


def mark_box(size_px: tuple[int, int], text: str, letterspace: bool = True,
             corner: str = "bl") -> tuple[float, float, float, float]:
    """The mark's scrim rectangle in normalised coords, y from the TOP."""
    w, h = size_px
    x0, y0, x1, y1 = _layout(size_px, text, letterspace, corner)[5]
    return x0 / w, y0 / h, x1 / w, y1 / h


def mark_is_safe(size_px: tuple[int, int], text: str, letterspace: bool = True,
                 corner: str = "bl") -> bool:
    """Does the mark land where a viewer can actually read it?"""
    top, bottom = safe_insets(*size_px)
    x0, y0, x1, y1 = mark_box(size_px, text, letterspace, corner)
    return x0 >= 0.0 and x1 <= 1.0 and y0 >= top and y1 <= 1.0 - bottom


def stamp(im: Image.Image, text: str, opacity: float = 0.62,
          corner: str = "bl", colour=AMBER, letterspace: bool = True) -> Image.Image:
    """Letterspaced wordmark over a soft scrim.

    The scrim matters more than it looks: without it the mark vanishes against
    a bright sky and is perfectly legible against dark ground, which is the
    opposite of what a guarantee needs. With it the mark reads on every frame
    of every scene, which is the only property that makes it a guarantee at all.

    TWO DIFFERENT JOBS, KEPT VISUALLY DISTINCT ON PURPOSE
    -----------------------------------------------------
    Amber, letterspaced, lower-LEFT is the **Concept signature**: a disclosure
    that the picture is authored.

    A Replay's source tag is a different thing wearing similar clothes. It is a
    statement of fact about where the motion came from, not a disclosure of
    authorship, and it must not be mistaken for one or neither means anything to
    a viewer. So it is deliberately styled apart — off-white, unspaced,
    lower-RIGHT. The same styling carries a Replay's other factual labels, slow
    motion being the case that forces one: a 66.7x clip has to say so on the
    frame.
    """
    x, y, tw, size, spaced, box = _layout(im.size, text, letterspace, corner)
    f = _font(size)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(list(box), radius=size, fill=(10, 14, 18, int(150 * opacity)))
    d.text((x, y), spaced, font=f, fill=(*colour, int(255 * opacity)))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def style_for(category: str) -> dict:
    """Draw settings for a category's mark. Concept is signed; a Replay is tagged."""
    if category == "Concept":
        return dict(corner="bl", colour=AMBER, letterspace=True)
    return dict(corner="br", colour=CLOUD, letterspace=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path, help="directory of frame_*.png")
    ap.add_argument("--category", required=True, choices=sorted(CATEGORY_SLUGS),
                    help="picks the mark and its styling; see the canonical "
                         "vocabulary page linked in this file's docstring")
    ap.add_argument("--text", default=None,
                    help="override the mark text — for a Replay's extra factual "
                         "label, e.g. '66.7x SLOW MOTION'. Never to weaken a mark.")
    ap.add_argument("--check", action="store_true",
                    help="report where the mark would land and stamp nothing")
    args = ap.parse_args()

    category = CATEGORY_SLUGS[args.category]
    text = args.text or MARKS[category]
    if text is None:
        raise SystemExit(
            f"{category} carries no mark — it is a real camera pointed at a real "
            f"thing. Nothing to stamp.")

    pngs = sorted(args.frames.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no frames in {args.frames}")

    kw = style_for(category)
    size_px = Image.open(pngs[0]).size
    x0, y0, x1, y1 = mark_box(size_px, text, kw["letterspace"], kw["corner"])
    top, bottom = safe_insets(*size_px)
    safe = mark_is_safe(size_px, text, kw["letterspace"], kw["corner"])
    print(f"{category} · mark '{text}' · {size_px[0]}x{size_px[1]} · "
          f"box x {x0:.3f}..{x1:.3f} y {y0:.3f}..{y1:.3f} · "
          f"safe y {top:.3f}..{1.0 - bottom:.3f} · "
          f"{'inside the safe area' if safe else 'OUTSIDE THE SAFE AREA'}")
    if not safe:
        raise SystemExit(
            "the mark would land under the platform's interface. A mark nobody "
            "can read is worse than no mark at all, because the frame then looks "
            "deliberately unmarked — refusing to stamp.")
    if args.check:
        return 0

    for i, p in enumerate(pngs):
        stamp(Image.open(p), text, **kw).save(p)
        if i % 100 == 0:
            print(f"  {i}/{len(pngs)}")
    print(f"stamped {len(pngs)} frames in {args.frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

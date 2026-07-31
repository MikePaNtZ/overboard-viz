#!/usr/bin/env python3
"""Build the social share card (`og.png`) from a published Sim Replay frame.

The card is what Hacker News, Reddit, Slack, Discord and iMessage render when
someone shares the link, so it is seen by more people than the page is. It is
also the one published asset most likely to be screenshotted away from its
context, which is why everything here is derived rather than authored: the card
is a crop of an artifact that already shipped with a sha256 in a render
manifest, and this script refuses to run if that hash does not match.

    /path/to/python viz/src/og_card.py --poster terrain_poster.jpg \
        --manifest terrain_render_manifest.json --out og.png

WHY A CROP AND NOT A NEW RENDER
-------------------------------
Every board-riding pose track committed to this repo predates the IMU frame-map
correction (all of them land 2026-07-26, before 15:08 -0700; the fix is
`5c1d11c` at 15:08:24). Rendering a card from one of them would put a pre-fix
trajectory on the single most-shared asset we own. The rolling-terrain artifacts
published to `sim-latest` are the only board-riding imagery that post-dates the
fix, so the source is forced -- and the card inherits **Sim Replay**, which is
also what the standing quota requires, because a milestone is never carried by
Concept.

WHAT THE CROP KEEPS, AND WHY IT IS NOT A FREE CHOICE
----------------------------------------------------
The MuJoCo frame carries a HUD: an info panel and a grade readout down the left,
chart strips along the bottom, a header with the wordmark and the source tag.

The tempting framing is to crop past the panel column and show clean terrain.
That framing is wrong, and the reason is the square crop: several surfaces
render a `summary_large_image` card square and keep only the middle 52.5% of its
width. The HUD sits to the board's LEFT, so cropping around it pushes the board
to the frame edge -- precisely where a square crop throws it away. The card then
looks correct in the file and shows empty ground to a large share of the people
who actually see it, with nothing in the image to reveal the problem.

So the board is centred instead, and the HUD column comes with it: the panels
overlap the board in both axes, and no crop keeps the whole board and drops
every panel. That is a fair trade. `LOCAL GRADE 8.0% DESCENDING` is legible at
thumbnail size and is backed by the run (`max_grade_seen_pct` 7.9999...), which
is the "true at a glance" the card needs; the finer figures beside it are noise
at that size but they are not false, and they are what the page argues -- that
the runs are real and the numbers are shown rather than claimed.

What the crop does remove is the header, and with it the burnt-in `SIM REPLAY`
tag. So the tag is re-applied here through `stamp_frames.stamp` -- the same code
path every other frame in this repo uses, not a second implementation that can
drift from it. Cropping the header is not hiding part of what happened: the HUD
is an overlay drawn on top of the run rather than the run, the motion in frame
is untouched, and this is a still, not a trimmed cut. The crop box is recorded
in the sidecar manifest so the framing is checkable rather than asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from delivery import MARKS  # noqa: E402
from stamp_frames import stamp, style_for  # noqa: E402

# The `summary_large_image` size the page already declares. Not negotiable here:
# index.html carries the meta tag and this must match what it promises.
CARD_W, CARD_H = 1200, 630

# The board and rider, measured on this frame rather than eyeballed:
# x 312..458, y 175..425, centre (385, 300). Everything below is derived from it.
SUBJECT = (312, 175, 458, 425)

# Crop box in the 1280x720 source, as (left, upper, right, lower). 760x399 is
# 40:21, which is exactly 1200:630 -- so the upscale is uniform and the board is
# not subtly the wrong shape. The round-looking 762x400 is NOT this ratio; it
# stretches by a hair, which is small enough to survive review and is therefore
# worth pinning in a test rather than a comment.
#
# Centred on the subject's x (385), which is the constraint that actually
# decides this box: several surfaces crop a share card to a SQUARE, keeping only
# the middle 52.5% of its width. A composition with the board off to one side --
# the obvious framing, since the HUD sits to its left -- loses the subject
# entirely on those surfaces. `square_crop_keeps_subject` below is the check.
#
# The consequence is that the HUD column comes with it: the panels overlap the
# board in BOTH axes, so no crop keeps the whole board and drops every panel.
# Rather than fight that, the top of the box (78) sits just above the info panel
# (which starts at 90) so the panels read as a deliberate column instead of
# being sliced by the frame edge. The bottom (478) clears the chart strips,
# which begin around 545.
CROP = (5, 78, 765, 477)

CATEGORY = "Sim Replay"

# A thumbnail is the real viewing condition, so it is the real test.
THUMB_W = 400


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_hash(manifest: dict, name: str) -> str:
    for out in manifest.get("outputs", []):
        if out["name"] == name:
            return out["sha256"]
    raise SystemExit(f"{name} is not listed in the render manifest's outputs")


def mean_luma(im: Image.Image) -> float:
    grey = im.convert("L")
    return sum(grey.getdata()) / (grey.width * grey.height)


def subject_in_card(crop=CROP, subject=SUBJECT) -> tuple[float, float, float, float]:
    """Where the board lands in the finished card, in normalised coords."""
    left, upper, right, lower = crop
    sx = CARD_W / (right - left)
    sy = CARD_H / (lower - upper)
    x0, y0, x1, y1 = subject
    return ((x0 - left) * sx / CARD_W, (y0 - upper) * sy / CARD_H,
            (x1 - left) * sx / CARD_W, (y1 - upper) * sy / CARD_H)


def square_crop_keeps_subject(crop=CROP, subject=SUBJECT) -> bool:
    """Does the board survive a centre square crop?

    Several surfaces render a `summary_large_image` card square, keeping only
    the middle CARD_H/CARD_W of its width and throwing the rest away. A card
    whose subject sits outside that band is a card that shows empty ground to
    a good fraction of the people who see it, and nothing in the image itself
    reveals the problem -- which is why this is checked rather than eyeballed.
    """
    band = CARD_H / CARD_W
    lo, hi = 0.5 - band / 2, 0.5 + band / 2
    x0, _, x1, _ = subject_in_card(crop, subject)
    return x0 >= lo and x1 <= hi


def build(poster: Path, manifest_path: Path, out: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("category") != CATEGORY:
        raise SystemExit(
            f"source manifest declares category {manifest.get('category')!r}, "
            f"not {CATEGORY!r}. The card inherits its source's category -- fix "
            f"the source or the card, do not relabel it here.")

    want = expected_hash(manifest, poster.name)
    got = sha256_of(poster)
    if want != got:
        raise SystemExit(
            f"{poster.name} does not match the manifest.\n"
            f"  manifest: {want}\n  file:     {got}\n"
            f"The whole point of this card is that it is traceable to a published "
            f"run. Re-download the artifact rather than overriding this.")

    src = Image.open(poster).convert("RGB")
    if src.size != (1280, 720):
        raise SystemExit(
            f"expected a 1280x720 source, got {src.size}. CROP is expressed in "
            f"source pixels and would land somewhere else.")

    if not square_crop_keeps_subject():
        x0, _, x1, _ = subject_in_card()
        raise SystemExit(
            f"the board lands at x {x0:.3f}..{x1:.3f} of the card, outside the "
            f"middle {CARD_H / CARD_W:.3f} a square crop keeps. Re-centre CROP "
            f"on the subject.")

    card = src.crop(CROP).resize((CARD_W, CARD_H), Image.LANCZOS)

    style = style_for(CATEGORY)
    card = stamp(card, MARKS[CATEGORY], **style)

    out.parent.mkdir(parents=True, exist_ok=True)
    card.save(out, "PNG", optimize=True)

    thumb = card.resize((THUMB_W, round(THUMB_W * CARD_H / CARD_W)), Image.LANCZOS)

    return {
        "schema": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset": out.name,
        "size_px": [CARD_W, CARD_H],
        "category": CATEGORY,
        "source_tag": MARKS[CATEGORY],
        "vocabulary": manifest.get("vocabulary"),
        "derived_from": {
            "artifact": poster.name,
            "sha256": got,
            "crop_box_in_source": list(CROP),
            "source_size_px": [1280, 720],
            "release": "sim-latest",
            "repo": "MikePaNtZ/overboard",
        },
        "source": manifest.get("source"),
        "scenario": manifest.get("scenario"),
        "measured": {
            "mean_luma_0_255": round(mean_luma(card), 1),
            "thumbnail_mean_luma_0_255": round(mean_luma(thumb), 1),
            "thumbnail_width_px": THUMB_W,
        },
        "sha256": sha256_of(out),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    record = build(args.poster, args.manifest, args.out)
    sidecar = args.out.with_suffix(".card.json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n")

    print(f"wrote {args.out} ({record['size_px'][0]}x{record['size_px'][1]})")
    print(f"  category   {record['category']} / {record['source_tag']}")
    print(f"  from       {record['derived_from']['artifact']} "
          f"@ {record['source']['commit_short']}")
    print(f"  mean luma  {record['measured']['mean_luma_0_255']}")
    print(f"  sidecar    {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

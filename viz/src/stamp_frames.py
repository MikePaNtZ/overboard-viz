#!/usr/bin/env python3
"""Burn a persistent SIMULATION mark into a rendered frame sequence.

Non-negotiable for any photoreal outdoor clip. The hardware does not exist —
nothing here has ever been built — and a photoreal clip of a self-balancing
onewheel outdoors will be read by most viewers as video of a real board. The
whole point of the asset is that it gets shared, and sharing strips every
caption the page puts around it. A card caption cannot travel with the file;
pixels can.

Deliberately *not* an ffmpeg filter at encode time: the mark has to live in the
frames, so re-encoding at a different bitrate, trimming, or exporting a still
cannot silently drop it.

    python3 viz/src/stamp_frames.py out/shuttle_outdoor
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AMBER = (242, 162, 74)


def _font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=1)
            except OSError:
                continue
    return ImageFont.load_default(size)


def stamp(im: Image.Image, text: str = "SIMULATION", opacity: float = 0.62) -> Image.Image:
    """Letterspaced wordmark, lower-left, over a soft scrim.

    The scrim matters more than it looks: without it the mark vanishes against
    a bright sky and is perfectly legible against dark ground, which is the
    opposite of what a guarantee needs. With it the mark reads on every frame
    of every scene, which is the only property that makes it a guarantee at all.
    """
    w, h = im.size
    size = max(13, round(h * 0.021))
    pad = round(h * 0.035)
    f = _font(size)

    spaced = " ".join(text)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tw = d.textlength(spaced, font=f)
    x, y = pad, h - pad - size

    d.rounded_rectangle(
        [x - size * 0.7, y - size * 0.5, x + tw + size * 0.7, y + size * 1.55],
        radius=size, fill=(10, 14, 18, int(150 * opacity)))
    d.text((x, y), spaced, font=f, fill=(*AMBER, int(255 * opacity)))

    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path, help="directory of frame_*.png")
    ap.add_argument("--text", default="SIMULATION")
    args = ap.parse_args()

    pngs = sorted(args.frames.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no frames in {args.frames}")
    for i, p in enumerate(pngs):
        stamp(Image.open(p), args.text).save(p)
        if i % 100 == 0:
            print(f"  {i}/{len(pngs)}")
    print(f"stamped {len(pngs)} frames in {args.frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


CLOUD = (244, 248, 247)


def stamp(im: Image.Image, text: str = "SIMULATION", opacity: float = 0.62,
          corner: str = "bl", colour=AMBER, letterspace: bool = True) -> Image.Image:
    """Letterspaced wordmark over a soft scrim.

    The scrim matters more than it looks: without it the mark vanishes against
    a bright sky and is perfectly legible against dark ground, which is the
    opposite of what a guarantee needs. With it the mark reads on every frame
    of every scene, which is the only property that makes it a guarantee at all.

    TWO DIFFERENT JOBS, KEPT VISUALLY DISTINCT ON PURPOSE
    -----------------------------------------------------
    The default — amber, letterspaced, lower-LEFT — is the **Lane B
    signature**: a disclosure that the footage is authored. Lane A carries no
    signature at all, by construction.

    But a Lane A clip may still need a **factual label**, and slow motion is
    the case that forces it: a 66.7x clip has to say so on the frame. That is
    not a disclosure of authorship and must not be mistaken for one, or the
    lanes stop meaning anything to a viewer. So it is deliberately styled
    apart — off-white, unspaced, lower-RIGHT — and the caller has to ask for it.
    """
    w, h = im.size
    size = max(13, round(h * 0.021))
    pad = round(h * 0.035)
    f = _font(size)

    spaced = " ".join(text) if letterspace else text
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tw = d.textlength(spaced, font=f)
    y = h - pad - size
    x = pad if corner == "bl" else w - pad - tw

    d.rounded_rectangle(
        [x - size * 0.7, y - size * 0.5, x + tw + size * 0.7, y + size * 1.55],
        radius=size, fill=(10, 14, 18, int(150 * opacity)))
    d.text((x, y), spaced, font=f, fill=(*colour, int(255 * opacity)))

    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path, help="directory of frame_*.png")
    ap.add_argument("--text", default="SIMULATION")
    ap.add_argument("--label", action="store_true",
                    help="factual Lane A label (off-white, lower-right) rather "
                         "than the Lane B signature")
    args = ap.parse_args()

    pngs = sorted(args.frames.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no frames in {args.frames}")
    kw = dict(corner="br", colour=CLOUD, letterspace=False) if args.label else {}
    for i, p in enumerate(pngs):
        stamp(Image.open(p), args.text, **kw).save(p)
        if i % 100 == 0:
            print(f"  {i}/{len(pngs)}")
    print(f"stamped {len(pngs)} frames in {args.frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

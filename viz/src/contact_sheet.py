#!/usr/bin/env python3
"""Tile the rendered variants into the single labelled PNG that goes to the owner.

This is the whole review artefact described in V1 §7. It exists because the
owner has said plainly that he cannot debug a render or give implementation
guidance — so he is never asked an open question. He gets one image, and he
answers with a cell ID plus a direction on a fixed vocabulary. A ballot, not a
canvas.

The layout is therefore not decoration: one row per variable, one variable per
row, so "A2, but warmer" is a complete and unambiguous instruction.

    ~/projects/overboard/.venv/bin/python viz/src/contact_sheet.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

ROWS = [
    ("A", "LIGHTING — where the garage door is, and how bright",
     ["A1", "A2", "A3"]),
    ("B", "FRAMING — lens compression, subject size held constant",
     ["B1", "B2", "B3"]),
    ("C", "MATERIAL — how polished vs. worn the board reads",
     ["C1", "C2", "C3"]),
]

BALLOT = ("Reply with one cell ID  +  any of:   brighter / darker  ·  "
          "tighter / wider  ·  more blur / less blur  ·  warmer / cooler")

PAD, GAP, HDR, CAP, TITLE = 34, 14, 74, 34, 118
BG, FG, DIM, ACCENT = (18, 20, 24), (238, 240, 243), (150, 156, 165), (242, 162, 74)


def _font(size: int, bold: bool = False):
    for p in ("/System/Library/Fonts/SFNSDisplay.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=1 if bold and p.endswith("ttc") else 0)
            except OSError:
                continue
    return ImageFont.load_default(size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "out/variants")
    ap.add_argument("--out", type=Path, default=ROOT / "out/V1.0_contact_sheet.png")
    ap.add_argument("--cell-width", type=int, default=760)
    args = ap.parse_args()

    # Import lazily so this script does not need Blender on the path.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bs", ROOT / "viz/src/build_scene.py")
    labels = {}
    try:  # build_scene imports bpy at module level, so read the labels textually
        for line in (ROOT / "viz/src/build_scene.py").read_text().splitlines():
            if 'label="' in line and line.strip().startswith('"'):
                key = line.strip().split('"')[1]
                labels[key] = line.split('label="')[1].split('"')[0]
    except OSError:
        pass

    first = Image.open(args.src / "A1.png")
    cw = args.cell_width
    ch = round(cw * first.height / first.width)

    grid_w = 3 * cw + 2 * GAP
    W = grid_w + 2 * PAD
    H = TITLE + len(ROWS) * (HDR + ch + CAP + GAP) + PAD + 46

    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)

    d.text((PAD, 26), "OVERBOARD — V1.0 garage hero shot", font=_font(30, True), fill=FG)
    d.text((PAD, 62), "EEVEE preview · pick one cell", font=_font(17), fill=DIM)

    y = TITLE
    for _, heading, keys in ROWS:
        d.text((PAD, y + 20), heading, font=_font(19, True), fill=ACCENT)
        y += HDR
        for i, k in enumerate(keys):
            p = args.src / f"{k}.png"
            if not p.exists():
                continue
            x = PAD + i * (cw + GAP)
            sheet.paste(Image.open(p).convert("RGB").resize((cw, ch), Image.LANCZOS), (x, y))
            d.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(60, 64, 70))
            # The cell ID is the thing the owner types back, so it gets to be
            # the most legible element on the sheet.
            d.rectangle([x, y, x + 54, y + 30], fill=ACCENT)
            d.text((x + 14, y + 5), k, font=_font(20, True), fill=(18, 20, 24))
            d.text((x + 2, y + ch + 8), labels.get(k, k), font=_font(16), fill=DIM)
        y += ch + CAP + GAP

    d.text((PAD, H - 40), BALLOT, font=_font(17), fill=FG)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Delivery constants shared by the renderer and the stamp pass.

Stdlib only, and deliberately so: `render_clip.py` runs inside Blender (which
has no Pillow) and `stamp_frames.py` runs under system python (which has no
`bpy`). Neither can import the other, so the two facts they must agree on live
here instead of being typed twice and drifting.

Those two facts are the category marks and the vertical safe area. Both are
"agree or the guarantee breaks" values: a renderer that thinks the safe area is
the bottom fifth and a stamp pass that thinks it is the bottom tenth will place
a `CONCEPT` mark underneath a platform caption, and the frame then looks
deliberately unmarked — which is worse than carrying no mark at all.
"""
from __future__ import annotations

# ------------------------------------------------------------------ categories
#
# 📖 Definitions live in exactly one place and are not restated here:
#    https://app.notion.com/p/3aa472a5fb6981ebaaa7cf2e996f1e8b
#
# This is only the text each category burns into the frame. `Footage` is a real
# camera pointed at a real thing and carries no mark, which is why its value is
# None rather than an empty string: nothing to draw is a different state from
# "draw nothing", and the stamp pass refuses the first.
MARKS = {
    "Footage": None,
    "Sim Replay": "SIM",
    "Hardware Replay": "HARDWARE",
    "Concept": "CONCEPT",
}

# CLI spelling → canonical name, so `--category sim-replay` is typeable.
CATEGORY_SLUGS = {c.lower().replace(" ", "-"): c for c in MARKS}

# ------------------------------------------------------------------ safe area
#
# Short-form video is watched vertically on a surface that draws its own
# interface over the picture. Two bands are effectively not ours:
#
#   · the top eighth   — account row, progress bar, feed tabs
#   · the bottom fifth — caption, handle, sound row, the action rail's lower half
#
# Exact pixel counts differ per platform and per app release, so these are
# generous round fractions rather than a table that goes stale. Nothing that has
# to be READ may sit in those bands — not the subject, and not the mark.
SAFE_TOP, SAFE_BOTTOM = 0.125, 0.20
SAFE_FRACTION = 1.0 - SAFE_TOP - SAFE_BOTTOM        # 0.675 of the frame height

# 16:9 is the reference framing. A vertical cut is defined from it.
ASPECTS = {"16:9": (1920, 1080), "9:16": (1080, 1920)}
REFERENCE_ASPECT = "16:9"


def vertical_sensor_height(sensor_width_mm: float) -> float:
    """Sensor height that makes a 9:16 render a re-frame of the 16:9 shot.

    Kept here, free of `bpy`, so the arithmetic behind the vertical framing can
    be tested without launching Blender — and so the one number it depends on
    (`SAFE_FRACTION`) is the same number the stamp pass uses.

    Blender's default `sensor_fit='AUTO'` applies the lens's field of view to
    whichever image dimension is larger, so rendering the same camera at
    1080x1920 swings a 50 mm lens onto the vertical axis: 39.6 degrees of
    vertical view instead of 22.9. That is not a crop of the landscape shot, it
    is a different and much wider shot in which the board is a speck near the
    horizon — and it is exactly what you get by only changing `resolution_y`.

    So the vertical frame is defined *from* the landscape frame:

        the 16:9 picture, whole and uncropped, placed in the safe area.

    Divide the 16:9 sensor height by the safe fraction and the arithmetic falls
    out at 36 x (9/16) / 0.675 = 30.0 mm. Every direction the landscape frame
    contained now lands between 12.5% and 80% of the vertical frame's height,
    and the bands the interface covers are filled with real rendered sky and
    ground rather than with anything that has to be seen.
    """
    ref_w, ref_h = ASPECTS[REFERENCE_ASPECT]
    return sensor_width_mm * (ref_h / ref_w) / SAFE_FRACTION


def safe_insets(width: int, height: int) -> tuple[float, float]:
    """Safe-area insets for a frame of this shape, as fractions of its height.

    Landscape gets (0, 0). That is not an oversight: the landscape deliverable
    plays in a player that does not overlay it, and applying vertical insets to
    it would move a mark that has already shipped in that position — changing
    output nobody asked to change.
    """
    return (SAFE_TOP, SAFE_BOTTOM) if height > width else (0.0, 0.0)

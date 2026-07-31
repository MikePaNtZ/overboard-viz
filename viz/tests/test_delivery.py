#!/usr/bin/env python3
"""Tests for the vertical delivery: safe area, mark placement, mark survival.

Runs under any python with Pillow — no Blender, no GPU, no render. That is the
point: the arithmetic that decides whether a `CONCEPT` mark ends up under a
platform caption should be checkable in a second, by anyone, without a 3D
application.

    python3 viz/tests/test_delivery.py        (or: pytest viz/tests)
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from delivery import (  # noqa: E402
    ASPECTS, CATEGORY_SLUGS, MARKS, REFERENCE_ASPECT, SAFE_BOTTOM, SAFE_FRACTION,
    SAFE_TOP, safe_insets, vertical_sensor_height, vertical_shift_y)
from stamp_frames import mark_box, mark_is_safe, style_for  # noqa: E402
import og_card  # noqa: E402

LANDSCAPE = ASPECTS[REFERENCE_ASPECT]
VERTICAL = ASPECTS["9:16"]
MARKED = [c for c, m in MARKS.items() if m]


def test_vertical_frame_is_the_landscape_frame_inside_the_safe_area():
    """The whole 16:9 picture, uncropped vertically, lands in the safe band.

    Sensor height sets the band's size and the lens shift sets its position. Get
    only the first right and the composition is the correct height but rides low
    — which is how a board's contact patch ends up under a caption while the
    numbers still look approximately correct.
    """
    sensor_w = 36.0
    ref_w, ref_h = LANDSCAPE
    landscape_sensor_h = sensor_w * ref_h / ref_w

    vertical_sensor_h = vertical_sensor_height(sensor_w)
    assert abs(vertical_sensor_h - 30.0) < 1e-9, vertical_sensor_h

    # The landscape frame's angular height as a fraction of the vertical one.
    band = landscape_sensor_h / vertical_sensor_h
    assert abs(band - SAFE_FRACTION) < 1e-12

    # ... and, once shifted, it occupies exactly [SAFE_BOTTOM, 1 - SAFE_TOP].
    centre = 0.5 - vertical_shift_y()
    assert abs((centre - band / 2) - SAFE_BOTTOM) < 1e-12
    assert abs((centre + band / 2) - (1.0 - SAFE_TOP)) < 1e-12


def test_naive_vertical_render_would_be_a_wider_shot_not_a_crop():
    """Guards the mistake this whole change exists to avoid.

    Only changing `resolution_y` leaves Blender's AUTO sensor fit to apply the
    lens to the larger image dimension, which on a portrait frame is its height.
    The vertical field of view then grows instead of holding, and the subject
    shrinks. Asserted as arithmetic so the claim in the docstrings is checkable.
    """
    sensor_w = 36.0
    ref_w, ref_h = LANDSCAPE
    naive_vertical_sensor_h = sensor_w                 # AUTO fit, portrait frame
    correct = vertical_sensor_height(sensor_w)
    assert naive_vertical_sensor_h > correct
    # The naive frame sees 20% more vertically than even our (deliberately
    # widened) one, and 60% more than the landscape shot it came from.
    assert naive_vertical_sensor_h / correct > 1.19
    assert naive_vertical_sensor_h / (sensor_w * ref_h / ref_w) > 1.7


def test_safe_insets_apply_to_portrait_only():
    assert safe_insets(*VERTICAL) == (SAFE_TOP, SAFE_BOTTOM)
    assert safe_insets(*LANDSCAPE) == (0.0, 0.0)
    assert safe_insets(1080, 1080) == (0.0, 0.0)       # square: no overlay bands


def test_every_marked_category_lands_inside_the_vertical_safe_area():
    """AC3, measured. Every mark this repo can burn, in the delivery shape."""
    for category in MARKED:
        kw = style_for(category)
        assert mark_is_safe(VERTICAL, MARKS[category], kw["letterspace"],
                            kw["corner"]), category
        x0, y0, x1, y1 = mark_box(VERTICAL, MARKS[category], kw["letterspace"],
                                  kw["corner"])
        assert y1 <= 1.0 - SAFE_BOTTOM, (category, y1)
        assert y0 >= SAFE_TOP, (category, y0)
        assert 0.0 <= x0 and x1 <= 1.0, (category, x0, x1)


def test_footage_has_no_mark_and_that_is_a_distinct_state():
    assert MARKS["Footage"] is None
    assert "footage" in CATEGORY_SLUGS
    assert CATEGORY_SLUGS["sim-replay"] == "Sim Replay"
    assert CATEGORY_SLUGS["hardware-replay"] == "Hardware Replay"


def test_a_mark_stamped_on_the_landscape_frame_does_not_survive_a_crop():
    """The reason the stamp pass must run on the vertical frames, not before.

    This is the failure the acceptance criterion names: a `CONCEPT` mark that is
    cropped out is worse than no mark, because the result looks deliberately
    unmarked. It is asserted here rather than trusted, because the tempting
    pipeline — render 16:9 once, stamp once, crop for each channel — silently
    produces exactly it.

    A centre 9:16 crop of a 16:9 frame keeps the middle 31.6% of its width. Both
    corner placements sit outside that, by a wide margin.
    """
    keep = (9 / 16) / (16 / 9)                          # 0.3164 of the width
    lo, hi = 0.5 - keep / 2, 0.5 + keep / 2
    for category in MARKED:
        kw = style_for(category)
        x0, _, x1, _ = mark_box(LANDSCAPE, MARKS[category], kw["letterspace"],
                                kw["corner"])
        survives = x0 >= lo and x1 <= hi
        assert not survives, (
            f"{category}: this test's premise no longer holds — the landscape "
            f"mark now falls inside the crop, so the warning it encodes is stale")


def test_landscape_mark_position_is_unchanged_by_this_work():
    """AC4's other half: nothing already shipped in 16:9 moves.

    The pre-change formula, transcribed. Landscape safe insets are zero, so the
    two must agree exactly — if they ever stop agreeing, an asset that shipped
    with the mark in one place would re-stamp with it in another.
    """
    w, h = LANDSCAPE
    size = max(13, round(h * 0.021))
    pad = round(h * 0.035)
    expected_y = h - pad - size                          # no inset term
    x0, y0, x1, y1 = mark_box(LANDSCAPE, "CONCEPT", True, "bl")
    assert abs(y0 * h - (expected_y - size * 0.5)) < 1e-9
    assert abs(y1 * h - (expected_y + size * 1.55)) < 1e-9
    assert abs(x0 * w - (pad - size * 0.7)) < 1e-9


def test_the_share_card_survives_a_centre_square_crop():
    """The board stays in frame on surfaces that crop the card square.

    A share card is the most-seen asset the project publishes and the one whose
    framing nobody re-checks, because the failure is invisible in the file
    itself: the 1200x630 looks correct and only the square rendering is empty
    ground. The obvious composition walks straight into it — the HUD column sits
    to the board's left, so framing "around the HUD" pushes the subject to the
    edge, exactly where a square crop discards it.
    """
    assert og_card.square_crop_keeps_subject()

    band = og_card.CARD_H / og_card.CARD_W
    x0, _, x1, _ = og_card.subject_in_card()
    assert x0 >= 0.5 - band / 2
    assert x1 <= 0.5 + band / 2


def test_the_share_card_crop_is_the_card_aspect_so_nothing_stretches():
    """Crop and card must agree on aspect, or the board is subtly wrong-shaped.

    An anisotropic resize is the kind of defect that survives review: a board
    2% too tall reads as a rendering choice, not a bug, and it ships.
    """
    left, upper, right, lower = og_card.CROP
    assert (right - left) / (lower - upper) == og_card.CARD_W / og_card.CARD_H


def test_the_share_card_is_a_replay_and_says_which_source():
    """The card inherits Sim Replay, and a Replay always names its source.

    The crop removes the source tag burnt into the frame's corner, so the card
    would otherwise ship as an unmarked Replay — the one state the category rule
    does not allow.
    """
    assert og_card.CATEGORY in MARKS
    assert MARKS[og_card.CATEGORY] == "SIM"

    style = style_for(og_card.CATEGORY)
    size = (og_card.CARD_W, og_card.CARD_H)
    assert mark_is_safe(size, MARKS[og_card.CATEGORY], style["letterspace"],
                        style["corner"])


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

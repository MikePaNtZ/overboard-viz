#!/usr/bin/env python3
"""Encode web derivatives of published Sim Replay clips, with provenance.

The landing page cannot use the CI artifacts directly: they are encoded for
inspection, not for a visitor on a phone. This turns them into what the page
needs -- H.264, a poster per clip, `faststart` so playback begins before the
file has arrived -- without re-rendering anything, so the delivered clip is the
published run rather than a lookalike.

Deliberately mp4-only, matching what is already in `overboard-web/assets/`. A
VP9 companion was built and thrown away: it came out LARGER than the H.264 on
this content, because the published `.webm` is already VP9 and re-encoding it is
a second generation of the same lossy transform. A bigger file in a format
offered as the smaller one is worse than not offering it.

    /path/to/python viz/src/web_derivatives.py --release-dir DIR --out-dir DIR

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No re-cut, no re-grade, no camera change, no trim. Every clip is transcoded
whole. That is not conservatism about effort: a Sim Replay's guarantee is that
a reader can reproduce the motion from the committed track and scene, and a
trim that removes part of what happened breaks it. Re-encoding does not -- it
changes how the picture is stored, not what it shows.

The source tag is burnt into the frames upstream, so it survives transcoding
here. Nothing in this file draws a mark, and nothing in this file should: a
second implementation of the mark is a second thing that can drift.

EVERY INPUT IS HASH-CHECKED against the render manifest before it is touched.
The failure this prevents is specific and has already been caught once on this
project: a clip from a run that predates the IMU frame-map fix shows one thing
while the copy beside it claims another. A filename cannot tell you which run
it came from; a sha256 can.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Web naming follows what is already in overboard-web/assets/: `sim-<thing>.mp4`
# with a matching `-poster.jpg`. Matching it matters more than it looks -- the
# page's markup and its CI check are written against that shape.
CLIPS = {
    "sim-terrain-ride": {
        "video": "terrain_ride.mp4",
        # The ride's poster already shipped as its own artifact, so it is copied
        # rather than re-extracted: a poster pulled at a slightly different
        # timestamp would be a third version of a frame that already exists.
        "poster_artifact": "terrain_poster.jpg",
        "poster_at_s": None,
        "summary": "rolling terrain, 8% peak grade, closed loop, estimator on. "
                   "Crest to dip to the next crest, no strike.",
    },
    "sim-terrain-compare": {
        "video": "terrain_compare.mp4",
        "poster_artifact": None,
        # The last second of the clip, where both panes carry their outcome
        # banner: truth at the next crest, estimate nose-down at 3.28 s. The
        # whole argument of the clip is legible in this one frame, which is
        # exactly what a poster has to do.
        "poster_at_s": 13.6,
        # Worded carefully. The manifest's `struck_phase` says "descent", but
        # that is the scenario classifier bucketing negative travel; the run
        # never reaches the descent. Both drift backwards off the start crest
        # during the 2 s settle, truth recovers and rides on, the estimate puts
        # the nose in while still behind the start. Captioning it as a descent
        # failure describes something the footage does not show.
        "summary": "the same 10% roller, same controller, run twice with only "
                   "the attitude source differing. Truth pitch reaches the next "
                   "crest at 24 m; the real estimator nose-strikes at 3.28 s, "
                   "0.87 m behind the start crest, before the descent begins.",
        "caption_warning": "Do not caption as a failure 'on the descent'. The "
                           "manifest's struck_phase says descent; that is a "
                           "classifier artefact for negative travel.",
    },
}

# CRF 23, which is quality-first on purpose. The obvious lever does almost
# nothing here: CRF 26/28/30 land at 2.3/2.0/2.0 MB against 2.3 MB at 23,
# because the terrain's fine hatched shading is high-frequency detail the
# encoder cannot cheaply discard. Since the page loads these with
# `preload="none"`, nothing is fetched until a visitor presses play, so trading
# visible quality for a tenth of a megabyte buys nothing.
#
# `faststart` moves the index to the front of the file: without it a browser
# downloads the whole clip before the first frame appears, which reads as a
# broken player rather than a slow one.
H264 = ["-c:v", "libx264", "-profile:v", "high", "-crf", "23", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, manifest: dict) -> str:
    """Refuse to use an artifact the manifest does not vouch for."""
    want = None
    for out in manifest.get("outputs", []):
        if out["name"] == path.name:
            want = out["sha256"]
    if want is None:
        raise SystemExit(
            f"{path.name} is not in the render manifest's outputs, so there is "
            f"nothing to check it against. Delivering it would be asserting its "
            f"provenance rather than showing it.")
    got = sha256_of(path)
    if got != want:
        raise SystemExit(
            f"{path.name} does not match the manifest.\n"
            f"  manifest: {want}\n  file:     {got}\n"
            f"Re-download from the release rather than overriding this.")
    return got


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{proc.stderr[-2000:]}")


def build(release_dir: Path, out_dir: Path) -> list[dict]:
    manifest = json.loads((release_dir / "terrain_render_manifest.json").read_text())
    if manifest.get("category") != "Sim Replay":
        raise SystemExit(
            f"source declares {manifest.get('category')!r}. These derivatives "
            f"inherit their source's category; they do not relabel it.")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for stem, spec in CLIPS.items():
        src = release_dir / spec["video"]
        src_hash = verify(src, manifest)

        mp4 = out_dir / f"{stem}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *H264, str(mp4)])

        poster = out_dir / f"{stem}-poster.jpg"
        if spec["poster_artifact"]:
            pa = release_dir / spec["poster_artifact"]
            verify(pa, manifest)
            shutil.copyfile(pa, poster)
        else:
            run(["ffmpeg", "-v", "error", "-y", "-ss", str(spec["poster_at_s"]),
                 "-i", str(src), "-frames:v", "1", "-q:v", "3", str(poster)])

        records.append({
            "schema": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "asset_stem": stem,
            "category": manifest["category"],
            "source_tag": manifest["source_tag"],
            "vocabulary": manifest.get("vocabulary"),
            "scenario": manifest.get("scenario"),
            "summary": spec["summary"],
            # Only present where a clip's metrics can be read to say something
            # the footage does not show. It travels with the file because the
            # role that writes the caption is not the role that watched the run.
            **({"caption_warning": spec["caption_warning"]}
               if spec.get("caption_warning") else {}),
            "source": manifest.get("source"),
            "derived_from": {
                "release": "sim-latest",
                "repo": "MikePaNtZ/overboard",
                "video": {"artifact": spec["video"], "sha256": src_hash},
                "poster": (
                    {"artifact": spec["poster_artifact"]} if spec["poster_artifact"]
                    else {"extracted_from": spec["video"], "at_s": spec["poster_at_s"]}
                ),
            },
            "transform": "transcode only -- no re-cut, re-grade, camera change or trim",
            "delivered": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha256_of(p)}
                for p in (mp4, poster)
            },
            "page_constraints": [
                "no autoplay", "no loop", 'preload="none"',
                "own poster per clip", "no CSS filter on the video",
            ],
        })

        sidecar = out_dir / f"{stem}.provenance.json"
        sidecar.write_text(json.dumps(records[-1], indent=2) + "\n")

    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    for rec in build(args.release_dir, args.out_dir):
        print(f"{rec['asset_stem']}  ({rec['category']} / {rec['source_tag']}) "
              f"@ {rec['source']['commit_short']}")
        for name, d in rec["delivered"].items():
            print(f"    {name:34s} {d['bytes'] / 1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill provenance for clips published before provenance existed.

Three clips are live on the site with no record of the run they replay:
`sim-balance.mp4`, `sim-inner-loop.mp4`, `sim-shuttle.mp4`. This writes a
`.provenance.json` beside each, in the same shape `web_derivatives.py` emits.

    /path/to/python viz/src/backfill_provenance.py --web-assets DIR --out-dir DIR

THE POINT OF THIS FILE IS WHAT IT REFUSES TO CLAIM
--------------------------------------------------
A provenance record that is confidently wrong is worse than the silence it
replaces: silence prompts someone to check, and a wrong record stops them. So
each field below is either **derived from something checkable** or explicitly
marked `undetermined` with the reason.

What is checkable, and is asserted:

  · The published clip is byte-identical to a master in `masters/`. Verified by
    sha256 at runtime, not asserted from a filename -- and the names do not
    match, so a filename would have been misleading. `sim-balance.mp4` is the
    IMPULSE clip.
  · Which `.otrk` it replays, by scenario, with the committed track's blob sha.
  · The model the run used, by the `model_sha256` inside the track manifest.
    That value uniquely identifies one of the six versions of
    `sim/models/overboard_onewheel.xml` in the controls repo's history.

What is NOT checkable, and is therefore declared undetermined:

  · **The controls commit the run was made from.** These tracks predate viz#15,
    which added `source.controls` to the manifest, so the commit was never
    recorded. It also cannot be recovered after the fact: the controls repo
    squash-merges, so a model version's arrival on master is the merge, not the
    authorship, and the branch commits it came from no longer exist.

    The impulse and closed-loop tracks make this concrete and are the reason
    this is spelled out rather than hand-waved: both were committed on
    2026-07-26 (00:29 and 01:15) recording a model that did not reach master
    until 08:42 the same day. That is not a contradiction -- it is what
    squash-merging looks like from the outside -- but it does mean **neither
    timestamp dates the run**, and any attempt to pin a commit from dates alone
    would be guessing with a confident face on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MASTERS = REPO / "masters"
SCENES = REPO / "viz" / "scenes"

# published name -> (master it is byte-identical to, the track it replays)
#
# The published names are not the master names and in one case actively mislead:
# `sim-balance.mp4` is the impulse/topple clip. index.html says so in a comment
# and keeps the name deliberately, because the analytics id has funnel history.
# Identification here is by sha256, so the naming cannot mislead this script.
CLIPS = {
    "sim-balance": ("V1.1_impulse_clip_web_G2_cycles.mp4", "impulse"),
    "sim-inner-loop": ("V1.1_closed_loop_web_cycles.mp4", "closed_loop"),
    "sim-shuttle": ("V1.3_shuttle_dusk_waterfront.mp4", "shuttle_run"),
}

UNDETERMINED = {
    "controls_commit": "undetermined",
    "why": (
        "The track predates viz#15, which added `source.controls` to the .otrk "
        "manifest, so the commit was never recorded. It cannot be recovered "
        "afterwards either: the controls repo squash-merges, so a model "
        "version's first appearance on master is its merge, not its "
        "authorship, and the branch commits are gone. Dates do not close the "
        "gap -- the impulse and closed-loop tracks were committed hours BEFORE "
        "the model they name reached master."
    ),
    "how_to_stop_needing_this": (
        "Nothing published after viz#15 has this problem; new .otrk manifests "
        "carry source.controls with a dirty flag. This is backlog, not a "
        "recurring gap."
    ),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def track_manifest(scenario: str) -> dict:
    import numpy as np
    d = np.load(SCENES / f"{scenario}.otrk.npz", allow_pickle=True)
    raw = d["manifest"]
    return json.loads(str(raw) if raw.dtype.kind in "US" else raw.item())


def build(web_assets: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for stem, (master_name, scenario) in CLIPS.items():
        published = web_assets / f"{stem}.mp4"
        master = MASTERS / master_name

        pub_hash, master_hash = sha256_of(published), sha256_of(master)
        if pub_hash != master_hash:
            raise SystemExit(
                f"{published.name} is NOT byte-identical to {master_name}\n"
                f"  published: {pub_hash}\n  master:    {master_hash}\n"
                f"The identification this record rests on does not hold. Do not "
                f"weaken the check -- find out which master it really is.")

        tm = track_manifest(scenario)
        track_path = f"viz/scenes/{scenario}.otrk.npz"

        records.append({
            "schema": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "asset_stem": stem,
            "backfilled": True,
            "backfill_note": (
                "Published before this repo emitted provenance. Reconstructed "
                "from evidence; every field is either checkable or marked "
                "undetermined."
            ),
            "category": "Sim Replay",
            "source_tag": "SIM",
            "vocabulary": "https://app.notion.com/p/3aa472a5fb6981ebaaa7cf2e996f1e8b",
            "scenario": tm["source"].get("scenario", scenario),
            "identified": {
                "master": {
                    "repo": "MikePaNtZ/overboard-viz",
                    "path": f"masters/{master_name}",
                    "sha256": master_hash,
                    "how": "byte-identical to the published file, verified by sha256",
                },
                "track": {
                    "path": track_path,
                    "blob_sha": git("rev-parse", f"HEAD:{track_path}"),
                    "committed": git("log", "-1", "--format=%aI", "--", track_path),
                    "exporter_version": tm["source"].get("exporter_version"),
                },
                "model": {
                    "model_file": tm["source"].get("model_file"),
                    "model_sha256": tm["source"].get("model_sha256"),
                    "note": (
                        "Uniquely identifies one of six versions of the model in "
                        "the controls repo's history. It identifies the model "
                        "CONTENT, not the commit the run was made from."
                    ) if tm["source"].get("model_sha256") else (
                        "This scenario builds its model programmatically "
                        "(sim.scenarios.plant.build_model), so there is no model "
                        "file to hash and this cannot be pinned at all."
                    ),
                },
                "mujoco_version": tm["source"].get("mujoco_version"),
            },
            "source": UNDETERMINED,
            "transform": (
                "Blender/Cycles render of the committed .otrk through the "
                "committed scene, then encoded for web. No re-cut or retime."
            ),
            "delivered": {
                published.name: {"bytes": published.stat().st_size,
                                 "sha256": pub_hash},
            },
            "page_constraints": [
                "no autoplay", "no loop", 'preload="none"',
                "own poster per clip", "no CSS filter on the video",
            ],
        })

        (out_dir / f"{stem}.provenance.json").write_text(
            json.dumps(records[-1], indent=2) + "\n")

    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-assets", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    for r in build(args.web_assets, args.out_dir):
        ident = r["identified"]
        print(f"{r['asset_stem']:18s} = {ident['master']['path']}")
        print(f"{'':18s}   track {ident['track']['path']}  "
              f"model {(ident['model']['model_sha256'] or 'n/a')[:12]}")
        print(f"{'':18s}   controls commit: {r['source']['controls_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
